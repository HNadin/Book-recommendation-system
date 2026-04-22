"""
Management command: python manage.py train_models

Trains all model variants offline and saves weights/artefacts to
bookflix/trained_models/.  Run this once after importing data;
web requests then load pre-trained models in milliseconds.

Steps
-----
1. Load ratings from DB; apply 80/20 stratified train/test split.
2. Apply VADER sentiment correction to train ratings (Section 2.5).
3. Train NCF on corrected train ratings (Section 2.3).
4. Train SVD on same train split (baseline for comparison, Section 3.5).
5. Fit LSA semantic embedder on all books (Section 2.2).
6. Evaluate all model variants on test split; save metrics JSON.
"""

import json
import logging
import os
import pickle
import random

import numpy as np
import pandas as pd
from django.core.management.base import BaseCommand

from bookflix.ml.model_store import MODELS_DIR, save_eval_results, _path
from bookflix.ml.embeddings import LSAEmbedder
from bookflix.ml.sentiment import build_book_sentiment_map, apply_sentiment_correction
from bookflix.ml.evaluation import (
    compute_rmse,
    compute_precision_at_k,
    compute_ndcg_at_k,
    RELEVANCE_THRESHOLD,
)
from bookflix.recommendation_algorithms import load_data_ratings, load_books_df, train_test_split

logger = logging.getLogger(__name__)
SEED = 42
random.seed(SEED)
np.random.seed(SEED)


class Command(BaseCommand):
    help = "Train NCF, SVD baseline, and LSA embedder; save artefacts + evaluation metrics."

    def add_arguments(self, parser):
        parser.add_argument(
            "--epochs", type=int, default=10,
            help="NCF training epochs (default: 10)"
        )
        parser.add_argument(
            "--embedding-dim", type=int, default=32,
            help="NCF embedding dimension (default: 32)"
        )
        parser.add_argument(
            "--batch-size", type=int, default=1024,
            help="NCF batch size (default: 1024)"
        )
        parser.add_argument(
            "--lr", type=float, default=1e-3,
            help="NCF learning rate (default: 0.001)"
        )
        parser.add_argument(
            "--reviews-csv", type=str, default="Reviews.csv",
            help="Path to reviews CSV for sentiment correction"
        )
        parser.add_argument(
            "--eval-k", type=int, default=10,
            help="K for Precision@K and NDCG@K (default: 10)"
        )

    def handle(self, *args, **options):
        os.makedirs(MODELS_DIR, exist_ok=True)

        self.stdout.write("=== Step 1: Loading data ===")
        ratings_df = load_data_ratings()
        books_df = load_books_df()
        self.stdout.write(f"  Ratings: {len(ratings_df):,}  |  Books: {len(books_df):,}")

        train_df, test_df = train_test_split(ratings_df, test_ratio=0.2)
        self.stdout.write(f"  Train: {len(train_df):,}  |  Test: {len(test_df):,}")

        # --- Sentiment correction ---
        self.stdout.write("=== Step 2: Sentiment correction ===")
        reviews_path = options["reviews_csv"]
        sentiment_map: dict = {}
        if os.path.exists(reviews_path):
            try:
                reviews_df = pd.read_csv(reviews_path, on_bad_lines="skip", low_memory=False)
                sentiment_map = build_book_sentiment_map(reviews_df)
                train_df = apply_sentiment_correction(train_df, sentiment_map)
                self.stdout.write(f"  Sentiment map: {len(sentiment_map):,} books corrected.")
            except Exception as e:
                self.stdout.write(self.style.WARNING(f"  Sentiment correction skipped: {e}"))
        else:
            self.stdout.write(self.style.WARNING(
                f"  {reviews_path} not found — skipping sentiment correction."
            ))

        # Save sentiment map
        with open(_path("sentiment_map.pkl"), "wb") as f:
            pickle.dump(sentiment_map, f)

        # --- LSA embedder ---
        self.stdout.write("=== Step 3: Training LSA semantic embedder ===")
        embedder = LSAEmbedder(n_components=100)
        embedder.fit(books_df)
        embedder.save(_path("lsa_embedder.pkl"))
        self.stdout.write("  LSA embedder saved.")

        # --- SVD baseline ---
        self.stdout.write("=== Step 4: Training SVD baseline ===")
        svd_model = self._train_svd(train_df)
        if svd_model:
            with open(_path("svd_baseline.pkl"), "wb") as f:
                pickle.dump(svd_model, f)
            self.stdout.write("  SVD baseline saved.")

        # --- NCF ---
        self.stdout.write("=== Step 5: Training NCF ===")
        rating_col = "adjusted_rating" if "adjusted_rating" in train_df.columns else "book_rating"
        ncf_model, meta = self._train_ncf(
            train_df,
            rating_col=rating_col,
            embedding_dim=options["embedding_dim"],
            epochs=options["epochs"],
            batch_size=options["batch_size"],
            lr=options["lr"],
        )
        if ncf_model:
            import torch
            torch.save(ncf_model.state_dict(), _path("ncf_model.pt"))
            with open(_path("ncf_meta.pkl"), "wb") as f:
                pickle.dump(meta, f)
            self.stdout.write("  NCF model saved.")

        # --- Evaluation ---
        self.stdout.write("=== Step 6: Evaluating all models ===")
        k = options["eval_k"]
        results = self._evaluate_all(
            test_df, books_df, embedder, svd_model, ncf_model, meta, k
        )
        save_eval_results(results)
        self.stdout.write("\n=== Evaluation Results ===")
        for model_name, metrics in results.items():
            self.stdout.write(f"  {model_name}:")
            for metric, val in metrics.items():
                self.stdout.write(f"    {metric}: {val:.4f}" if val is not None else f"    {metric}: N/A")

        self.stdout.write(self.style.SUCCESS("\nAll models trained and saved successfully."))

    # -----------------------------------------------------------------------

    def _train_svd(self, train_df):
        try:
            from surprise import Dataset, Reader, SVD as SurpriseSVD
        except ImportError:
            self.stdout.write(self.style.WARNING("  scikit-surprise not installed; skipping SVD."))
            return None

        rating_col = "adjusted_rating" if "adjusted_rating" in train_df.columns else "book_rating"
        reader = Reader(rating_scale=(1, 10))
        data = Dataset.load_from_df(
            train_df[["user_id", "book__isbn", rating_col]].rename(
                columns={rating_col: "book_rating"}
            ),
            reader,
        )
        trainset = data.build_full_trainset()
        svd = SurpriseSVD(n_factors=50, n_epochs=20, random_state=SEED)
        svd.fit(trainset)
        return svd

    def _train_ncf(self, train_df, rating_col, embedding_dim, epochs, batch_size, lr):
        try:
            import torch
            import torch.nn as nn
            from torch.utils.data import DataLoader, TensorDataset
        except ImportError:
            self.stdout.write(self.style.WARNING("  PyTorch not installed; skipping NCF."))
            return None, {}

        from bookflix.ml.ncf import NCF

        # Build index maps
        users = train_df["user_id"].unique().tolist()
        isbns = train_df["book__isbn"].unique().tolist()
        user_to_idx = {u: i + 1 for i, u in enumerate(users)}
        isbn_to_idx = {isbn: i + 1 for i, isbn in enumerate(isbns)}

        u_tensor = torch.tensor(
            [user_to_idx[u] for u in train_df["user_id"]], dtype=torch.long
        )
        v_tensor = torch.tensor(
            [isbn_to_idx[isbn] for isbn in train_df["book__isbn"]], dtype=torch.long
        )
        r_tensor = torch.tensor(
            train_df[rating_col].values, dtype=torch.float32
        )

        dataset = TensorDataset(u_tensor, v_tensor, r_tensor)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        model = NCF(
            num_users=len(users),
            num_items=len(isbns),
            embedding_dim=embedding_dim,
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = nn.MSELoss()

        model.train()
        for epoch in range(1, epochs + 1):
            total_loss = 0.0
            for u_b, v_b, r_b in loader:
                optimizer.zero_grad()
                pred = model(u_b, v_b)
                loss = criterion(pred, r_b)
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * len(r_b)
            avg_loss = total_loss / len(dataset)
            self.stdout.write(f"  Epoch {epoch}/{epochs}  loss={avg_loss:.4f}")

        meta = {
            "num_users": len(users),
            "num_items": len(isbns),
            "embedding_dim": embedding_dim,
            "user_to_idx": user_to_idx,
            "isbn_to_idx": isbn_to_idx,
        }
        return model, meta

    def _evaluate_all(self, test_df, books_df, embedder, svd, ncf_model, ncf_meta, k):
        results = {}

        # Relevance sets
        user_relevant = {}
        for uid, grp in test_df.groupby("user_id"):
            liked = set(grp.loc[grp["book_rating"] >= RELEVANCE_THRESHOLD, "book__isbn"])
            if liked:
                user_relevant[int(uid)] = liked

        sample_users = list(user_relevant.keys())[:200]

        def _rmse_for(predict_fn):
            actuals, preds = [], []
            for _, row in test_df.iterrows():
                try:
                    p = predict_fn(int(row["user_id"]), str(row["book__isbn"]))
                    if p is not None:
                        actuals.append(float(row["book_rating"]))
                        preds.append(float(p))
                except Exception:
                    pass
            return compute_rmse(actuals, preds) if actuals else None

        def _ranking(recommend_fn):
            user_recs = {}
            for uid in sample_users:
                try:
                    user_recs[uid] = recommend_fn(uid)
                except Exception:
                    pass
            prec = compute_precision_at_k(user_recs, user_relevant, k)
            ndcg = compute_ndcg_at_k(user_recs, user_relevant, k)
            return prec, ndcg

        all_isbns = books_df["isbn"].tolist()

        # --- SVD baseline ---
        if svd:
            def svd_predict(uid, isbn):
                return svd.predict(uid, isbn).est

            def svd_recommend(uid):
                test_user_rated = set(
                    test_df[test_df["user_id"] == uid]["book__isbn"].tolist()
                )
                candidates = [i for i in all_isbns if i not in test_user_rated]
                preds = [(i, svd.predict(uid, i).est) for i in candidates[:500]]
                preds.sort(key=lambda x: x[1], reverse=True)
                return [i for i, _ in preds[:k]]

            rmse = _rmse_for(svd_predict)
            prec, ndcg = _ranking(svd_recommend)
            results["SVD (baseline)"] = {
                "rmse": round(rmse, 4) if rmse else None,
                f"precision_at_{k}": round(prec, 4),
                f"ndcg_at_{k}": round(ndcg, 4),
            }

        # --- Content-only (LSA cosine) ---
        if embedder:
            def content_recommend(uid):
                test_user_rated = set(
                    test_df[test_df["user_id"] == uid]["book__isbn"].tolist()
                )
                liked = [i for i in test_user_rated
                         if test_df[(test_df["user_id"] == uid) &
                                    (test_df["book__isbn"] == i)]["book_rating"].values[0]
                         >= RELEVANCE_THRESHOLD]
                profile = embedder.build_user_profile(liked)
                candidates = [i for i in all_isbns if i not in test_user_rated]
                if profile is None or not candidates:
                    return []
                vecs = np.stack([embedder.get(i) for i in candidates[:500]])
                sims = vecs @ profile
                ranked = np.argsort(sims)[::-1][:k]
                return [candidates[i] for i in ranked]

            prec, ndcg = _ranking(content_recommend)
            results["Content-only (LSA)"] = {
                "rmse": None,
                f"precision_at_{k}": round(prec, 4),
                f"ndcg_at_{k}": round(ndcg, 4),
            }

        # --- NCF ---
        if ncf_model is not None:
            import torch
            from bookflix.ml.model_store import ncf_predict as _ncf_predict
            # Temporarily inject meta into module-level cache for evaluation
            import bookflix.ml.model_store as ms
            ms._cache["ncf"] = (ncf_model, ncf_meta)

            def ncf_recommend_fn(uid):
                test_user_rated = set(
                    test_df[test_df["user_id"] == uid]["book__isbn"].tolist()
                )
                candidates = [i for i in all_isbns if i not in test_user_rated]
                return ms.ncf_recommend(uid, candidates[:500], top_n=k)

            rmse = _rmse_for(_ncf_predict)
            prec, ndcg = _ranking(ncf_recommend_fn)
            results["NCF"] = {
                "rmse": round(rmse, 4) if rmse else None,
                f"precision_at_{k}": round(prec, 4),
                f"ndcg_at_{k}": round(ndcg, 4),
            }

            # --- Feature Combination Hybrid ---
            if embedder:
                from bookflix.ml.hybrid import feature_combination_recommend

                def hybrid_recommend_fn(uid):
                    test_user_rated = set(
                        test_df[test_df["user_id"] == uid]["book__isbn"].tolist()
                    )
                    liked = [i for i in test_user_rated
                             if test_df[(test_df["user_id"] == uid) &
                                        (test_df["book__isbn"] == i)]["book_rating"].values[0]
                             >= RELEVANCE_THRESHOLD]
                    candidates = [i for i in all_isbns if i not in test_user_rated]
                    ranked = feature_combination_recommend(
                        user_id=uid,
                        candidate_isbns=candidates[:500],
                        embedder=embedder,
                        rated_isbns=liked,
                        ncf_score_fn=_ncf_predict,
                        top_n=k,
                    )
                    return [r["isbn"] for r in ranked]

                prec, ndcg = _ranking(hybrid_recommend_fn)
                results["NCF + LSA Hybrid (Feature Combination)"] = {
                    "rmse": None,
                    f"precision_at_{k}": round(prec, 4),
                    f"ndcg_at_{k}": round(ndcg, 4),
                }

                # --- Hybrid + Sentiment ---
                if sentiment_map := (lambda: None)():
                    pass  # placeholder — sentiment map loaded from pkl during eval
                with open(_path("sentiment_map.pkl"), "rb") as f:
                    s_map = pickle.load(f)

                if s_map:
                    from bookflix.ml.sentiment import apply_sentiment_correction
                    test_adj = apply_sentiment_correction(test_df, s_map)

                    def hybrid_sentiment_recommend_fn(uid):
                        test_user_rated = set(
                            test_adj[test_adj["user_id"] == uid]["book__isbn"].tolist()
                        )
                        liked = [i for i in test_user_rated
                                 if test_adj[(test_adj["user_id"] == uid) &
                                             (test_adj["book__isbn"] == i)]["adjusted_rating"].values[0]
                                 >= RELEVANCE_THRESHOLD]
                        candidates = [i for i in all_isbns if i not in test_user_rated]
                        ranked = feature_combination_recommend(
                            user_id=uid,
                            candidate_isbns=candidates[:500],
                            embedder=embedder,
                            rated_isbns=liked,
                            ncf_score_fn=_ncf_predict,
                            top_n=k,
                        )
                        return [r["isbn"] for r in ranked]

                    def hybrid_rmse_sentiment(uid, isbn):
                        base = _ncf_predict(uid, isbn)
                        if base is None:
                            return None
                        return base + 0.5 * s_map.get(isbn, 0.0)

                    rmse = _rmse_for(hybrid_rmse_sentiment)
                    prec, ndcg = _ranking(hybrid_sentiment_recommend_fn)
                    results["NCF + LSA + Sentiment Correction"] = {
                        "rmse": round(rmse, 4) if rmse else None,
                        f"precision_at_{k}": round(prec, 4),
                        f"ndcg_at_{k}": round(ndcg, 4),
                    }

        return results
