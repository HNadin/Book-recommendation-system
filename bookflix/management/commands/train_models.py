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
5. Fit BERT semantic embedder on all books (Section 2.2).
6. Evaluate all model variants on test split; save metrics JSON.
"""

import logging
import os
import pickle
import random

from django.core.management.base import BaseCommand

import numpy as np
import pandas as pd

from bookflix.ml.embeddings import BERTEmbedder
from bookflix.ml.evaluation import (
    RELEVANCE_THRESHOLD,
    compute_ndcg_at_k,
    compute_precision_at_k,
    compute_rmse,
)
from bookflix.ml.model_store import MODELS_DIR, _path, save_eval_results
from bookflix.ml.sentiment import (
    apply_sentiment_correction,
    build_amazon_sentiment_map,
    build_book_sentiment_map,
    build_metadata_sentiment_map,
)
from bookflix.recommendation_algorithms import load_books_df, load_data_ratings, train_test_split

logger = logging.getLogger(__name__)
SEED = 42
random.seed(SEED)
np.random.seed(SEED)


class Command(BaseCommand):
    help = "Train NCF, SVD baseline, and BERT embedder; save artefacts + evaluation metrics."

    def add_arguments(self, parser):
        parser.add_argument(
            "--epochs", type=int, default=20,
            help="NCF max training epochs; early stopping usually halts sooner (default: 20)"
        )
        parser.add_argument(
            "--embedding-dim", type=int, default=32,
            help="NCF embedding dimension (default: 32)"
        )
        parser.add_argument(
            "--dropout", type=float, default=0.5,
            help="NCF dropout rate — higher fights overfitting on sparse data (default: 0.5)"
        )
        parser.add_argument(
            "--min-interactions", type=int, default=5,
            help="Keep only users and books with >= N ratings (k-core, default: 5; 0 disables)"
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
            "--weight-decay", type=float, default=1e-5,
            help="NCF L2 regularization on embeddings/weights (default: 1e-5)"
        )
        parser.add_argument(
            "--patience", type=int, default=5,
            help="Early-stopping patience: stop after N epochs without val improvement (default: 5)"
        )
        parser.add_argument(
            "--sentiments-csv", type=str, default="",
            help="Path to pre-calculated sentiments CSV (columns: isbn/final_isbn + polarity/compound). "
                 "Takes priority over --amazon-csv and --reviews-csv."
        )
        parser.add_argument(
            "--reviews-csv", type=str, default="Reviews.csv",
            help="Path to reviews CSV for sentiment correction"
        )
        parser.add_argument(
            "--amazon-csv", type=str, default="",
            help="Path to Amazon Books_rating.csv for sentiment correction "
                 "(columns: Id, Title, review/text, review/summary). "
                 "When provided, takes priority over --reviews-csv."
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

        # k-core filtering: drop "one-off" users/books so NCF has enough
        # interactions per embedding to learn from (combats extreme sparsity).
        min_int = options["min_interactions"]
        if min_int and min_int > 1:
            before = len(ratings_df)
            ratings_df = self._kcore_filter(ratings_df, min_int)
            n_users = ratings_df["user_id"].nunique()
            n_books = ratings_df["book__isbn"].nunique()
            density = len(ratings_df) / max(1, n_books)
            self.stdout.write(
                f"  k-core (>= {min_int}): {before:,} -> {len(ratings_df):,} ratings  |  "
                f"{n_users:,} users, {n_books:,} books  ({density:.1f} ratings/book)"
            )

        train_df, test_df = train_test_split(ratings_df, test_ratio=0.2)
        self.stdout.write(f"  Train: {len(train_df):,}  |  Test: {len(test_df):,}")

        # --- Sentiment correction ---
        self.stdout.write("=== Step 2: Sentiment correction ===")
        sentiment_map: dict = {}

        sentiments_csv = options.get("sentiments_csv", "")
        if sentiments_csv and os.path.exists(sentiments_csv):
            try:
                self.stdout.write(f"  Loading pre-calculated sentiments from {sentiments_csv} …")
                sent_df = pd.read_csv(sentiments_csv, low_memory=False)
                # Support column name variants from different pre-processing scripts
                isbn_col = next(
                    (c for c in ("final_isbn", "isbn", "ISBN", "Id", "id") if c in sent_df.columns), None
                )
                score_col = next(
                    (c for c in ("polarity", "compound", "sentiment", "score") if c in sent_df.columns), None
                )
                if isbn_col and score_col:
                    sentiment_map = (
                        sent_df.dropna(subset=[isbn_col, score_col])
                        .groupby(isbn_col)[score_col].mean()
                        .to_dict()
                    )
                    sentiment_map = {str(k): float(v) for k, v in sentiment_map.items()}
                    n = len(sentiment_map)
                    avg = sum(sentiment_map.values()) / n if n else 0.0
                    train_df = apply_sentiment_correction(train_df, sentiment_map)
                    self.stdout.write(
                        f"  Matched {n:,} books via pre-calculated sentiments (avg compound: {avg:+.3f})."
                    )
                else:
                    self.stdout.write(self.style.WARNING(
                        f"  sentiments.csv columns not recognised: {sent_df.columns.tolist()}"
                    ))
            except Exception as e:
                self.stdout.write(self.style.WARNING(f"  Failed to load sentiments CSV: {e}"))
        else:
            sentiments_csv = ""  # fall through to amazon/reviews paths

        amazon_path = options.get("amazon_csv", "") if not sentiments_csv else ""
        if amazon_path and os.path.exists(amazon_path):
            try:
                self.stdout.write(f"  Loading Amazon Books Reviews from {amazon_path} …")
                amazon_df = pd.read_csv(amazon_path, on_bad_lines="skip", low_memory=False)
                self.stdout.write(f"  Amazon CSV columns: {amazon_df.columns.tolist()}")
                sentiment_map = build_amazon_sentiment_map(amazon_df, books_df)
                n_matched = len(sentiment_map)
                if n_matched:
                    avg = sum(sentiment_map.values()) / n_matched
                    train_df = apply_sentiment_correction(train_df, sentiment_map)
                    self.stdout.write(
                        f"  Matched {n_matched:,} books via Amazon Reviews "
                        f"(avg compound: {avg:+.3f})."
                    )
                else:
                    self.stdout.write(self.style.WARNING("  Amazon CSV produced no matches."))
            except Exception as e:
                self.stdout.write(self.style.WARNING(f"  Amazon sentiment failed: {e}"))
        else:
            reviews_path = options["reviews_csv"]
            if os.path.exists(reviews_path):
                try:
                    reviews_df = pd.read_csv(reviews_path, encoding="cp1252", on_bad_lines="skip", low_memory=False)
                    sentiment_map = build_book_sentiment_map(reviews_df)
                    if sentiment_map:
                        train_df = apply_sentiment_correction(train_df, sentiment_map)
                        self.stdout.write(f"  Sentiment map: {len(sentiment_map):,} books corrected.")
                    else:
                        self.stdout.write(self.style.WARNING(
                            "  Reviews CSV has no ISBN column — sentiment correction skipped."
                        ))
                except Exception as e:
                    self.stdout.write(self.style.WARNING(f"  Sentiment correction skipped: {e}"))
            else:
                self.stdout.write(self.style.WARNING(
                    f"  No Amazon CSV and {reviews_path} not found — sentiment correction skipped."
                ))

        # Save sentiment map
        with open(_path("sentiment_map.pkl"), "wb") as f:
            pickle.dump(sentiment_map, f)

        # --- BERT embedder (Section 2.2) ---
        self.stdout.write("=== Step 3: Training BERT semantic embedder (all-MiniLM-L6-v2) ===")
        try:
            embedder = BERTEmbedder()
            embedder.fit(books_df)
            embedder.save(_path("bert_embedder.pkl"))
            self.stdout.write("  BERT embedder saved.")
        except Exception as e:
            self.stdout.write(self.style.WARNING(
                f"  BERT embedder failed ({e}); falling back to LSA embedder."
            ))
            from bookflix.ml.embeddings import LSAEmbedder
            embedder = LSAEmbedder()
            embedder.fit(books_df)
            embedder.save(_path("lsa_embedder.pkl"))
            self.stdout.write("  LSA embedder saved (fallback).")

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
            weight_decay=options["weight_decay"],
            patience=options["patience"],
            dropout=options["dropout"],
        )
        if ncf_model:
            import torch
            torch.save(ncf_model.state_dict(), _path("ncf_model.pt"))
            with open(_path("ncf_meta.pkl"), "wb") as f:
                pickle.dump(meta, f)
            self.stdout.write("  NCF model saved.")

        # --- NCF ablation: train on raw ratings to isolate sentiment contribution ---
        ncf_raw_model, ncf_raw_meta = None, {}
        if rating_col == "adjusted_rating":
            self.stdout.write("=== Step 5b: NCF ablation (raw ratings, no sentiment correction) ===")
            ncf_raw_model, ncf_raw_meta = self._train_ncf(
                train_df,
                rating_col="book_rating",
                embedding_dim=options["embedding_dim"],
                epochs=options["epochs"],
                batch_size=options["batch_size"],
                lr=options["lr"],
                weight_decay=options["weight_decay"],
                patience=options["patience"],
                dropout=options["dropout"],
            )
            if ncf_raw_model:
                self.stdout.write("  NCF ablation model trained (not saved to disk).")

        # --- Evaluation ---
        self.stdout.write("=== Step 6: Evaluating all models ===")
        k = options["eval_k"]
        results = self._evaluate_all(
            train_df, test_df, books_df, embedder, svd_model,
            ncf_model, meta, k,
            ncf_raw_model=ncf_raw_model, ncf_raw_meta=ncf_raw_meta,
        )
        save_eval_results(results)
        self.stdout.write("\n=== Evaluation Results ===")
        for model_name, metrics in results.items():
            self.stdout.write(f"  {model_name}:")
            for metric, val in metrics.items():
                self.stdout.write(f"    {metric}: {val:.4f}" if val is not None else f"    {metric}: N/A")

        self.stdout.write(self.style.SUCCESS("\nAll models trained and saved successfully."))

    # -----------------------------------------------------------------------

    def _kcore_filter(self, ratings_df, min_int):
        """Iteratively drop users and books with fewer than min_int ratings.

        Filtering users changes book counts and vice versa, so we repeat until
        the set is stable (a k-core), guaranteeing every surviving user AND
        book has at least min_int interactions."""
        df = ratings_df
        while True:
            n0 = len(df)
            ub = df["user_id"].value_counts()
            df = df[df["user_id"].isin(ub[ub >= min_int].index)]
            bb = df["book__isbn"].value_counts()
            df = df[df["book__isbn"].isin(bb[bb >= min_int].index)]
            if len(df) == n0 or df.empty:
                break
        return df.reset_index(drop=True)

    def _train_svd(self, train_df):
        try:
            from surprise import SVD as SurpriseSVD
            from surprise import Dataset, Reader
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

    def _train_ncf(self, train_df, rating_col, embedding_dim, epochs, batch_size, lr,
                   weight_decay=1e-5, patience=5, dropout=0.5, val_frac=0.1):
        try:
            import copy

            import torch
            import torch.nn as nn
            from torch.utils.data import DataLoader, TensorDataset, random_split
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

        # Hold out a validation split to detect overfitting and drive early stopping.
        n_val = max(1, int(len(dataset) * val_frac))
        n_train = len(dataset) - n_val
        gen = torch.Generator().manual_seed(SEED)
        train_set, val_set = random_split(dataset, [n_train, n_val], generator=gen)
        loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
        self.stdout.write(
            f"  Train interactions: {n_train:,}  |  Validation: {n_val:,}  "
            f"(weight_decay={weight_decay}, patience={patience})"
        )

        model = NCF(
            num_users=len(users),
            num_items=len(isbns),
            embedding_dim=embedding_dim,
            dropout=dropout,
        )
        # weight_decay applies L2 regularization to the embeddings — the main
        # source of overfitting on a sparse catalogue (~1.6 ratings/book).
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        criterion = nn.MSELoss()

        best_val = float("inf")
        best_state = copy.deepcopy(model.state_dict())
        best_epoch = 0
        epochs_no_improve = 0

        for epoch in range(1, epochs + 1):
            model.train()
            total_loss = 0.0
            for u_b, v_b, r_b in loader:
                optimizer.zero_grad()
                pred = model(u_b, v_b)
                loss = criterion(pred, r_b)
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * len(r_b)
            avg_loss = total_loss / n_train

            # Validation pass
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for u_b, v_b, r_b in val_loader:
                    val_loss += criterion(model(u_b, v_b), r_b).item() * len(r_b)
            val_loss /= n_val

            marker = ""
            if val_loss < best_val - 1e-4:
                best_val = val_loss
                best_state = copy.deepcopy(model.state_dict())
                best_epoch = epoch
                epochs_no_improve = 0
                marker = "  *"
            else:
                epochs_no_improve += 1

            self.stdout.write(
                f"  Epoch {epoch}/{epochs}  loss={avg_loss:.4f}  val_loss={val_loss:.4f}{marker}"
            )

            if epochs_no_improve >= patience:
                self.stdout.write(
                    f"  Early stopping at epoch {epoch} "
                    f"(best val_loss={best_val:.4f} @ epoch {best_epoch})."
                )
                break

        # Restore the weights with the lowest validation loss.
        model.load_state_dict(best_state)
        self.stdout.write(
            f"  Restored best model: epoch {best_epoch}, val_loss={best_val:.4f}."
        )

        meta = {
            "num_users": len(users),
            "num_items": len(isbns),
            "embedding_dim": embedding_dim,
            "user_to_idx": user_to_idx,
            "isbn_to_idx": isbn_to_idx,
            "best_epoch": best_epoch,
            "best_val_loss": best_val,
            "weight_decay": weight_decay,
        }
        return model, meta

    def _evaluate_all(self, train_df, test_df, books_df, embedder, svd, ncf_model, ncf_meta, k,
                      ncf_raw_model=None, ncf_raw_meta=None):
        results = {}
        all_isbns = books_df["isbn"].tolist()

        # ------------------------------------------------------------------
        # Per-rating split evaluation (Section 3.5):
        #   train_df  — items already seen by the model (context)
        #   test_df   — held-out items; liked ones (>= threshold) = ground truth
        #
        # Since train_test_split is now per-rating (not per-user), every user
        # appears in train_df, so all models can predict for all users.
        # Candidates = all_isbns - user's training items.
        # ------------------------------------------------------------------
        user_context: dict[int, set] = {}
        user_context_liked: dict[int, list] = {}

        for uid, grp in train_df.groupby("user_id"):
            uid = int(uid)
            user_context[uid] = set(grp["book__isbn"].tolist())
            user_context_liked[uid] = grp[
                grp["book_rating"] >= RELEVANCE_THRESHOLD
            ]["book__isbn"].tolist()

        user_relevant: dict[int, set] = {}
        for uid, grp in test_df.groupby("user_id"):
            uid = int(uid)
            liked = set(grp[grp["book_rating"] >= RELEVANCE_THRESHOLD]["book__isbn"])
            if liked:
                user_relevant[uid] = liked

        # Disjoint user groups: TUNE drives hybrid weight selection, EVAL is the
        # held-out group on which every reported metric is computed.  No user
        # appears in both, so tuned weights never see the evaluation data.
        _rel_users = list(user_relevant.keys())
        tune_users = _rel_users[:100]
        eval_users = _rel_users[100:300]

        # Per-user seeded candidate sampling: the pool for a given user is fixed
        # regardless of call order, so the weight grid search cannot perturb the
        # pools used to score other models.  Each pool draws up to 2 000 items
        # and always includes the user's relevant test items so that Precision@K
        # and NDCG@K are meaningful against a catalogue of 200 000+ books.
        _POOL_SIZE = 2000

        def _pool(uid, candidates):
            rng = np.random.default_rng(SEED + int(uid))
            rel = user_relevant.get(uid, set())
            rel_in = [c for c in candidates if c in rel]
            negatives = [c for c in candidates if c not in rel]
            n_neg = max(0, min(_POOL_SIZE - len(rel_in), len(negatives)))
            neg_sample = rng.choice(negatives, n_neg, replace=False).tolist() if n_neg else []
            pool = rel_in + neg_sample
            rng.shuffle(pool)
            return pool

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

        def _ranking(recommend_fn, users=None):
            users = eval_users if users is None else users
            user_recs = {}
            for uid in users:
                try:
                    user_recs[uid] = recommend_fn(uid)
                except Exception:
                    pass
            prec = compute_precision_at_k(user_recs, user_relevant, k)
            ndcg = compute_ndcg_at_k(user_recs, user_relevant, k)
            return prec, ndcg

        # --- SVD baseline ---
        if svd:
            def svd_predict(uid, isbn):
                return svd.predict(uid, isbn).est

            def svd_recommend(uid):
                ctx = user_context.get(uid, set())
                candidates = [i for i in all_isbns if i not in ctx]
                preds = [(i, svd.predict(uid, i).est) for i in _pool(uid, candidates)]
                preds.sort(key=lambda x: x[1], reverse=True)
                return [i for i, _ in preds[:k]]

            rmse = _rmse_for(svd_predict)
            prec, ndcg = _ranking(svd_recommend)
            results["SVD (baseline)"] = {
                "rmse": round(rmse, 4) if rmse else None,
                f"precision_at_{k}": round(prec, 4),
                f"ndcg_at_{k}": round(ndcg, 4),
            }

        # --- Content-only (BERT cosine) ---
        if embedder:
            def content_recommend(uid):
                ctx = user_context.get(uid, set())
                liked_ctx = user_context_liked.get(uid, [])
                profile = embedder.build_user_profile(liked_ctx)
                candidates = [i for i in all_isbns if i not in ctx]
                if profile is None or not candidates:
                    return []
                pool = _pool(uid, candidates)
                vecs = np.stack([embedder.get(i) for i in pool])
                sims = vecs @ profile
                ranked = np.argsort(sims)[::-1][:k]
                return [pool[i] for i in ranked]

            prec, ndcg = _ranking(content_recommend)
            results["Content-only (BERT)"] = {
                "rmse": None,
                f"precision_at_{k}": round(prec, 4),
                f"ndcg_at_{k}": round(ndcg, 4),
            }

        # --- NCF ---
        if ncf_model is not None:
            import bookflix.ml.model_store as ms
            from bookflix.ml.model_store import ncf_predict as _ncf_predict
            ms._cache["ncf"] = (ncf_model, ncf_meta)

            ncf_known = set(ncf_meta.get("isbn_to_idx", {}).keys())

            def ncf_recommend_fn(uid):
                ctx = user_context.get(uid, set())
                # Use all unseen candidates so _pool() can guarantee relevant test
                # items are included; ncf_recommend internally filters to known items.
                candidates = [i for i in all_isbns if i not in ctx]
                return ms.ncf_recommend(uid, _pool(uid, candidates), top_n=k)

            rmse = _rmse_for(_ncf_predict)
            prec, ndcg = _ranking(ncf_recommend_fn)
            results["NCF"] = {
                "rmse": round(rmse, 4) if rmse else None,
                f"precision_at_{k}": round(prec, 4),
                f"ndcg_at_{k}": round(ndcg, 4),
            }

            # --- Ablation: NCF trained on raw (uncorrected) ratings ---
            # Isolates the contribution of sentiment correction.
            # Comparing "NCF (no sentiment)" vs "NCF" shows whether adjusted_rating
            # actually helps — without this, the sentiment benefit is unproven.
            if ncf_raw_model is not None and ncf_raw_meta:
                import bookflix.ml.model_store as ms_raw
                ms_raw._cache["ncf_ablation"] = (ncf_raw_model, ncf_raw_meta)

                def _ncf_raw_predict(uid, isbn):
                    uid_map = ncf_raw_meta["user_to_idx"]
                    iid_map = ncf_raw_meta["isbn_to_idx"]
                    u = uid_map.get(uid)
                    v = iid_map.get(isbn)
                    if u is None or v is None:
                        return None
                    import torch
                    with torch.no_grad():
                        score = ncf_raw_model(
                            torch.tensor([u]), torch.tensor([v])
                        ).item()
                    return float(np.clip(score, 1.0, 10.0))

                def ncf_raw_recommend_fn(uid):
                    ctx = user_context.get(uid, set())
                    candidates = [i for i in all_isbns if i not in ctx]
                    pool = _pool(uid, candidates)
                    known = ncf_raw_meta.get("isbn_to_idx", {})
                    scores = [
                        (isbn, _ncf_raw_predict(uid, isbn) or 5.0)
                        for isbn in pool if isbn in known
                    ]
                    scores.sort(key=lambda x: x[1], reverse=True)
                    return [isbn for isbn, _ in scores[:k]]

                rmse_raw = _rmse_for(_ncf_raw_predict)
                prec_raw, ndcg_raw = _ranking(ncf_raw_recommend_fn)
                results["NCF (no sentiment — ablation)"] = {
                    "rmse": round(rmse_raw, 4) if rmse_raw else None,
                    f"precision_at_{k}": round(prec_raw, 4),
                    f"ndcg_at_{k}": round(ndcg_raw, 4),
                }

            # --- Feature Combination Hybrid (weights tuned on held-out users) ---
            if embedder:
                from bookflix.ml.hybrid import _normalise

                with open(_path("sentiment_map.pkl"), "rb") as f:
                    s_map = pickle.load(f)

                # Precompute, once per user, the three normalised signal vectors
                # over that user's fixed candidate pool.  Weight selection then
                # reduces to cheap weighted sums instead of re-embedding books.
                _comp_cache: dict = {}

                def _components(uid):
                    if uid in _comp_cache:
                        return _comp_cache[uid]
                    ctx = user_context.get(uid, set())
                    liked_ctx = user_context_liked.get(uid, [])
                    candidates = [i for i in all_isbns if i not in ctx]
                    pool = _pool(uid, candidates)
                    profile = embedder.build_user_profile(liked_ctx)
                    if profile is None:
                        sem = np.zeros(len(pool))
                    else:
                        sem = np.stack([embedder.get(i) for i in pool]) @ profile
                    ncf_raw = np.array([_ncf_predict(uid, i) or 5.0 for i in pool])
                    sent_raw = (np.array([s_map.get(i, 0.0) for i in pool])
                                if s_map else np.zeros(len(pool)))
                    comp = (pool, _normalise(ncf_raw), _normalise(sem), _normalise(sent_raw))
                    _comp_cache[uid] = comp
                    return comp

                def _rank(uid, w_ncf, w_sem, w_sent):
                    pool, ncf_n, sem_n, sent_n = _components(uid)
                    combined = w_ncf * ncf_n + w_sem * sem_n + w_sent * sent_n
                    idx = np.argsort(combined)[::-1][:k]
                    return [pool[i] for i in idx]

                def _ndcg_on(users, w_ncf, w_sem, w_sent):
                    recs = {uid: _rank(uid, w_ncf, w_sem, w_sent) for uid in users}
                    return compute_ndcg_at_k(recs, user_relevant, k)

                # --- Grid search 1: NCF + BERT (two-way blend) ---
                steps = [i / 10 for i in range(11)]   # 0.0 .. 1.0
                best2 = max(
                    ((_ndcg_on(tune_users, w, 1 - w, 0.0), w, 1 - w, 0.0) for w in steps),
                    key=lambda t: t[0],
                )
                _, w_ncf2, w_sem2, _ = best2
                prec, ndcg = _ranking(lambda uid: _rank(uid, w_ncf2, w_sem2, 0.0))
                self.stdout.write(
                    f"  Hybrid (NCF+BERT) tuned weights: w_ncf={w_ncf2:.1f} w_sem={w_sem2:.1f}"
                )
                results["NCF + BERT Hybrid (Feature Combination)"] = {
                    "rmse": None,
                    f"precision_at_{k}": round(prec, 4),
                    f"ndcg_at_{k}": round(ndcg, 4),
                    "w_ncf": w_ncf2, "w_sem": w_sem2, "w_sent": 0.0,
                }

                tuned_weights = {"w_ncf": w_ncf2, "w_sem": w_sem2, "w_sent": 0.0}

                # --- Grid search 2: NCF + BERT + Sentiment (three-way simplex) ---
                if s_map:
                    grid3 = [
                        (a / 10, b / 10, 1 - a / 10 - b / 10)
                        for a in range(11) for b in range(11 - a)
                    ]
                    best3 = max(
                        ((_ndcg_on(tune_users, wn, ws, wse), wn, ws, wse)
                         for wn, ws, wse in grid3),
                        key=lambda t: t[0],
                    )
                    _, w_ncf3, w_sem3, w_sent3 = best3
                    prec, ndcg = _ranking(lambda uid: _rank(uid, w_ncf3, w_sem3, w_sent3))

                    def hybrid_rmse_sentiment(uid, isbn):
                        base = _ncf_predict(uid, isbn)
                        if base is None:
                            return None
                        return base + 0.5 * s_map.get(isbn, 0.0)

                    rmse = _rmse_for(hybrid_rmse_sentiment)
                    self.stdout.write(
                        f"  Hybrid (NCF+BERT+Sentiment) tuned weights: "
                        f"w_ncf={w_ncf3:.1f} w_sem={w_sem3:.1f} w_sent={w_sent3:.1f}"
                    )
                    results["NCF + BERT + Sentiment Correction"] = {
                        "rmse": round(rmse, 4) if rmse else None,
                        f"precision_at_{k}": round(prec, 4),
                        f"ndcg_at_{k}": round(ndcg, 4),
                        "w_ncf": w_ncf3, "w_sem": w_sem3, "w_sent": w_sent3,
                    }

                    # Persist the best blend overall for the live recommender.
                    if best3[0] >= best2[0]:
                        tuned_weights = {"w_ncf": w_ncf3, "w_sem": w_sem3, "w_sent": w_sent3}

                with open(_path("hybrid_weights.pkl"), "wb") as f:
                    pickle.dump(tuned_weights, f)
                self.stdout.write(f"  Saved tuned hybrid weights: {tuned_weights}")

        return results
