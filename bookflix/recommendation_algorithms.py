"""
Recommendation algorithms — unified entry point for all model variants.

Delegates to the ml/ sub-package.  Views should import from here, not
from ml/ directly, so the API surface stays stable as models evolve.
"""

import logging
import random

import numpy as np
import pandas as pd
from django.db.models import Avg

from bookflix.ml.model_store import (
    load_embedder,
    load_ncf,
    load_svd_baseline,
    ncf_predict,
    ncf_recommend,
)
from bookflix.ml.hybrid import feature_combination_recommend, cascade_recommend
from bookflix.ml.evaluation import (
    compute_rmse,
    compute_precision_at_k,
    compute_ndcg_at_k,
    RELEVANCE_THRESHOLD,
)

logger = logging.getLogger(__name__)
SEED = 42
random.seed(SEED)
np.random.seed(SEED)


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def load_data_ratings() -> pd.DataFrame:
    from bookflix.models import Rating
    ratings = list(Rating.objects.values("user_id", "book__isbn", "book_rating"))
    return pd.DataFrame(ratings)


def load_books_df() -> pd.DataFrame:
    from bookflix.models import Book, Rating
    books = pd.DataFrame(list(Book.objects.all().values()))
    avg = (
        Rating.objects
        .values("book__isbn")
        .annotate(avg_rating=Avg("book_rating"))
    )
    avg_df = pd.DataFrame(list(avg))
    if not avg_df.empty:
        books = books.merge(avg_df, left_on="isbn", right_on="book__isbn", how="left")
        books["avg_rating"] = books.get("avg_rating", pd.Series(dtype=float)).fillna(0)
    else:
        books["avg_rating"] = 0.0
    return books


def train_test_split(ratings_df: pd.DataFrame, test_ratio: float = 0.2):
    """
    Stratified user split: 80 % of users → train, 20 % → test.
    Returns (train_df, test_df).
    """
    users = ratings_df["user_id"].unique().tolist()
    random.shuffle(users)
    split = int(len(users) * (1 - test_ratio))
    train_users = set(users[:split])
    train_df = ratings_df[ratings_df["user_id"].isin(train_users)]
    test_df = ratings_df[~ratings_df["user_id"].isin(train_users)]
    return train_df, test_df


# ---------------------------------------------------------------------------
# Per-user helpers (used by views at request time)
# ---------------------------------------------------------------------------

def get_user_rated_isbns(user_id: int, ratings_df: pd.DataFrame,
                          min_rating: int = RELEVANCE_THRESHOLD) -> list[str]:
    user_rows = ratings_df[ratings_df["user_id"] == user_id]
    liked = user_rows[user_rows["book_rating"] >= min_rating]["book__isbn"].tolist()
    return liked


def get_user_all_rated_isbns(user_id: int, ratings_df: pd.DataFrame) -> list[str]:
    return ratings_df[ratings_df["user_id"] == user_id]["book__isbn"].tolist()


# ---------------------------------------------------------------------------
# Recommendation entry points
# ---------------------------------------------------------------------------

def hybrid_recommendations(
    user_id: int,
    ratings_df: pd.DataFrame,
    books_df: pd.DataFrame,
    top_n: int = 10,
    use_sentiment_adjusted: bool = False,
) -> list[dict]:
    """
    Feature Combination hybrid (Section 2.4):
    NCF collaborative score + LSA semantic similarity, combined with
    configurable weights and normalised per candidate pool.

    Falls back to SVD-cascade when NCF weights are absent.
    """
    embedder = load_embedder()
    rating_col = "adjusted_rating" if use_sentiment_adjusted else "book_rating"

    all_isbns = books_df["isbn"].tolist()
    rated_isbns = get_user_all_rated_isbns(user_id, ratings_df)
    liked_isbns = get_user_rated_isbns(user_id, ratings_df)
    candidate_isbns = [i for i in all_isbns if i not in set(rated_isbns)]

    # --- NCF path ---
    ncf_result = load_ncf()
    if ncf_result is not None and embedder is not None:
        model, meta = ncf_result
        ranked = feature_combination_recommend(
            user_id=user_id,
            candidate_isbns=candidate_isbns[:2000],
            embedder=embedder,
            rated_isbns=liked_isbns,
            ncf_score_fn=ncf_predict,
            top_n=top_n,
        )
        return ranked

    # --- Cascade SVD fallback ---
    if embedder is not None:
        svd = load_svd_baseline()
        def svd_score(uid, isbn):
            if svd is None:
                return 5.0
            try:
                return svd.predict(uid, isbn).est
            except Exception:
                return 5.0

        return cascade_recommend(
            user_id=user_id,
            rated_isbns=liked_isbns,
            all_isbns=all_isbns,
            embedder=embedder,
            ncf_score_fn=svd_score,
            top_n=top_n,
        )

    # --- Bare fallback: popularity ---
    logger.warning("No trained models found; falling back to popularity ranking.")
    popular = books_df.sort_values("avg_rating", ascending=False)
    popular = popular[~popular["isbn"].isin(set(rated_isbns))]
    return [{"isbn": row["isbn"], "score": row["avg_rating"]}
            for _, row in popular.head(top_n).iterrows()]


def svd_recommendations(
    user_id: int,
    ratings_df: pd.DataFrame,
    books_df: pd.DataFrame,
    top_n: int = 10,
) -> list[dict]:
    """SVD baseline for the comparison table (Section 3.5)."""
    from surprise import Dataset, Reader, SVD as SurpriseSVD
    svd = load_svd_baseline()
    if svd is None:
        return []

    rated = set(get_user_all_rated_isbns(user_id, ratings_df))
    candidates = [isbn for isbn in books_df["isbn"].unique() if isbn not in rated]
    preds = [(isbn, svd.predict(user_id, isbn).est) for isbn in candidates[:3000]]
    preds.sort(key=lambda x: x[1], reverse=True)
    return [{"isbn": isbn, "score": round(s, 3)} for isbn, s in preds[:top_n]]


# ---------------------------------------------------------------------------
# Evaluation helpers (called by the evaluate view and train_models command)
# ---------------------------------------------------------------------------

def evaluate_user_model(user_id: int, ratings_df: pd.DataFrame) -> tuple:
    """
    Returns (metrics_dict, error_str | None).
    metrics_dict has keys: rmse, precision_at_10, ndcg_at_10.
    """
    user_rows = ratings_df[ratings_df["user_id"] == user_id]
    if user_rows.empty:
        return None, "No ratings found for user"

    ncf_result = load_ncf()
    if ncf_result is None:
        return None, "NCF model not trained yet — run train_models"

    actuals, preds = [], []
    for _, row in user_rows.iterrows():
        p = ncf_predict(user_id, row["book__isbn"])
        if p is not None:
            actuals.append(float(row["book_rating"]))
            preds.append(p)

    if not actuals:
        return None, "No NCF predictions available for user"

    rmse = compute_rmse(actuals, preds)
    return {"rmse": round(rmse, 4)}, None


# ---------------------------------------------------------------------------
# Backward-compatibility shims
# Old views.py / management commands may import these names.
# ---------------------------------------------------------------------------

def compute_average_ratings():
    return load_books_df()


def build_tfidf_matrix():
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.neighbors import NearestNeighbors
    books = load_books_df()
    tfidf = TfidfVectorizer(stop_words="english", max_features=1000)
    matrix = tfidf.fit_transform(books["title"].fillna(""))
    return matrix, books


def load_or_compute_nn(tfidf_matrix):
    from sklearn.neighbors import NearestNeighbors
    nn = NearestNeighbors(metric="cosine", algorithm="brute")
    nn.fit(tfidf_matrix)
    return nn


def load_or_compute_svd(ratings_df):
    from surprise import Dataset, Reader, SVD as SurpriseSVD
    reader = Reader(rating_scale=(1, 10))
    data = Dataset.load_from_df(
        ratings_df[["user_id", "book__isbn", "book_rating"]], reader
    )
    svd = SurpriseSVD(random_state=SEED)
    svd.fit(data.build_full_trainset())
    return svd


def content_based_recommendations(user_id, ratings_df, tfidf_matrix, books, nn,
                                   num_recommendations=10):
    from bookflix.ml.embeddings import LSAEmbedder
    embedder = load_embedder()
    if embedder is None:
        return books.head(num_recommendations)
    rated = set(get_user_all_rated_isbns(user_id, ratings_df))
    liked = get_user_rated_isbns(user_id, ratings_df)
    profile = embedder.build_user_profile(liked)
    if profile is None:
        return books.head(num_recommendations)
    import numpy as np
    candidates = books[~books["isbn"].isin(rated)].copy()
    vecs = np.stack([embedder.get(isbn) for isbn in candidates["isbn"]])
    candidates["_sim"] = vecs @ profile
    return candidates.nlargest(num_recommendations, "_sim")


def collaborative_filtering_recommendations(user_id, ratings_df, svd,
                                            num_recommendations=10):
    rated = set(get_user_all_rated_isbns(user_id, ratings_df))
    all_isbns = ratings_df["book__isbn"].unique()
    unrated = [i for i in all_isbns if i not in rated]
    preds = sorted([(i, svd.predict(user_id, i).est) for i in unrated],
                   key=lambda x: x[1], reverse=True)
    top = [isbn for isbn, _ in preds[:num_recommendations]]
    from bookflix.models import Book
    return Book.objects.filter(isbn__in=top)
