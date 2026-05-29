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
)
from bookflix.ml.hybrid import feature_combination_recommend, cascade_recommend
from bookflix.ml.evaluation import (
    compute_rmse,
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


def get_session_recommendations(
    session_ratings: dict,
    books_df: pd.DataFrame,
    top_n: int = 10,
) -> list:
    """
    Гібридні рекомендації для сесійного користувача (cold-start, Section 2.4).

    Оскільки сесійний користувач відсутній у навчальній вибірці NCF,
    його ID немає в ембедінгах моделі. Тому застосовується гібрид:

        score = W_SEM × semantic_sim + W_POP × normalized_avg_rating

    де:
      semantic_sim     — косинусна схожість між LSA-профілем користувача
                         (середній вектор вподобаних книг мінус вектор нелюбляних)
                         та LSA-вектором кандидата
      normalized_avg_rating — середній рейтинг книги в БД, нормалізований до [0,1],
                              виступає як колаборативний сигнал без явного NCF-ID

    Це і є демонстрація cold-start обробки з розділу 2.4 диплому.

    session_ratings: {isbn: rating_1_to_10}
    """
    W_SEM = 0.65
    W_POP = 0.35

    embedder = load_embedder()
    rated_set = set(session_ratings.keys())

    # --- Запасний варіант: лише популярність ---
    if embedder is None or not session_ratings:
        popular = books_df[~books_df["isbn"].isin(rated_set)].sort_values(
            "avg_rating", ascending=False
        )
        return [
            {"isbn": row["isbn"], "score": round(float(row["avg_rating"]), 3), "method": "popularity"}
            for _, row in popular.head(top_n).iterrows()
        ]

    # --- Будуємо профіль користувача ---
    liked = [isbn for isbn, r in session_ratings.items() if r >= 7]
    disliked = [isbn for isbn, r in session_ratings.items() if r <= 3]
    if not liked:
        # Якщо лайків немає — беремо все оцінене як базу
        liked = list(session_ratings.keys())

    user_profile = embedder.build_user_profile(liked)
    if user_profile is None:
        return []

    # Віднімаємо нелюбляні книги (контр-сигнал)
    if disliked:
        dislike_vec = embedder.build_user_profile(disliked)
        if dislike_vec is not None:
            user_profile = user_profile - 0.3 * dislike_vec
            norm = np.linalg.norm(user_profile)
            if norm > 0:
                user_profile = user_profile / norm

    # --- Кандидати ---
    candidates_df = books_df[~books_df["isbn"].isin(rated_set)].copy()
    if candidates_df.empty:
        return []

    candidate_isbns = candidates_df["isbn"].tolist()
    book_vecs = np.stack([embedder.get(isbn) for isbn in candidate_isbns])

    # --- Семантичний компонент ---
    sem_scores = book_vecs @ user_profile

    # --- Популярнісний / якісний компонент (нормалізований avg_rating) ---
    avg_ratings = candidates_df["avg_rating"].values.astype(float)
    max_r, min_r = avg_ratings.max(), avg_ratings.min()
    if max_r > min_r:
        pop_scores = (avg_ratings - min_r) / (max_r - min_r)
    else:
        pop_scores = np.zeros_like(avg_ratings)

    # --- Нормалізуємо семантичний компонент до [0,1] ---
    sem_min, sem_max = sem_scores.min(), sem_scores.max()
    if sem_max > sem_min:
        sem_norm = (sem_scores - sem_min) / (sem_max - sem_min)
    else:
        sem_norm = np.zeros_like(sem_scores)

    # --- Об'єднуємо ---
    combined = W_SEM * sem_norm + W_POP * pop_scores
    top_idx = np.argsort(combined)[::-1][:top_n]

    return [
        {
            "isbn": candidate_isbns[i],
            "score": round(float(combined[i]), 3),
            "sem_score": round(float(sem_scores[i]), 3),
            "pop_score": round(float(pop_scores[i]), 3),
            "method": "semantic+popularity hybrid",
        }
        for i in top_idx
    ]


def _normalize_array(arr: np.ndarray) -> np.ndarray:
    lo, hi = arr.min(), arr.max()
    return (arr - lo) / (hi - lo) if hi > lo else np.zeros_like(arr)


def get_session_recommendations_full_hybrid(
    session_ratings: dict,
    books_df: pd.DataFrame,
    top_n: int = 10,
) -> list:
    """
    Повний NCF + LSA гібрид для сесійного користувача через
    інференс ембедінгу (Section 2.3 + 2.4).

    Замість user_id використовується псевдо-ембедінг:
        user_emb_inferred = Σ (item_emb(isbn) × rating) / Σ rating

    Потім він подається напряму в MLP — точно так само як для відомих
    юзерів, але без необхідності перенавчання моделі.

    Якщо NCF не навчений → автоматично повертається до cold-start гібриду.

    Метод     |  Компоненти                          | Ваги
    ----------|--------------------------------------|----------
    Є NCF     |  NCF(inferred) + LSA semantic        | 0.6 + 0.4
    Немає NCF |  LSA semantic  + avg_rating          | 0.65 + 0.35
    """
    ncf_result = load_ncf()
    embedder = load_embedder()
    rated_set = set(session_ratings.keys())

    # --- Запасний варіант: cold-start без NCF ---
    if ncf_result is None:
        return get_session_recommendations(session_ratings, books_df, top_n)

    model, meta = ncf_result
    isbn_to_idx: dict = meta["isbn_to_idx"]

    # --- Будуємо item_ratings для інференсу ембедінгу ---
    item_ratings = [
        (isbn_to_idx[isbn], float(rating))
        for isbn, rating in session_ratings.items()
        if isbn in isbn_to_idx
    ]
    if not item_ratings:
        # Жодна оцінена книга не є в навчальній вибірці → cold-start
        return get_session_recommendations(session_ratings, books_df, top_n)

    # --- Виводимо псевдо-ембедінг користувача ---
    user_emb = model.infer_user_embedding(item_ratings)   # (1, emb_dim)

    # --- Кандидати ---
    candidates_df = books_df[~books_df["isbn"].isin(rated_set)].copy()
    if candidates_df.empty:
        return []

    candidate_isbns = [
        isbn for isbn in candidates_df["isbn"].tolist()
        if isbn in isbn_to_idx
    ]
    if not candidate_isbns:
        return get_session_recommendations(session_ratings, books_df, top_n)

    # --- NCF-компонент ---
    import torch
    idxs = torch.tensor([isbn_to_idx[isbn] for isbn in candidate_isbns], dtype=torch.long)
    with torch.no_grad():
        ncf_scores = model.forward_with_inferred_embedding(user_emb, idxs).numpy()

    # --- LSA-компонент ---
    if embedder is not None:
        liked = [isbn for isbn, r in session_ratings.items() if r >= 7]
        disliked = [isbn for isbn, r in session_ratings.items() if r <= 3]
        if not liked:
            liked = list(session_ratings.keys())
        user_profile = embedder.build_user_profile(liked)
        if user_profile is not None:
            if disliked:
                d_vec = embedder.build_user_profile(disliked)
                if d_vec is not None:
                    user_profile = user_profile - 0.3 * d_vec
                    n = np.linalg.norm(user_profile)
                    if n > 0:
                        user_profile /= n
            book_vecs = np.stack([embedder.get(isbn) for isbn in candidate_isbns])
            sem_scores = book_vecs @ user_profile
        else:
            sem_scores = np.zeros(len(candidate_isbns))
    else:
        sem_scores = np.zeros(len(candidate_isbns))

    # --- Feature Combination (ті ж ваги що й для відомих юзерів) ---
    ncf_norm = _normalize_array(ncf_scores)
    sem_norm = _normalize_array(sem_scores)
    combined = 0.6 * ncf_norm + 0.4 * sem_norm

    top_idx = np.argsort(combined)[::-1][:top_n]
    return [
        {
            "isbn": candidate_isbns[i],
            "score": round(float(combined[i]), 3),
            "ncf_score": round(float(ncf_scores[i]), 3),
            "sem_score": round(float(sem_scores[i]), 3),
            "method": "NCF (inferred) + LSA hybrid",
        }
        for i in top_idx
    ]


def get_book_semantic_neighbours(isbn: str, books_df: pd.DataFrame, top_n: int = 10) -> list:
    """Повертає семантично схожі книги для конкретного ISBN."""
    embedder = load_embedder()
    if embedder is None:
        return []
    emb = embedder.get(isbn)
    if not np.any(emb):
        return []
    all_isbns = books_df["isbn"].tolist()
    book_vecs = np.stack([embedder.get(i) for i in all_isbns])
    sims = book_vecs @ emb
    top_idx = np.argsort(sims)[::-1]
    results = []
    for i in top_idx:
        if all_isbns[i] != isbn:
            results.append({"isbn": all_isbns[i], "score": round(float(sims[i]), 3)})
        if len(results) == top_n:
            break
    return results


def train_test_split(ratings_df: pd.DataFrame, test_ratio: float = 0.2):
    """
    Per-rating split (Section 3.5): for each user, 20 % of their ratings
    go to the test set, 80 % stay in training.

    This ensures all users appear in training so every model can make
    predictions for every user — prerequisite for non-zero Precision@K
    and NDCG@K when models are evaluated against held-out liked items.
    """
    rng = random.Random(SEED)
    train_idx, test_idx = [], []
    for _, group in ratings_df.groupby("user_id"):
        idxs = group.index.tolist()
        rng.shuffle(idxs)
        split = max(1, int(len(idxs) * (1 - test_ratio)))
        train_idx.extend(idxs[:split])
        test_idx.extend(idxs[split:])
    return ratings_df.loc[train_idx].copy(), ratings_df.loc[test_idx].copy()


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


def content_based_recommendations(
        user_id, ratings_df, tfidf_matrix, books, nn,
        num_recommendations=10):
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
