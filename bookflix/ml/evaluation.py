"""
Ranking and rating quality metrics — Section 3.5 of thesis.

Replaces the original MSE-only evaluation with a full suite:
  • RMSE  — rating prediction accuracy
  • Precision@K — fraction of top-K recommendations that are relevant
  • NDCG@K — normalised discounted cumulative gain (ranking quality)

A book is considered "relevant" if its true rating ≥ RELEVANCE_THRESHOLD.
"""

import math
import numpy as np
from sklearn.metrics import mean_squared_error
from typing import Callable

RELEVANCE_THRESHOLD = 7  # ratings ≥ 7 are "liked"


# ---------------------------------------------------------------------------
# Rating-quality metric
# ---------------------------------------------------------------------------

def compute_rmse(actuals: list[float], predictions: list[float]) -> float:
    return float(np.sqrt(mean_squared_error(actuals, predictions)))


# ---------------------------------------------------------------------------
# Ranking metrics
# ---------------------------------------------------------------------------

def _dcg_at_k(relevances: list[int], k: int) -> float:
    relevances = relevances[:k]
    return sum(
        rel / math.log2(rank + 2)
        for rank, rel in enumerate(relevances)
    )


def compute_precision_at_k(
    user_recommendations: dict[int, list[str]],
    user_relevant_items: dict[int, set[str]],
    k: int = 10,
) -> float:
    """
    user_recommendations: {user_id: [isbn, ...]} ordered best-first
    user_relevant_items:  {user_id: {isbn, ...}} ground-truth liked books
    """
    scores = []
    for uid, recs in user_recommendations.items():
        relevant = user_relevant_items.get(uid, set())
        if not relevant:
            continue
        hits = sum(1 for isbn in recs[:k] if isbn in relevant)
        scores.append(hits / k)
    return float(np.mean(scores)) if scores else 0.0


def compute_ndcg_at_k(
    user_recommendations: dict[int, list[str]],
    user_relevant_items: dict[int, set[str]],
    k: int = 10,
) -> float:
    scores = []
    for uid, recs in user_recommendations.items():
        relevant = user_relevant_items.get(uid, set())
        if not relevant:
            continue
        gains = [1 if isbn in relevant else 0 for isbn in recs[:k]]
        dcg = _dcg_at_k(gains, k)
        ideal = _dcg_at_k([1] * min(len(relevant), k), k)
        if ideal > 0:
            scores.append(dcg / ideal)
    return float(np.mean(scores)) if scores else 0.0


# ---------------------------------------------------------------------------
# Convenience: evaluate a predict_fn on a test split
# ---------------------------------------------------------------------------

def evaluate_rating_model(
    test_df,
    predict_fn: Callable[[int, str], float],
) -> dict[str, float]:
    """
    test_df: DataFrame with columns [user_id, book__isbn, book_rating]
    predict_fn: (user_id, isbn) -> predicted_rating
    """
    actuals, preds = [], []
    for _, row in test_df.iterrows():
        try:
            p = predict_fn(int(row["user_id"]), str(row["book__isbn"]))
            actuals.append(float(row["book_rating"]))
            preds.append(float(p))
        except Exception:
            pass
    if not actuals:
        return {"rmse": None}
    return {"rmse": compute_rmse(actuals, preds)}


def evaluate_ranking_model(
    test_df,
    recommend_fn: Callable[[int, int], list[str]],
    k: int = 10,
) -> dict[str, float]:
    """
    test_df: DataFrame with columns [user_id, book__isbn, book_rating]
    recommend_fn: (user_id, n) -> [isbn, ...]
    """
    user_relevant: dict[int, set[str]] = {}
    for uid, group in test_df.groupby("user_id"):
        liked = set(group.loc[group["book_rating"] >= RELEVANCE_THRESHOLD, "book__isbn"])
        if liked:
            user_relevant[int(uid)] = liked

    user_recs: dict[int, list[str]] = {}
    for uid in list(user_relevant.keys())[:200]:  # cap to avoid slow eval
        try:
            user_recs[uid] = recommend_fn(uid, k)
        except Exception:
            pass

    return {
        f"precision_at_{k}": compute_precision_at_k(user_recs, user_relevant, k),
        f"ndcg_at_{k}": compute_ndcg_at_k(user_recs, user_relevant, k),
    }


def full_evaluation(
    test_df,
    predict_fn: Callable[[int, str], float],
    recommend_fn: Callable[[int, int], list[str]],
    k: int = 10,
) -> dict[str, float]:
    metrics = {}
    metrics.update(evaluate_rating_model(test_df, predict_fn))
    metrics.update(evaluate_ranking_model(test_df, recommend_fn, k))
    return metrics
