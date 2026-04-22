"""
Feature Combination hybrid model — Section 2.4 of thesis.

Moves beyond the original cascade approach (content → SVD re-rank)
to a true meta-level Feature Combination that merges:
  1. NCF collaborative signal  (learned user–item interaction score)
  2. Semantic content signal   (cosine similarity via LSA embeddings)

The combined score for a (user, book) pair:
    score = w_ncf · ncf_score_norm + w_sem · semantic_similarity

Both components are normalised to [0, 1] across the candidate pool
before weighting so neither dominates by scale.
"""

import numpy as np
import logging

logger = logging.getLogger(__name__)

W_NCF = 0.6   # weight for collaborative signal
W_SEM = 0.4   # weight for semantic content signal


def _normalise(arr: np.ndarray) -> np.ndarray:
    lo, hi = arr.min(), arr.max()
    if hi == lo:
        return np.zeros_like(arr)
    return (arr - lo) / (hi - lo)


def feature_combination_recommend(
    user_id: int,
    candidate_isbns: list[str],
    embedder,
    rated_isbns: list[str],
    ncf_score_fn,
    top_n: int = 10,
    w_ncf: float = W_NCF,
    w_sem: float = W_SEM,
) -> list[dict]:
    """
    Parameters
    ----------
    user_id          : integer user ID
    candidate_isbns  : pool of candidate book ISBNs (unrated by user)
    embedder         : fitted LSAEmbedder or BERTEmbedder
    rated_isbns      : ISBNs of books the user rated >= RELEVANCE_THRESHOLD
    ncf_score_fn     : callable(user_id, isbn) -> float | None
    top_n            : number of recommendations to return

    Returns
    -------
    List of dicts: [{"isbn": ..., "ncf_score": ..., "sem_score": ..., "score": ...}, ...]
    """
    if not candidate_isbns:
        return []

    # --- Semantic component ---
    user_profile = embedder.build_user_profile(rated_isbns)
    if user_profile is None:
        sem_scores = np.zeros(len(candidate_isbns))
    else:
        book_vecs = np.stack([embedder.get(isbn) for isbn in candidate_isbns])
        # Cosine similarity (embeddings already L2-normalised)
        sem_scores = book_vecs @ user_profile

    # --- NCF component ---
    ncf_raw = np.array([
        ncf_score_fn(user_id, isbn) or 5.0
        for isbn in candidate_isbns
    ])

    # --- Normalise and combine ---
    ncf_norm = _normalise(ncf_raw)
    sem_norm = _normalise(sem_scores)
    combined = w_ncf * ncf_norm + w_sem * sem_norm

    # --- Rank and return top_n ---
    ranked_idx = np.argsort(combined)[::-1][:top_n]
    return [
        {
            "isbn": candidate_isbns[i],
            "ncf_score": round(float(ncf_raw[i]), 3),
            "sem_score": round(float(sem_scores[i]), 3),
            "score": round(float(combined[i]), 3),
        }
        for i in ranked_idx
    ]


def cascade_recommend(
    user_id: int,
    rated_isbns: list[str],
    all_isbns: list[str],
    embedder,
    ncf_score_fn,
    candidate_pool_size: int = 40,
    top_n: int = 10,
) -> list[dict]:
    """
    Original cascade approach kept for comparison (Section 1.2):
    1. Semantic content-based filtering → candidate pool
    2. NCF re-ranking of the pool
    """
    user_profile = embedder.build_user_profile(rated_isbns)
    unrated = [isbn for isbn in all_isbns if isbn not in set(rated_isbns)]

    if user_profile is None or not unrated:
        # Cold-start: return NCF-ranked unrated books
        scores = [(isbn, ncf_score_fn(user_id, isbn) or 0.0) for isbn in unrated[:200]]
        scores.sort(key=lambda x: x[1], reverse=True)
        return [{"isbn": isbn, "score": s} for isbn, s in scores[:top_n]]

    book_vecs = np.stack([embedder.get(isbn) for isbn in unrated])
    sim = book_vecs @ user_profile
    top_content_idx = np.argsort(sim)[::-1][:candidate_pool_size]
    candidates = [unrated[i] for i in top_content_idx]

    scores = [(isbn, ncf_score_fn(user_id, isbn) or 0.0) for isbn in candidates]
    scores.sort(key=lambda x: x[1], reverse=True)
    return [{"isbn": isbn, "score": s} for isbn, s in scores[:top_n]]
