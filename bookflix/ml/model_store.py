"""
Singleton model store — Section 3.4 of thesis.

All models are trained offline via the `train_models` management command
and saved to disk.  At request time they are loaded once and cached in
memory, eliminating the per-request re-training that made the original
system unusable.

Stored artefacts (under bookflix/trained_models/):
  ncf_model.pt        — NCF PyTorch weights
  ncf_meta.pkl        — user/item index mappings + hyperparameters
  bert_embedder.pkl   — fitted BERTEmbedder (all-MiniLM-L6-v2 embeddings)
  svd_baseline.pkl    — SVD baseline (for comparison table)
  eval_results.json   — cached metric comparison across all model variants
"""

import json
import logging
import os
import pickle

import numpy as np

logger = logging.getLogger(__name__)

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "trained_models")


def _path(name: str) -> str:
    os.makedirs(MODELS_DIR, exist_ok=True)
    return os.path.join(MODELS_DIR, name)


# ---------------------------------------------------------------------------
# In-process cache (populated lazily on first request)
# ---------------------------------------------------------------------------
_cache: dict = {}


def _cached(key: str, loader):
    if key not in _cache:
        result = loader()
        if result is not None:
            _cache[key] = result
    return _cache.get(key)


# ---------------------------------------------------------------------------
# NCF
# ---------------------------------------------------------------------------

def load_ncf():
    def _load():
        pt_path = _path("ncf_model.pt")
        meta_path = _path("ncf_meta.pkl")
        if not os.path.exists(pt_path) or not os.path.exists(meta_path):
            logger.warning("NCF model not found — run `python manage.py train_models`.")
            return None
        import torch

        from bookflix.ml.ncf import NCF
        with open(meta_path, "rb") as f:
            meta = pickle.load(f)
        model = NCF(
            num_users=meta["num_users"],
            num_items=meta["num_items"],
            embedding_dim=meta.get("embedding_dim", 32),
        )
        model.load_state_dict(torch.load(pt_path, map_location="cpu"))
        model.eval()
        logger.info("NCF model loaded.")
        return model, meta

    return _cached("ncf", _load)


def load_embedder():
    def _load():
        bert_path = _path("bert_embedder.pkl")
        if os.path.exists(bert_path):
            from bookflix.ml.embeddings import BERTEmbedder
            emb = BERTEmbedder.load(bert_path)
            logger.info("BERT embedder loaded.")
            return emb
        # Backward-compat: fall back to legacy LSA artefact if present
        lsa_path = _path("lsa_embedder.pkl")
        if os.path.exists(lsa_path):
            from bookflix.ml.embeddings import LSAEmbedder
            emb = LSAEmbedder.load(lsa_path)
            logger.info("LSA embedder loaded (legacy).")
            return emb
        logger.warning("No embedder found — run `python manage.py train_models`.")
        return None

    return _cached("embedder", _load)


def load_svd_baseline():
    def _load():
        path = _path("svd_baseline.pkl")
        if not os.path.exists(path):
            return None
        with open(path, "rb") as f:
            svd = pickle.load(f)
        logger.info("SVD baseline loaded.")
        return svd

    return _cached("svd", _load)


def load_eval_results() -> dict:
    path = _path("eval_results.json")
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_eval_results(results: dict):
    with open(_path("eval_results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)


# ---------------------------------------------------------------------------
# NCF-based predict / recommend helpers (used in views)
# ---------------------------------------------------------------------------

def ncf_predict(user_id: int, isbn: str) -> float | None:
    result = load_ncf()
    if result is None:
        return None
    model, meta = result
    uid_map: dict = meta["user_to_idx"]
    iid_map: dict = meta["isbn_to_idx"]
    u = uid_map.get(user_id)
    v = iid_map.get(isbn)
    if u is None or v is None:
        return None
    import torch
    with torch.no_grad():
        score = model(torch.tensor([u]), torch.tensor([v])).item()
    # Scale sigmoid-like output back to [1, 10]
    return float(np.clip(score, 1.0, 10.0))


def ncf_recommend(user_id: int, isbn_pool: list[str], top_n: int = 10) -> list[str]:
    """Return top_n ISBNs from isbn_pool ranked by NCF score."""
    result = load_ncf()
    if result is None:
        return isbn_pool[:top_n]
    model, meta = result
    uid_map = meta["user_to_idx"]
    iid_map = meta["isbn_to_idx"]
    u = uid_map.get(user_id)
    if u is None:
        return isbn_pool[:top_n]
    import torch
    mapped = [(isbn, iid_map[isbn]) for isbn in isbn_pool if isbn in iid_map]
    if not mapped:
        return isbn_pool[:top_n]
    isbns_m, idxs = zip(*mapped)
    u_tensor = torch.tensor([u] * len(idxs))
    v_tensor = torch.tensor(list(idxs))
    with torch.no_grad():
        scores = model(u_tensor, v_tensor).numpy()
    ranked = sorted(zip(isbns_m, scores), key=lambda x: x[1], reverse=True)
    return [isbn for isbn, _ in ranked[:top_n]]


def clear_cache():
    _cache.clear()
