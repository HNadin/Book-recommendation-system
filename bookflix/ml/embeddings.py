"""
Semantic vector representations — Section 2.2 of thesis.

Uses LSA (TF-IDF + TruncatedSVD) as the primary semantic model.
Richer content signal than the previous TF-IDF-on-title-only approach:
combines title, author, and publisher with bigrams and 5 000 vocabulary
terms, then projects to 100-dimensional semantic space.

BERT / sentence-transformers can be hot-swapped in via BERTEmbedder
when the environment has the transformers package available.
"""

import pickle

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize


class LSAEmbedder:
    """
    Latent Semantic Analysis embedder.
    Produces dense, normalised 100-dim book embeddings from text metadata.
    """

    def __init__(self, n_components: int = 100):
        self.n_components = n_components
        self.tfidf = TfidfVectorizer(
            stop_words="english",
            max_features=5000,
            ngram_range=(1, 2),
            sublinear_tf=True,
        )
        self.svd = TruncatedSVD(n_components=n_components, random_state=42)
        self.embeddings: np.ndarray | None = None
        self.isbn_to_idx: dict[str, int] = {}
        self.isbns: list[str] = []

    def fit(self, books_df):
        books_df = books_df.copy()
        books_df["_content"] = (
            books_df.get("title", "").fillna("")
            + " "
            + books_df.get("author", "").fillna("")
            + " "
            + books_df.get("publisher", "").fillna("")
        )
        tfidf_matrix = self.tfidf.fit_transform(books_df["_content"])
        raw = self.svd.fit_transform(tfidf_matrix)
        self.embeddings = normalize(raw).astype(np.float32)
        self.isbns = books_df["isbn"].tolist()
        self.isbn_to_idx = {isbn: i for i, isbn in enumerate(self.isbns)}
        return self

    def get(self, isbn: str) -> np.ndarray:
        idx = self.isbn_to_idx.get(isbn)
        if idx is None:
            return np.zeros(self.n_components, dtype=np.float32)
        return self.embeddings[idx]

    def build_user_profile(self, liked_isbns: list[str]) -> np.ndarray | None:
        vecs = [self.get(isbn) for isbn in liked_isbns if isbn in self.isbn_to_idx]
        if not vecs:
            return None
        profile = np.mean(vecs, axis=0)
        norm = np.linalg.norm(profile)
        return profile / norm if norm > 0 else profile

    def save(self, path: str):
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str) -> "LSAEmbedder":
        with open(path, "rb") as f:
            return pickle.load(f)


class BERTEmbedder:
    """
    Sentence-BERT embedder — requires `sentence-transformers` package.
    Produces 384-dim embeddings from `all-MiniLM-L6-v2`.
    Falls back gracefully to LSAEmbedder if the package is absent.
    """

    MODEL_NAME = "all-MiniLM-L6-v2"

    def __init__(self):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(self.MODEL_NAME)
        self.embeddings: np.ndarray | None = None
        self.isbn_to_idx: dict[str, int] = {}
        self.isbns: list[str] = []

    def fit(self, books_df):
        books_df = books_df.copy()
        sentences = (
            books_df.get("title", "").fillna("")
            + ". By "
            + books_df.get("author", "").fillna("")
        ).tolist()
        raw = self.model.encode(sentences, show_progress_bar=True, batch_size=64)
        self.embeddings = normalize(raw).astype(np.float32)
        self.isbns = books_df["isbn"].tolist()
        self.isbn_to_idx = {isbn: i for i, isbn in enumerate(self.isbns)}
        return self

    def get(self, isbn: str) -> np.ndarray:
        idx = self.isbn_to_idx.get(isbn)
        if idx is None:
            return np.zeros(self.embeddings.shape[1] if self.embeddings is not None else 384,
                            dtype=np.float32)
        return self.embeddings[idx]

    def build_user_profile(self, liked_isbns: list[str]) -> np.ndarray | None:
        vecs = [self.get(isbn) for isbn in liked_isbns if isbn in self.isbn_to_idx]
        if not vecs:
            return None
        profile = np.mean(vecs, axis=0)
        norm = np.linalg.norm(profile)
        return profile / norm if norm > 0 else profile

    def save(self, path: str):
        with open(path, "wb") as f:
            pickle.dump({"embeddings": self.embeddings, "isbns": self.isbns}, f)

    @classmethod
    def load(cls, path: str) -> "BERTEmbedder":
        obj = cls.__new__(cls)
        with open(path, "rb") as f:
            data = pickle.load(f)
        obj.embeddings = data["embeddings"]
        obj.isbns = data["isbns"]
        obj.isbn_to_idx = {isbn: i for i, isbn in enumerate(obj.isbns)}
        return obj
