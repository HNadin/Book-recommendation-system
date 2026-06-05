"""
Sentiment-based rating correction — Section 2.5 of thesis.

Computes VADER compound sentiment per book from review text, then applies
a soft correction to raw ratings before model training.  This nudges
under- or over-rated books toward their true perceived quality.

Formula:
    adjusted_rating = clip(raw_rating + α · compound, 1, 10)

where compound ∈ [-1, 1] and α = 0.5 (empirically chosen).
"""

import logging
import re

import pandas as pd

logger = logging.getLogger(__name__)
ALPHA = 0.5


def _vader_compound(text: str) -> float:
    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        _vader_compound._analyzer = getattr(
            _vader_compound, "_analyzer",
            SentimentIntensityAnalyzer()
        )
        return _vader_compound._analyzer.polarity_scores(str(text))["compound"]
    except ImportError:
        pass

    try:
        from textblob import TextBlob
        return float(TextBlob(str(text)).sentiment.polarity)
    except ImportError:
        pass

    return 0.0


def _normalise_title(title: str) -> str:
    """Lowercase, strip punctuation and extra spaces for fuzzy title matching."""
    t = str(title).lower()
    t = re.sub(r"[^a-z0-9\s]", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def build_amazon_sentiment_map(
    amazon_df: pd.DataFrame,
    books_df: pd.DataFrame,
    min_reviews: int = 3,
) -> dict[str, float]:
    """
    Build a {isbn: mean_compound} map from the Amazon Books Reviews dataset
    (Books_rating.csv).  Expected columns: Id, Title, review/text, review/summary.

    Matching strategy (two passes):
      1. Direct ISBN match: amazon Id (ISBN-10) == books_df isbn.
      2. Normalised-title match for books not covered by ISBN.

    Only books with >= min_reviews reviews contribute to the map.
    """
    # Rename columns to internal names regardless of case/slash variants
    col_map = {}
    for c in amazon_df.columns:
        lc = c.lower().replace("/", "_").replace(" ", "_")
        if lc in ("id", "asin"):
            col_map[c] = "_isbn"
        elif lc == "title":
            col_map[c] = "_title"
        elif lc in ("review_text", "reviewtext", "text"):
            col_map[c] = "_text"
        elif lc in ("review_summary", "reviewsummary", "summary"):
            col_map[c] = "_summary"
    amz = amazon_df.rename(columns=col_map).copy()

    if "_text" not in amz.columns and "_summary" not in amz.columns:
        logger.warning("Amazon CSV has no review text column — skipping.")
        return {}

    # Combine text + summary for richer VADER input
    text_parts = []
    if "_summary" in amz.columns:
        text_parts.append(amz["_summary"].fillna(""))
    if "_text" in amz.columns:
        text_parts.append(amz["_text"].fillna(""))
    amz["_combined"] = " ".join(text_parts) if len(text_parts) == 1 else (
        text_parts[0] + " " + text_parts[1] if len(text_parts) == 2 else pd.Series("")
    )
    # Correct column-wise concat
    if len(text_parts) == 2:
        amz["_combined"] = text_parts[0] + " " + text_parts[1]
    elif len(text_parts) == 1:
        amz["_combined"] = text_parts[0]

    amz = amz.dropna(subset=["_combined"])
    amz = amz[amz["_combined"].str.strip() != ""]

    logger.info("Computing VADER on %d Amazon reviews…", len(amz))
    amz["_compound"] = amz["_combined"].apply(_vader_compound)

    # --- Pass 1: match by ISBN ---
    isbn_set = set(books_df["isbn"].astype(str))
    isbn_col_present = "_isbn" in amz.columns

    sentiment_map: dict[str, float] = {}
    matched_isbns: set[str] = set()

    if isbn_col_present:
        amz["_isbn"] = amz["_isbn"].astype(str).str.strip()
        isbn_grp = amz[amz["_isbn"].isin(isbn_set)].groupby("_isbn")["_compound"]
        counts = isbn_grp.count()
        means = isbn_grp.mean()
        for isbn, cnt in counts.items():
            if cnt >= min_reviews:
                sentiment_map[isbn] = float(means[isbn])
                matched_isbns.add(isbn)
        logger.info("Pass 1 (ISBN match): %d books.", len(sentiment_map))

    # --- Pass 2: normalised-title match for remaining books ---
    title_col_present = "_title" in amz.columns
    books_with_title = "title" in books_df.columns

    if title_col_present and books_with_title:
        remaining_books = books_df[~books_df["isbn"].astype(str).isin(matched_isbns)].copy()
        remaining_books["_norm_title"] = remaining_books["title"].apply(_normalise_title)

        amz["_norm_title"] = amz["_title"].apply(_normalise_title)
        title_grp = amz.groupby("_norm_title")["_compound"]
        title_counts = title_grp.count()
        title_means = title_grp.mean()

        for _, row in remaining_books.iterrows():
            nt = row["_norm_title"]
            if nt in title_counts and title_counts[nt] >= min_reviews:
                sentiment_map[str(row["isbn"])] = float(title_means[nt])

        logger.info(
            "Pass 2 (title match): %d books added. Total: %d.",
            len(sentiment_map) - len(matched_isbns), len(sentiment_map),
        )

    return sentiment_map


def build_book_sentiment_map(reviews_df: pd.DataFrame) -> dict[str, float]:
    """
    reviews_df must have columns: isbn (or Book-ISBN), ReviewContent (or review_text).
    Returns {isbn: mean_compound_sentiment}.
    """
    isbn_col = next(
        (c for c in ("isbn", "ISBN", "Book-ISBN", "ProductId", "book_isbn", "ASIN", "asin")
         if c in reviews_df.columns),
        None,
    )
    text_col = next(
        (c for c in ("ReviewContent", "review_text", "Text", "Summary", "text")
         if c in reviews_df.columns),
        None,
    )
    if isbn_col is None or text_col is None:
        logger.warning(
            "Cannot build sentiment map: isbn_col=%s text_col=%s. Columns: %s",
            isbn_col, text_col, reviews_df.columns.tolist(),
        )
        return {}

    logger.info("Computing VADER sentiment for %d reviews…", len(reviews_df))
    reviews_df = reviews_df.dropna(subset=[isbn_col, text_col]).copy()
    reviews_df["_compound"] = reviews_df[text_col].apply(_vader_compound)
    sentiment_map = reviews_df.groupby(isbn_col)["_compound"].mean().to_dict()
    logger.info("Sentiment map built for %d books.", len(sentiment_map))
    return sentiment_map


def apply_sentiment_correction(
    ratings_df: pd.DataFrame,
    sentiment_map: dict[str, float],
    alpha: float = ALPHA,
) -> pd.DataFrame:
    """
    Adds an `adjusted_rating` column to ratings_df.
    Ratings without a sentiment entry are left unchanged.
    """
    isbn_col = "book__isbn"
    ratings_df = ratings_df.copy()
    ratings_df["sentiment_score"] = (
        ratings_df[isbn_col].map(sentiment_map).fillna(0.0)
    )
    ratings_df["adjusted_rating"] = (
        ratings_df["book_rating"] + alpha * ratings_df["sentiment_score"]
    ).clip(1, 10)
    return ratings_df


def build_metadata_sentiment_map(books_df) -> dict[str, float]:
    """
    Fallback when no ISBN-linked review corpus is available.
    Computes VADER compound sentiment from book title + author text.
    This is a weak but honest signal — documented in Section 2.5 of thesis.
    """
    rows = []
    for _, row in books_df.iterrows():
        text = f"{row.get('title', '')}. By {row.get('author', '')}".strip()
        compound = _vader_compound(text)
        rows.append((str(row["isbn"]), compound))
    result = {isbn: score for isbn, score in rows if isbn}
    logger.info("Metadata sentiment map built for %d books.", len(result))
    return result


def label_sentiment(compound: float) -> str:
    if compound >= 0.05:
        return "Positive"
    if compound <= -0.05:
        return "Negative"
    return "Neutral"



def _vader_compound(text: str) -> float:
    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        _vader_compound._analyzer = getattr(
            _vader_compound, "_analyzer",
            SentimentIntensityAnalyzer()
        )
        return _vader_compound._analyzer.polarity_scores(str(text))["compound"]
    except ImportError:
        pass

    try:
        from textblob import TextBlob
        return float(TextBlob(str(text)).sentiment.polarity)
    except ImportError:
        pass

    return 0.0


def build_book_sentiment_map(reviews_df: pd.DataFrame) -> dict[str, float]:
    """
    reviews_df must have columns: isbn (or Book-ISBN), ReviewContent (or review_text).
    Returns {isbn: mean_compound_sentiment}.
    """
    isbn_col = next(
        (c for c in ("isbn", "ISBN", "Book-ISBN", "ProductId", "book_isbn", "ASIN", "asin")
         if c in reviews_df.columns),
        None,
    )
    text_col = next(
        (c for c in ("ReviewContent", "review_text", "Text", "Summary", "text")
         if c in reviews_df.columns),
        None,
    )
    if isbn_col is None or text_col is None:
        logger.warning(
            "Cannot build sentiment map: isbn_col=%s text_col=%s. Columns: %s",
            isbn_col, text_col, reviews_df.columns.tolist(),
        )
        return {}

    logger.info("Computing VADER sentiment for %d reviews…", len(reviews_df))
    reviews_df = reviews_df.dropna(subset=[isbn_col, text_col]).copy()
    reviews_df["_compound"] = reviews_df[text_col].apply(_vader_compound)
    sentiment_map = reviews_df.groupby(isbn_col)["_compound"].mean().to_dict()
    logger.info("Sentiment map built for %d books.", len(sentiment_map))
    return sentiment_map


def apply_sentiment_correction(
    ratings_df: pd.DataFrame,
    sentiment_map: dict[str, float],
    alpha: float = ALPHA,
) -> pd.DataFrame:
    """
    Adds an `adjusted_rating` column to ratings_df.
    Ratings without a sentiment entry are left unchanged.
    """
    isbn_col = "book__isbn"
    ratings_df = ratings_df.copy()
    ratings_df["sentiment_score"] = (
        ratings_df[isbn_col].map(sentiment_map).fillna(0.0)
    )
    ratings_df["adjusted_rating"] = (
        ratings_df["book_rating"] + alpha * ratings_df["sentiment_score"]
    ).clip(1, 10)
    return ratings_df


def build_metadata_sentiment_map(books_df) -> dict[str, float]:
    """
    Fallback when no ISBN-linked review corpus is available.
    Computes VADER compound sentiment from book title + author text.
    This is a weak but honest signal — documented in Section 2.5 of thesis.
    """
    rows = []
    for _, row in books_df.iterrows():
        text = f"{row.get('title', '')}. By {row.get('author', '')}".strip()
        compound = _vader_compound(text)
        rows.append((str(row["isbn"]), compound))
    result = {isbn: score for isbn, score in rows if isbn}
    logger.info("Metadata sentiment map built for %d books.", len(result))
    return result


def label_sentiment(compound: float) -> str:
    if compound >= 0.05:
        return "Positive"
    if compound <= -0.05:
        return "Negative"
    return "Neutral"
