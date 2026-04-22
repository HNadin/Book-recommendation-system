"""
Sentiment-based rating correction — Section 2.5 of thesis.

Computes VADER compound sentiment per book from review text, then applies
a soft correction to raw ratings before model training.  This nudges
under- or over-rated books toward their true perceived quality.

Formula:
    adjusted_rating = clip(raw_rating + α · compound, 1, 10)

where compound ∈ [-1, 1] and α = 0.5 (empirically chosen).
"""

import numpy as np
import pandas as pd
import logging

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


def build_book_sentiment_map(reviews_df: pd.DataFrame) -> dict[str, float]:
    """
    reviews_df must have columns: isbn (or Book-ISBN), ReviewContent (or review_text).
    Returns {isbn: mean_compound_sentiment}.
    """
    isbn_col = "isbn" if "isbn" in reviews_df.columns else "Book-ISBN"
    text_col = next(
        (c for c in ("ReviewContent", "review_text", "text") if c in reviews_df.columns),
        None,
    )
    if text_col is None:
        logger.warning("No review text column found; skipping sentiment map.")
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


def label_sentiment(compound: float) -> str:
    if compound >= 0.05:
        return "Positive"
    if compound <= -0.05:
        return "Negative"
    return "Neutral"
