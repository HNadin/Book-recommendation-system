"""
Django views for BOOKFLIX recommendation system.

Endpoints
---------
GET  /                         → landing page
GET  /dashboard/               → analytics + sentiment dashboard
GET  /ratingsrecommend/        → user-ID entry page
GET  /ratings/<user_id>/       → user's rated books
GET  /recommendations/<id>/    → personalised recommendations (HTML)
GET  /evaluate/                → model comparison dashboard (Section 3.5)
GET  /api/recommendations/<id>/    → JSON: top-N recommendations
GET  /api/evaluate/                → JSON: full metrics comparison table
"""

import logging
import random
import os
import pickle

import pandas as pd
import numpy as np
from django.shortcuts import render, get_object_or_404
from django.http import JsonResponse
from django.db.models import Count, Avg
from rest_framework.decorators import api_view
from rest_framework.response import Response

from .models import User, Rating, Book
from .recommendation_algorithms import (
    load_data_ratings,
    load_books_df,
    get_user_all_rated_isbns,
    get_user_rated_isbns,
    hybrid_recommendations,
    evaluate_user_model,
)
from .ml.model_store import load_eval_results, MODELS_DIR

logger = logging.getLogger(__name__)
SEED = 42
random.seed(SEED)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_random_user_ids(n: int = 10) -> list[int]:
    users = User.objects.annotate(n=Count("rating")).filter(n__gte=5)
    ids = list(users.values_list("user_id", flat=True))
    random.shuffle(ids)
    return ids[:n]


def _enrich_with_book_objects(isbn_list: list[str]) -> list[dict]:
    book_map = {b.isbn: b for b in Book.objects.filter(isbn__in=isbn_list)}
    enriched = []
    for isbn in isbn_list:
        b = book_map.get(isbn)
        if b:
            enriched.append({
                "isbn": isbn,
                "title": b.title,
                "author": b.author,
                "year_of_publication": b.year_of_publication,
                "image_url_m": b.image_url_m,
            })
    return enriched


# ---------------------------------------------------------------------------
# Page views
# ---------------------------------------------------------------------------

def home(request):
    return render(request, "home.html")


def dashboard(request):
    """
    Analytics dashboard — Section 3.3.
    Shows dataset statistics, sentiment distribution, and top books.
    """
    context = {}

    # Dataset stats from DB
    context["num_books"] = Book.objects.count()
    context["num_users"] = User.objects.count()
    context["num_ratings"] = Rating.objects.count()

    # Top-rated and most-rated books
    popular = (
        Book.objects.annotate(n=Count("rating"), avg=Avg("rating__book_rating"))
        .filter(n__gte=10)
        .order_by("-n")[:5]
    )
    top_rated = (
        Book.objects.annotate(n=Count("rating"), avg=Avg("rating__book_rating"))
        .filter(n__gte=50)
        .order_by("-avg")[:5]
    )
    context["popular_books"] = popular
    context["top_rated_books"] = top_rated

    # Rating distribution (for Chart.js)
    dist = (
        Rating.objects.values("book_rating")
        .annotate(count=Count("id"))
        .order_by("book_rating")
    )
    context["rating_dist_labels"] = [d["book_rating"] for d in dist]
    context["rating_dist_values"] = [d["count"] for d in dist]

    # Sentiment data (from cached sentiments.csv if present)
    sentiment_data = _load_sentiment_summary()
    context.update(sentiment_data)

    # Evaluation results (from trained_models/eval_results.json)
    context["eval_results"] = load_eval_results()

    return render(request, "dashboard.html", context)


def homepage_view(request):
    return render(request, "ratings_recommend.html", {
        "random_user_ids": _get_random_user_ids(),
        "selected_user_id": None,
    })


def user_ratings_view(request, user_id: int):
    user = get_object_or_404(User, user_id=user_id)
    user_ratings = Rating.objects.filter(user=user).select_related("book")
    return render(request, "user_ratings.html", {
        "user_id": user_id,
        "user_ratings": user_ratings,
        "random_user_ids": _get_random_user_ids(),
        "selected_user_id": user_id,
    })


def user_recommendations_view(request, user_id: int):
    """
    Personalised recommendations page — calls the internal API and renders HTML.
    Supports ?mode=hybrid (default) | cascade | svd via query param.
    """
    use_sentiment = request.GET.get("sentiment", "0") == "1"

    try:
        ratings_df = load_data_ratings()
        books_df = load_books_df()
        ranked = hybrid_recommendations(
            user_id, ratings_df, books_df, top_n=10,
            use_sentiment_adjusted=use_sentiment,
        )
        isbn_list = [r["isbn"] for r in ranked]
        enriched = _enrich_with_book_objects(isbn_list)

        # Attach scores from ranked list
        score_map = {r["isbn"]: r for r in ranked}
        for item in enriched:
            item.update(score_map.get(item["isbn"], {}))

        metrics, err = evaluate_user_model(user_id, ratings_df)
    except Exception as e:
        logger.exception("Error generating recommendations for user %s", user_id)
        enriched = []
        metrics = None
        err = str(e)

    return render(request, "user_recommendations.html", {
        "user_id": user_id,
        "recommendations": enriched,
        "random_user_ids": _get_random_user_ids(),
        "selected_user_id": user_id,
        "metrics": metrics,
        "use_sentiment": use_sentiment,
        "error": err,
    })


def evaluate_view(request):
    """
    Model comparison dashboard — Section 3.5 of thesis.
    Loads pre-computed metrics from eval_results.json.
    """
    results = load_eval_results()
    models_list = []
    for name, metrics in results.items():
        models_list.append({"name": name, "metrics": metrics})

    return render(request, "evaluate.html", {
        "models": models_list,
        "has_results": bool(results),
    })


# ---------------------------------------------------------------------------
# REST API endpoints
# ---------------------------------------------------------------------------

@api_view(["GET"])
def api_recommendations(request, user_id: int):
    if not User.objects.filter(user_id=user_id).exists():
        return Response({"error": "User not found"}, status=404)

    use_sentiment = request.GET.get("sentiment", "0") == "1"

    try:
        ratings_df = load_data_ratings()
        books_df = load_books_df()
        ranked = hybrid_recommendations(
            user_id, ratings_df, books_df, top_n=10,
            use_sentiment_adjusted=use_sentiment,
        )
        isbn_list = [r["isbn"] for r in ranked]
        enriched = _enrich_with_book_objects(isbn_list)

        score_map = {r["isbn"]: r for r in ranked}
        for item in enriched:
            item.update(score_map.get(item["isbn"], {}))

        metrics, err = evaluate_user_model(user_id, ratings_df)
    except Exception as e:
        logger.exception("API error for user %s", user_id)
        return Response({"error": str(e)}, status=500)

    return Response({
        "user_id": user_id,
        "recommendations": enriched,
        "metrics": metrics,
        "sentiment_corrected": use_sentiment,
    })


@api_view(["GET"])
def api_evaluate(request):
    """Return the full model comparison metrics table as JSON."""
    return Response(load_eval_results())


# ---------------------------------------------------------------------------
# Legacy endpoint alias (keeps old URL working during transition)
# ---------------------------------------------------------------------------

@api_view(["GET"])
def fetch_hybrid_recommendations(request, user_id: int):
    return api_recommendations(request, user_id)


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------

def _load_sentiment_summary() -> dict:
    sentiment_csv = "sentiments.csv"
    if not os.path.exists(sentiment_csv):
        return {"sentiment_available": False}

    try:
        df = pd.read_csv(sentiment_csv)
        counts = df["analysis"].value_counts().to_dict() if "analysis" in df.columns else {}
        polarity_mean = float(df["polarity"].mean()) if "polarity" in df.columns else None
        return {
            "sentiment_available": True,
            "sentiment_positive": counts.get("Positive", 0),
            "sentiment_neutral": counts.get("Neutral", 0),
            "sentiment_negative": counts.get("Negative", 0),
            "polarity_mean": round(polarity_mean, 3) if polarity_mean is not None else None,
        }
    except Exception as e:
        logger.warning("Could not load sentiment CSV: %s", e)
        return {"sentiment_available": False}
