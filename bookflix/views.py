"""
Django views for BOOKFLIX recommendation system.

Endpoints
---------
GET  /                              → landing page
GET  /dashboard/                    → analytics + sentiment dashboard
GET  /taste/                        → profile builder (session-based)
POST /api/session/rate/             → AJAX: add/remove rating to session profile
DELETE /api/session/rate/<isbn>/    → AJAX: remove specific rating
GET  /explore/                      → book search + semantic neighbours
GET  /my-recommendations/           → recommendations for session user
GET  /ratingsrecommend/             → user-ID entry page (legacy)
GET  /ratings/<user_id>/            → user's rated books
GET  /recommendations/<id>/         → personalised recommendations (HTML)
GET  /evaluate/                     → model comparison dashboard (Section 3.5)
GET  /api/recommendations/<id>/     → JSON: top-N recommendations
GET  /api/evaluate/                 → JSON: full metrics comparison table
"""

import json
import logging
import os
import random

from django.db.models import Avg, Count, Q
from django.http import JsonResponse
from django.shortcuts import get_object_or_404, render
from django.views.decorators.http import require_http_methods

import pandas as pd
from rest_framework.decorators import api_view
from rest_framework.response import Response

from .ml.model_store import load_eval_results
from .models import Book, Rating, User
from .recommendation_algorithms import (
    evaluate_user_model,
    get_book_semantic_neighbours,
    get_session_recommendations_full_hybrid,
    hybrid_recommendations,
    load_books_df,
    load_data_ratings,
)

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
    import json as _json
    results = load_eval_results()
    models_list = []
    chart_precisions = []
    chart_ndcgs = []
    for name, metrics in results.items():
        models_list.append({"name": name, "metrics": metrics})
        chart_precisions.append(next((v for k, v in metrics.items() if "precision" in k), 0) or 0)
        chart_ndcgs.append(next((v for k, v in metrics.items() if "ndcg" in k), 0) or 0)

    return render(request, "evaluate.html", {
        "models": models_list,
        "has_results": bool(results),
        "chart_labels_json": _json.dumps([m["name"] for m in models_list]),
        "chart_precisions_json": _json.dumps(chart_precisions),
        "chart_ndcgs_json": _json.dumps(chart_ndcgs),
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


def _load_goodreads_map() -> dict:
    """Завантажує Goodreads-рейтинги з final_books3.csv якщо файл є."""
    for path in ("final_books3.csv", "data/final_books3.csv"):
        if os.path.exists(path):
            try:
                df = pd.read_csv(path, low_memory=False)
                # isbn може бути відсутнім — пробуємо різні назви колонок
                isbn_col = next((c for c in ("isbn", "ISBN", "isbn13") if c in df.columns), None)
                if isbn_col is None:
                    return {}
                result = {}
                for _, row in df.iterrows():
                    isbn = str(row[isbn_col]).strip()
                    has_avg = "average_rating" in df.columns and pd.notna(row.get("average_rating"))
                    has_cnt = "ratings_count" in df.columns and pd.notna(row.get("ratings_count"))
                    result[isbn] = {
                        "rating": round(float(row["average_rating"]), 2) if has_avg else None,
                        "count": int(row["ratings_count"]) if has_cnt else None,
                    }
                return result
            except Exception as e:
                logger.warning("Could not load Goodreads CSV: %s", e)
    return {}


_goodreads_cache: dict = {}


def _get_goodreads(isbn: str) -> dict | None:
    global _goodreads_cache
    if not _goodreads_cache:
        _goodreads_cache = _load_goodreads_map()
    return _goodreads_cache.get(isbn)


def _enrich_books_with_stats(isbn_list: list) -> list:
    """Збагачує список ISBN об'єктами Book + статистикою + Goodreads."""
    book_map = {
        b.isbn: b
        for b in Book.objects.filter(isbn__in=isbn_list)
        .annotate(n=Count("rating"), avg=Avg("rating__book_rating"))
    }
    result = []
    for isbn in isbn_list:
        b = book_map.get(isbn)
        if not b:
            continue
        result.append({
            "isbn": isbn,
            "title": b.title,
            "author": b.author,
            "year_of_publication": b.year_of_publication,
            "image_url_m": b.image_url_m,
            "bx_rating": round(b.avg, 1) if b.avg else None,
            "bx_count": b.n,
            "goodreads": _get_goodreads(isbn),
        })
    return result


# ---------------------------------------------------------------------------
# Варіант Б: сесійний профіль
# ---------------------------------------------------------------------------

SESSION_KEY = "bookflix_profile"
MIN_RATINGS_FOR_RECS = 3


def taste_view(request):
    """
    Сторінка формування смаку — користувач оцінює книги,
    оцінки зберігаються в сесії.
    """
    profile: dict = request.session.get(SESSION_KEY, {})

    # Показуємо 24 випадкові книги з обкладинками, яких ще не оцінено
    exclude_isbns = list(profile.keys())
    books_qs = (
        Book.objects
        .exclude(isbn__in=exclude_isbns)
        .exclude(image_url_m="")
        .filter(image_url_m__startswith="http")
        .order_by("?")[:24]
    )

    # Вже оцінені книги для бічної панелі
    rated_books = []
    if profile:
        rated_qs = Book.objects.filter(isbn__in=list(profile.keys()))
        rated_map = {b.isbn: b for b in rated_qs}
        for isbn, rating in profile.items():
            b = rated_map.get(isbn)
            if b:
                rated_books.append({"book": b, "rating": rating})
        rated_books.sort(key=lambda x: x["rating"], reverse=True)

    return render(request, "taste.html", {
        "books": books_qs,
        "rated_books": rated_books,
        "profile_count": len(profile),
        "min_ratings": MIN_RATINGS_FOR_RECS,
        "ready": len(profile) >= MIN_RATINGS_FOR_RECS,
    })


@require_http_methods(["POST"])
def api_session_rate(request):
    """AJAX: додає або оновлює оцінку книги в сесійному профілі."""
    try:
        data = json.loads(request.body)
        isbn = str(data.get("isbn", "")).strip()
        rating = data.get("rating")
        if not isbn:
            return JsonResponse({"error": "isbn required"}, status=400)
        if rating is not None:
            rating = int(rating)
            if not 1 <= rating <= 10:
                return JsonResponse({"error": "rating must be 1–10"}, status=400)
    except (json.JSONDecodeError, ValueError):
        return JsonResponse({"error": "invalid JSON"}, status=400)

    profile: dict = request.session.get(SESSION_KEY, {})
    if rating is None:
        profile.pop(isbn, None)
    else:
        profile[isbn] = rating
    request.session[SESSION_KEY] = profile
    request.session.modified = True

    return JsonResponse({
        "status": "ok",
        "profile_count": len(profile),
        "ready": len(profile) >= MIN_RATINGS_FOR_RECS,
    })


@require_http_methods(["POST"])
def api_session_clear(request):
    """AJAX: очищає весь сесійний профіль."""
    request.session[SESSION_KEY] = {}
    request.session.modified = True
    return JsonResponse({"status": "cleared"})


def explore_view(request):
    """
    Пошук книг за назвою / автором.
    При виборі конкретної книги — показуємо семантично схожі.
    """
    query = request.GET.get("q", "").strip()
    isbn = request.GET.get("isbn", "").strip()
    profile: dict = request.session.get(SESSION_KEY, {})

    search_results = []
    selected = None
    neighbours = []

    if query:
        qs = (
            Book.objects
            .filter(Q(title__icontains=query) | Q(author__icontains=query))
            .annotate(n=Count("rating"), avg=Avg("rating__book_rating"))
            .order_by("-n")[:24]
        )
        search_results = [
            {
                "isbn": b.isbn,
                "title": b.title,
                "author": b.author,
                "image_url_m": b.image_url_m,
                "bx_rating": round(b.avg, 1) if b.avg else None,
                "bx_count": b.n,
                "goodreads": _get_goodreads(b.isbn),
                "user_rating": profile.get(b.isbn),
            }
            for b in qs
        ]

    if isbn:
        book_qs = Book.objects.filter(isbn=isbn).annotate(
            n=Count("rating"), avg=Avg("rating__book_rating")
        ).first()
        if book_qs:
            selected = {
                "isbn": isbn,
                "title": book_qs.title,
                "author": book_qs.author,
                "year_of_publication": book_qs.year_of_publication,
                "image_url_m": book_qs.image_url_m,
                "bx_rating": round(book_qs.avg, 1) if book_qs.avg else None,
                "bx_count": book_qs.n,
                "goodreads": _get_goodreads(isbn),
                "user_rating": profile.get(isbn),
            }
            try:
                books_df = load_books_df()
                raw_neighbours = get_book_semantic_neighbours(isbn, books_df, top_n=8)
                nb_isbns = [r["isbn"] for r in raw_neighbours]
                score_map = {r["isbn"]: r["score"] for r in raw_neighbours}
                nb_books = {
                    b.isbn: b
                    for b in Book.objects.filter(isbn__in=nb_isbns)
                    .annotate(n=Count("rating"), avg=Avg("rating__book_rating"))
                }
                neighbours = [
                    {
                        "isbn": nb_isbn,
                        "title": nb_books[nb_isbn].title,
                        "author": nb_books[nb_isbn].author,
                        "image_url_m": nb_books[nb_isbn].image_url_m,
                        "bx_rating": round(nb_books[nb_isbn].avg, 1) if nb_books[nb_isbn].avg else None,
                        "goodreads": _get_goodreads(nb_isbn),
                        "score": score_map[nb_isbn],
                        "user_rating": profile.get(nb_isbn),
                    }
                    for nb_isbn in nb_isbns if nb_isbn in nb_books
                ]
            except Exception as e:
                logger.warning("Could not compute neighbours: %s", e)

    return render(request, "explore.html", {
        "query": query,
        "selected_isbn": isbn,
        "search_results": search_results,
        "selected": selected,
        "neighbours": neighbours,
        "profile_count": len(profile),
        "profile": profile,
    })


def my_recommendations_view(request):
    """Персональні рекомендації для сесійного користувача."""
    profile: dict = request.session.get(SESSION_KEY, {})

    if len(profile) < MIN_RATINGS_FOR_RECS:
        return render(request, "my_recommendations.html", {
            "recommendations": [],
            "profile_count": len(profile),
            "min_ratings": MIN_RATINGS_FOR_RECS,
            "needs_more": True,
        })

    try:
        books_df = load_books_df()
        # Спробуємо повний NCF + LSA гібрид з інференсом ембедінгу
        ranked = get_session_recommendations_full_hybrid(profile, books_df, top_n=10)
        isbn_list = [r["isbn"] for r in ranked]
        score_map = {r["isbn"]: r for r in ranked}

        enriched = _enrich_books_with_stats(isbn_list)
        for item in enriched:
            item.update(score_map.get(item["isbn"], {}))
            item["user_rating"] = profile.get(item["isbn"])

        liked_count = sum(1 for r in profile.values() if r >= 7)
        method = ranked[0].get("method", "unknown") if ranked else "unknown"
    except Exception as e:
        logger.exception("Error generating session recommendations: %s", e)
        enriched = []
        liked_count = 0
        method = "error"

    return render(request, "my_recommendations.html", {
        "recommendations": enriched,
        "profile_count": len(profile),
        "min_ratings": MIN_RATINGS_FOR_RECS,
        "liked_count": liked_count,
        "needs_more": False,
        "method": method,
        "ncf_available": "NCF" in method,
    })
