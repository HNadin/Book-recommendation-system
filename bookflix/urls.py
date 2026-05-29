from django.urls import path
from . import views

urlpatterns = [
    # --- Головні сторінки ---
    path("", views.home, name="home"),
    path("dashboard/", views.dashboard, name="dashboard"),
    path("evaluate/", views.evaluate_view, name="evaluate"),

    # --- Варіант Б: сесійний профіль ---
    path("taste/", views.taste_view, name="taste"),
    path("explore/", views.explore_view, name="explore"),
    path("my-recommendations/", views.my_recommendations_view, name="my_recommendations"),

    # --- Legacy: пошук за ID ---
    path("ratingsrecommend/", views.homepage_view, name="ratingsrecommend"),
    path("ratings/<int:user_id>/", views.user_ratings_view, name="user_ratings"),
    path("recommendations/<int:user_id>/", views.user_recommendations_view, name="user_recommendations"),

    # --- REST API ---
    path("api/recommendations/<int:user_id>/", views.api_recommendations, name="api_recommendations"),
    path("api/evaluate/", views.api_evaluate, name="api_evaluate"),
    path("api/session/rate/", views.api_session_rate, name="api_session_rate"),
    path("api/session/clear/", views.api_session_clear, name="api_session_clear"),

    # Legacy alias
    path(
        "api/fetch_hybrid_recommendations/<int:user_id>/",
        views.fetch_hybrid_recommendations,
        name="fetch_hybrid_recommendations",
    ),
]
