from django.urls import path
from . import views

urlpatterns = [
    # Pages
    path("", views.home, name="home"),
    path("dashboard/", views.dashboard, name="dashboard"),
    path("ratingsrecommend/", views.homepage_view, name="ratingsrecommend"),
    path("ratings/<int:user_id>/", views.user_ratings_view, name="user_ratings"),
    path("recommendations/<int:user_id>/", views.user_recommendations_view, name="user_recommendations"),
    path("evaluate/", views.evaluate_view, name="evaluate"),

    # REST API
    path("api/recommendations/<int:user_id>/", views.api_recommendations, name="api_recommendations"),
    path("api/evaluate/", views.api_evaluate, name="api_evaluate"),

    # Legacy alias
    path("api/fetch_hybrid_recommendations/<int:user_id>/", views.fetch_hybrid_recommendations, name="fetch_hybrid_recommendations"),
]
