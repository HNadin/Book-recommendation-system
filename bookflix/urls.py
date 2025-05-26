from unicodedata import name
from django.urls import path
from . import views

urlpatterns = [
    path('', views.Home, name='Home'),
    path('index/', views.index, name='index'),
    path('recommend/', views.recommend, name='recommend'),
    path('ratingsreccomend/', views.homepage_view, name='ratingsreccomend'),
    path('ratings/<int:user_id>/', views.user_ratings_view, name='user_ratings'),
    path('recommendations/<int:user_id>/', views.user_recommendations_view , name='user_recommendations'),
    path('api/fetch_hybrid_recommendations/<int:user_id>/',views.fetch_hybrid_recommendations, name='fetch_hybrid_recommendations'),
]
