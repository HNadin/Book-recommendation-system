import numpy as np
import pandas as pd
from django.test import TestCase

from bookflix.models import User, Book, Rating
from bookflix.recommendation_algorithms import (
    load_data_ratings,
    compute_average_ratings,
    create_user_profile,
    load_or_compute_svd,
    evaluate_user_model,
)


class TestRecommendationAlgorithms(TestCase):
    """Тестування алгоритмів рекомендацій"""

    def setUp(self):
        self.user1 = User.objects.create(user_id=1, location="Kyiv, Ukraine", age=25)
        self.user2 = User.objects.create(user_id=2, location="Lviv, Ukraine", age=30)

        self.book1 = Book.objects.create(
            isbn='123456789',
            title='Test Book 1',
            author='Author 1',
            year_of_publication=2000,
            publisher='Publisher 1',
            image_url_s='http://example.com/s1.jpg',
            image_url_m='http://example.com/m1.jpg',
            image_url_l='http://example.com/l1.jpg',
        )
        self.book2 = Book.objects.create(
            isbn='987654321',
            title='Test Book 2',
            author='Author 2',
            year_of_publication=2010,
            publisher='Publisher 2',
            image_url_s='http://example.com/s2.jpg',
            image_url_m='http://example.com/m2.jpg',
            image_url_l='http://example.com/l2.jpg',
        )

        # 3 ratings total: book1 avg = (8+9)/2 = 8.5, user1 has one positive (>=6) rating
        Rating.objects.create(user=self.user1, book=self.book1, book_rating=8)
        Rating.objects.create(user=self.user2, book=self.book1, book_rating=9)
        Rating.objects.create(user=self.user1, book=self.book2, book_rating=3)

    def test_load_data_ratings(self):
        """Тест завантаження даних рейтингів"""
        ratings_df = load_data_ratings()
        expected_columns = ['user_id', 'book__isbn', 'book_rating']
        self.assertTrue(all(col in ratings_df.columns for col in expected_columns))
        self.assertEqual(len(ratings_df), 3)
        self.assertTrue(ratings_df['book_rating'].dtype in [np.int64, int])

    def test_compute_average_ratings(self):
        """Тест обчислення середніх рейтингів"""
        books_with_avg = compute_average_ratings()
        self.assertIn('avg_rating', books_with_avg.columns)
        book1_avg = books_with_avg[books_with_avg['isbn'] == '123456789']['avg_rating'].iloc[0]
        expected_avg = (8 + 9) / 2
        self.assertEqual(book1_avg, expected_avg)

    def test_create_user_profile_existing_user(self):
        """Тест створення профілю для існуючого користувача"""
        ratings_df = load_data_ratings()
        mock_tfidf_matrix = np.array([[1, 0, 1], [0, 1, 1]])
        mock_books = pd.DataFrame({
            'isbn': ['123456789', '987654321'],
            'title': ['Test Book 1', 'Test Book 2'],
        })
        user_profile = create_user_profile(1, ratings_df, mock_tfidf_matrix, mock_books)
        self.assertIsNotNone(user_profile)
        self.assertIsInstance(user_profile, np.ndarray)

    def test_create_user_profile_nonexistent_user(self):
        """Тест створення профілю для неіснуючого користувача"""
        ratings_df = load_data_ratings()
        mock_tfidf_matrix = np.array([[1, 0, 1], [0, 1, 1]])
        mock_books = pd.DataFrame({
            'isbn': ['123456789', '987654321'],
            'title': ['Test Book 1', 'Test Book 2'],
        })
        user_profile = create_user_profile(999, ratings_df, mock_tfidf_matrix, mock_books)
        self.assertIsNone(user_profile)


class TestBookFlixModels(TestCase):
    """Тестування моделей Django"""

    def test_user_model_creation(self):
        """Тест створення моделі User"""
        user = User.objects.create(
            user_id=100,
            location="Kharkiv, Ukraine",
            age=28,
        )
        self.assertEqual(user.user_id, 100)
        self.assertEqual(user.location, "Kharkiv, Ukraine")
        self.assertEqual(user.age, 28)
        self.assertEqual(str(user), "User 100 from Kharkiv, Ukraine")

    def test_book_model_creation(self):
        """Тест створення моделі Book"""
        book = Book.objects.create(
            isbn='111222333',
            title='Django for Beginners',
            author='William Vincent',
            year_of_publication=2019,
            publisher='WelcomeToCode',
            image_url_s='http://example.com/s.jpg',
            image_url_m='http://example.com/m.jpg',
            image_url_l='http://example.com/l.jpg',
        )
        self.assertEqual(book.isbn, '111222333')
        self.assertEqual(book.title, 'Django for Beginners')
        self.assertEqual(book.author, 'William Vincent')
        self.assertEqual(str(book), 'Django for Beginners')

    def test_rating_model_creation(self):
        """Тест створення моделі Rating"""
        user = User.objects.create(user_id=200, location="Odesa, Ukraine", age=35)
        book = Book.objects.create(
            isbn='444555666',
            title='Python Cookbook',
            author='David Beazley',
            year_of_publication=2013,
            publisher='OReilly',
            image_url_s='http://example.com/s.jpg',
            image_url_m='http://example.com/m.jpg',
            image_url_l='http://example.com/l.jpg',
        )
        rating = Rating.objects.create(user=user, book=book, book_rating=9)
        self.assertEqual(rating.user, user)
        self.assertEqual(rating.book, book)
        self.assertEqual(rating.book_rating, 9)
        self.assertEqual(str(rating), 'User 200 rated 444555666 with 9')


class TestRecommendationAccuracy(TestCase):
    """Тестування точності алгоритмів рекомендацій"""

    def setUp(self):
        """Реалістичний тестовий датасет: 10 користувачів та 20 книг з випадковими рейтингами"""
        np.random.seed(42)

        self.users = []
        for i in range(1, 11):
            user = User.objects.create(user_id=i, location=f"City {i}, Ukraine", age=20 + i)
            self.users.append(user)

        self.books = []
        for i in range(1, 21):
            book = Book.objects.create(
                isbn=f'ISBN{i:010d}',
                title=f'Book Title {i}',
                author=f'Author {i}',
                year_of_publication=2000 + i,
                publisher=f'Publisher {i}',
                image_url_s=f'http://example.com/s{i}.jpg',
                image_url_m=f'http://example.com/m{i}.jpg',
                image_url_l=f'http://example.com/l{i}.jpg',
            )
            self.books.append(book)

        for user in self.users:
            for book in self.books:
                if np.random.random() > 0.5:
                    Rating.objects.create(
                        user=user,
                        book=book,
                        book_rating=int(np.random.randint(1, 11)),
                    )

        # Ensure user 1 always has at least one rating for evaluate_user_model
        if not Rating.objects.filter(user=self.users[0]).exists():
            Rating.objects.create(user=self.users[0], book=self.books[0], book_rating=8)

    def test_mse_calculation(self):
        """Тест розрахунку MSE"""
        ratings_df = load_data_ratings()
        svd = load_or_compute_svd(ratings_df)
        mse, error = evaluate_user_model(1, ratings_df, svd)
        self.assertIsNone(error)
        self.assertIsNotNone(mse)
        self.assertIsInstance(mse, (int, float))
        self.assertGreaterEqual(mse, 0)
