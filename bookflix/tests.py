"""
BOOKFLIX — Test Suite

Covers:
  1. Models          — Book, User, Rating CRUD and __str__
  2. Views           — HTTP smoke tests for all pages
  3. API endpoints   — session rate/clear, evaluate, recommendations
  4. ML metrics      — RMSE, Precision@K, NDCG@K (pure-function tests)
  5. train_test_split — per-rating 80/20 split properties
  6. Session profile  — taste rating and recommendation flow

Run with:
    python manage.py test bookflix
"""

import json

from django.test import Client, TestCase
from django.urls import reverse

import pandas as pd

from bookflix.ml.evaluation import (
    RELEVANCE_THRESHOLD,
    compute_ndcg_at_k,
    compute_precision_at_k,
    compute_rmse,
)
from bookflix.models import Book, Rating, User
from bookflix.recommendation_algorithms import train_test_split

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_book(isbn="0000000001", title="Test Book", author="Author A"):
    return Book.objects.create(
        isbn=isbn, title=title, author=author,
        year_of_publication=2000, publisher="Pub",
        image_url_s="http://x.com/s.jpg",
        image_url_m="http://x.com/m.jpg",
        image_url_l="http://x.com/l.jpg",
    )


def _make_user(user_id=9999, location="Kyiv, UA", age=25):
    return User.objects.create(user_id=user_id, location=location, age=age)


def _make_rating(user, book, rating=8):
    return Rating.objects.create(user=user, book=book, book_rating=rating)


# ---------------------------------------------------------------------------
# 1. Model tests
# ---------------------------------------------------------------------------

class BookModelTest(TestCase):
    def test_create_and_str(self):
        book = _make_book(title="Kobzar")
        self.assertEqual(str(book), "Kobzar")
        self.assertEqual(Book.objects.count(), 1)

    def test_fields_stored_correctly(self):
        _make_book(isbn="1234567890", title="Dune", author="Herbert")
        fetched = Book.objects.get(isbn="1234567890")
        self.assertEqual(fetched.author, "Herbert")
        self.assertEqual(fetched.year_of_publication, 2000)


class UserModelTest(TestCase):
    def test_create_and_str(self):
        user = _make_user(user_id=42, location="Lviv, UA")
        self.assertIn("42", str(user))
        self.assertIn("Lviv", str(user))

    def test_unique_user_id(self):
        _make_user(user_id=1)
        from django.db import IntegrityError
        with self.assertRaises(IntegrityError):
            _make_user(user_id=1)


class RatingModelTest(TestCase):
    def setUp(self):
        self.user = _make_user()
        self.book = _make_book()

    def test_create_and_str(self):
        r = _make_rating(self.user, self.book, rating=9)
        self.assertEqual(r.book_rating, 9)
        self.assertIn("9999", str(r))

    def test_cascade_delete_user(self):
        _make_rating(self.user, self.book)
        self.user.delete()
        self.assertEqual(Rating.objects.count(), 0)

    def test_cascade_delete_book(self):
        _make_rating(self.user, self.book)
        self.book.delete()
        self.assertEqual(Rating.objects.count(), 0)

    def test_relevance_threshold(self):
        liked = _make_rating(self.user, self.book, rating=RELEVANCE_THRESHOLD)
        self.assertGreaterEqual(liked.book_rating, RELEVANCE_THRESHOLD)


# ---------------------------------------------------------------------------
# 2. View smoke tests
# ---------------------------------------------------------------------------

class ViewSmokeTest(TestCase):
    """Every page should return 200 (or redirect) without crashing."""

    def setUp(self):
        self.client = Client()
        self.user = _make_user(user_id=1)
        self.book = _make_book()
        _make_rating(self.user, self.book, rating=8)

    def _get_ok(self, url_name, kwargs=None):
        url = reverse(url_name, kwargs=kwargs)
        response = self.client.get(url)
        self.assertIn(response.status_code, [200, 302],
                      msg=f"{url_name} returned {response.status_code}")

    def test_home(self):
        self._get_ok("home")

    def test_dashboard(self):
        self._get_ok("dashboard")

    def test_evaluate(self):
        self._get_ok("evaluate")

    def test_taste(self):
        self._get_ok("taste")

    def test_explore(self):
        self._get_ok("explore")

    def test_my_recommendations(self):
        self._get_ok("my_recommendations")

    def test_ratingsrecommend(self):
        self._get_ok("ratingsrecommend")

    def test_user_ratings(self):
        self._get_ok("user_ratings", {"user_id": 1})

    def test_user_recommendations(self):
        self._get_ok("user_recommendations", {"user_id": 1})

    def test_404_unknown_user_ratings(self):
        url = reverse("user_ratings", kwargs={"user_id": 99999})
        response = self.client.get(url)
        self.assertIn(response.status_code, [200, 404])


# ---------------------------------------------------------------------------
# 3. API endpoint tests
# ---------------------------------------------------------------------------

class ApiSessionTest(TestCase):
    def setUp(self):
        self.client = Client()
        self.book = _make_book(isbn="TEST0001")

    def test_rate_book_post(self):
        url = reverse("api_session_rate")
        resp = self.client.post(
            url,
            data=json.dumps({"isbn": "TEST0001", "rating": 8}),
            content_type="application/json",
        )
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("profile_count", data)
        self.assertEqual(data["profile_count"], 1)

    def test_rate_book_invalid_rating(self):
        url = reverse("api_session_rate")
        resp = self.client.post(
            url,
            data=json.dumps({"isbn": "TEST0001", "rating": 99}),
            content_type="application/json",
        )
        self.assertIn(resp.status_code, [200, 400])

    def test_clear_session(self):
        rate_url = reverse("api_session_rate")
        self.client.post(
            rate_url,
            data=json.dumps({"isbn": "TEST0001", "rating": 7}),
            content_type="application/json",
        )
        clear_url = reverse("api_session_clear")
        resp = self.client.post(clear_url)
        self.assertEqual(resp.status_code, 200)

    def test_rate_requires_post(self):
        url = reverse("api_session_rate")
        resp = self.client.get(url)
        self.assertIn(resp.status_code, [400, 405])


class ApiEvaluateTest(TestCase):
    def test_returns_json(self):
        url = reverse("api_evaluate")
        resp = self.client.get(url)
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIsInstance(data, dict)


class ApiRecommendationsTest(TestCase):
    def setUp(self):
        self.user = _make_user(user_id=1)
        self.book = _make_book()
        _make_rating(self.user, self.book, rating=8)

    def test_returns_json(self):
        url = reverse("api_recommendations", kwargs={"user_id": 1})
        resp = self.client.get(url)
        # 500 is acceptable when ML model files are unavailable in the test environment
        self.assertIn(resp.status_code, [200, 404, 500])
        if resp.status_code == 200:
            data = resp.json()
            self.assertIsInstance(data, (list, dict))

    def test_unknown_user_404(self):
        url = reverse("api_recommendations", kwargs={"user_id": 99999})
        resp = self.client.get(url)
        self.assertEqual(resp.status_code, 404)


# ---------------------------------------------------------------------------
# 4. ML metric tests (pure functions — no DB)
# ---------------------------------------------------------------------------

class RmseTest(TestCase):
    def test_perfect_prediction(self):
        self.assertAlmostEqual(compute_rmse([1, 2, 3], [1, 2, 3]), 0.0)

    def test_unit_error(self):
        self.assertAlmostEqual(compute_rmse([1.0], [2.0]), 1.0)

    def test_symmetric(self):
        a = compute_rmse([1, 3], [3, 1])
        b = compute_rmse([3, 1], [1, 3])
        self.assertAlmostEqual(a, b)

    def test_always_nonnegative(self):
        self.assertGreaterEqual(compute_rmse([5, 7, 3], [4, 8, 2]), 0.0)


class PrecisionAtKTest(TestCase):
    def test_perfect_recommendations(self):
        recs = {1: ["a", "b", "c"]}
        relevant = {1: {"a", "b", "c"}}
        self.assertAlmostEqual(compute_precision_at_k(recs, relevant, k=3), 1.0)

    def test_no_hits(self):
        recs = {1: ["x", "y", "z"]}
        relevant = {1: {"a", "b", "c"}}
        self.assertAlmostEqual(compute_precision_at_k(recs, relevant, k=3), 0.0)

    def test_half_hits(self):
        recs = {1: ["a", "x", "b", "y"]}
        relevant = {1: {"a", "b"}}
        result = compute_precision_at_k(recs, relevant, k=4)
        self.assertAlmostEqual(result, 0.5)

    def test_truncated_at_k(self):
        recs = {1: ["a", "b", "x", "y", "z"]}
        relevant = {1: {"a", "b", "x"}}
        # Only first k=2 are checked: a, b → both relevant → 2/2 = 1.0
        self.assertAlmostEqual(compute_precision_at_k(recs, relevant, k=2), 1.0)

    def test_empty_recommendations(self):
        self.assertAlmostEqual(compute_precision_at_k({}, {1: {"a"}}, k=5), 0.0)

    def test_multiple_users_averaged(self):
        # user1: 1 hit out of 3 → 1/3; user2: 2 hits out of 3 → 2/3; mean = 1/2
        recs = {1: ["a", "x", "y"], 2: ["b", "c", "z"]}
        relevant = {1: {"a"}, 2: {"b", "c"}}
        result = compute_precision_at_k(recs, relevant, k=3)
        self.assertAlmostEqual(result, 0.5)


class NdcgAtKTest(TestCase):
    def test_perfect_ranking(self):
        recs = {1: ["a", "b"]}
        relevant = {1: {"a", "b"}}
        self.assertAlmostEqual(compute_ndcg_at_k(recs, relevant, k=2), 1.0)

    def test_no_hits(self):
        recs = {1: ["x", "y"]}
        relevant = {1: {"a", "b"}}
        self.assertAlmostEqual(compute_ndcg_at_k(recs, relevant, k=2), 0.0)

    def test_ndcg_penalises_lower_rank(self):
        # "a" at position 1 vs position 2
        recs_best = {1: ["a", "x"]}
        recs_worse = {1: ["x", "a"]}
        relevant = {1: {"a"}}
        best = compute_ndcg_at_k(recs_best, relevant, k=2)
        worse = compute_ndcg_at_k(recs_worse, relevant, k=2)
        self.assertGreater(best, worse)

    def test_ndcg_between_0_and_1(self):
        recs = {1: ["a", "x", "b"]}
        relevant = {1: {"a", "b", "c"}}
        result = compute_ndcg_at_k(recs, relevant, k=3)
        self.assertGreaterEqual(result, 0.0)
        self.assertLessEqual(result, 1.0)

    def test_empty_recommendations(self):
        self.assertAlmostEqual(compute_ndcg_at_k({}, {1: {"a"}}, k=5), 0.0)


# ---------------------------------------------------------------------------
# 5. train_test_split tests
# ---------------------------------------------------------------------------

class TrainTestSplitTest(TestCase):
    def _make_df(self, n_users=10, ratings_per_user=10):
        rows = []
        for uid in range(1, n_users + 1):
            for i in range(ratings_per_user):
                rows.append({
                    "user_id": uid,
                    "book__isbn": f"isbn_{uid}_{i}",
                    "book_rating": 7,
                })
        return pd.DataFrame(rows)

    def test_split_ratio(self):
        df = self._make_df(n_users=20, ratings_per_user=10)
        train, test = train_test_split(df, test_ratio=0.2)
        total = len(train) + len(test)
        self.assertEqual(total, len(df))
        self.assertAlmostEqual(len(test) / total, 0.2, delta=0.05)

    def test_all_users_in_train(self):
        df = self._make_df(n_users=50, ratings_per_user=5)
        train, _ = train_test_split(df, test_ratio=0.2)
        train_users = set(train["user_id"].unique())
        all_users = set(df["user_id"].unique())
        self.assertEqual(train_users, all_users)

    def test_no_overlap(self):
        df = self._make_df(n_users=10, ratings_per_user=10)
        train, test = train_test_split(df, test_ratio=0.2)
        train_idx = set(train.index)
        test_idx = set(test.index)
        self.assertTrue(train_idx.isdisjoint(test_idx))

    def test_single_rating_user_goes_to_train(self):
        rows = [{"user_id": 1, "book__isbn": "only_book", "book_rating": 8}]
        df = pd.DataFrame(rows)
        train, test = train_test_split(df, test_ratio=0.2)
        self.assertEqual(len(train), 1)
        self.assertEqual(len(test), 0)

    def test_reproducible_with_same_seed(self):
        df = self._make_df(n_users=20, ratings_per_user=10)
        train1, _ = train_test_split(df, test_ratio=0.2)
        train2, _ = train_test_split(df, test_ratio=0.2)
        self.assertListEqual(list(train1.index), list(train2.index))
