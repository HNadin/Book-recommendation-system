import csv
import re

from django.core.management.base import BaseCommand

from bookflix.models import Book


def _clean_isbn(raw: str) -> str:
    """Normalize ISBN: strip spaces, hyphens, float/scientific notation."""
    val = raw.strip()
    if "e" in val.lower() or ("." in val and val.replace(".", "", 1).replace("-", "").isdigit()):
        try:
            val = str(int(float(val)))
        except (ValueError, OverflowError):
            pass
    val = val.replace("-", "")
    return val


def _normalize_title(title: str) -> str:
    """Lowercase, strip series suffixes like '(HP #1)', remove articles, punctuation."""
    t = title.lower()
    t = re.sub(r"\s*\([^)]*\)", "", t)
    t = re.sub(r"^(the|a|an)\s+", "", t)
    t = re.sub(r"[^a-z0-9\s]", "", t)
    return " ".join(t.split())


class Command(BaseCommand):
    help = "Import Goodreads average ratings from goodbooks-10k books.csv"

    def add_arguments(self, parser):
        parser.add_argument(
            "csv_path",
            type=str,
            help="Path to goodbooks-10k books.csv",
        )

    def handle(self, *args, **options):
        path = options["csv_path"]
        updated = 0
        skipped = 0
        isbn_matched = 0
        title_matched = 0

        isbn_to_rating: dict[str, float] = {}
        title_to_rating: dict[str, float] = {}

        with open(path, encoding="utf-8", errors="ignore") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rating_str = row.get("average_rating", "").strip()
                if not rating_str:
                    continue
                try:
                    rating = float(rating_str)
                except ValueError:
                    continue

                for col in ("isbn", "isbn13"):
                    raw = row.get(col, "")
                    if not raw or raw.strip() in ("", "nan", "NaN"):
                        continue
                    val = _clean_isbn(raw)
                    if val:
                        isbn_to_rating[val] = rating
                        isbn_to_rating[val.lstrip("0")] = rating

                title = row.get("title", "").strip()
                if title:
                    norm = _normalize_title(title)
                    if norm:
                        title_to_rating[norm] = rating

        self.stdout.write(
            f"Loaded {len(isbn_to_rating)} ISBN entries, {len(title_to_rating)} title entries from CSV"
        )

        books = Book.objects.filter(goodreads_rating__isnull=True)
        to_update = []

        for book in books.iterator(chunk_size=500):
            cleaned = _clean_isbn(book.isbn)
            match = isbn_to_rating.get(cleaned) or isbn_to_rating.get(cleaned.lstrip("0"))
            if match:
                isbn_matched += 1
            else:
                norm_title = _normalize_title(book.title or "")
                match = title_to_rating.get(norm_title)
                if match:
                    title_matched += 1

            if match:
                book.goodreads_rating = match
                to_update.append(book)
                updated += 1
            else:
                skipped += 1

        Book.objects.bulk_update(to_update, ["goodreads_rating"], batch_size=500)
        self.stdout.write(
            self.style.SUCCESS(
                f"Done: {updated} updated ({isbn_matched} by ISBN, {title_matched} by title), "
                f"{skipped} not matched"
            )
        )
