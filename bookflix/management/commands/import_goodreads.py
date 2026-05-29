import csv
import re

from django.core.management.base import BaseCommand

from bookflix.models import Book


def _clean_isbn(raw: str) -> str:
    """Normalize ISBN: strip spaces, hyphens, float/scientific notation."""
    val = raw.strip()
    # handle scientific notation e.g. "9.78043902348e+12" or float "374528373.0"
    if "e" in val.lower() or (val.replace(".", "", 1).replace("-", "").isdigit() and "." in val):
        try:
            val = str(int(float(val)))
        except (ValueError, OverflowError):
            pass
    # remove hyphens
    val = val.replace("-", "")
    return val


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

        isbn_to_rating: dict[str, float] = {}

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
                        # store both with and without leading zeros
                        isbn_to_rating[val] = rating
                        isbn_to_rating[val.lstrip("0")] = rating

        self.stdout.write(f"Loaded {len(isbn_to_rating)} ISBN entries from CSV")

        books = Book.objects.filter(goodreads_rating__isnull=True)
        to_update = []

        for book in books.iterator(chunk_size=500):
            cleaned = _clean_isbn(book.isbn)
            match = (
                isbn_to_rating.get(cleaned)
                or isbn_to_rating.get(cleaned.lstrip("0"))
            )
            if match:
                book.goodreads_rating = match
                to_update.append(book)
                updated += 1
            else:
                skipped += 1

        Book.objects.bulk_update(to_update, ["goodreads_rating"], batch_size=500)
        self.stdout.write(
            self.style.SUCCESS(f"Done: {updated} updated, {skipped} not matched")
        )
