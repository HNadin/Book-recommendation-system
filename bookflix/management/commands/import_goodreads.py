import csv

from django.core.management.base import BaseCommand

from bookflix.models import Book


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
                rating = row.get("average_rating", "").strip()
                if not rating:
                    continue
                for col in ("isbn", "isbn13"):
                    val = row.get(col, "").strip().lstrip("0")
                    if val:
                        try:
                            isbn_to_rating[val] = float(rating)
                        except ValueError:
                            pass

        books = Book.objects.filter(goodreads_rating__isnull=True)
        to_update = []

        for book in books.iterator(chunk_size=500):
            key = book.isbn.lstrip("0")
            if key in isbn_to_rating:
                book.goodreads_rating = isbn_to_rating[key]
                to_update.append(book)
                updated += 1
            else:
                skipped += 1

        Book.objects.bulk_update(to_update, ["goodreads_rating"], batch_size=500)
        self.stdout.write(
            self.style.SUCCESS(f"Done: {updated} updated, {skipped} not matched")
        )
