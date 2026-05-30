from django.db import models


class Book(models.Model):
    isbn = models.CharField(max_length=200, db_index=True)
    title = models.TextField(db_index=True)
    author = models.TextField()
    year_of_publication = models.IntegerField(null=True, blank=True)
    publisher = models.TextField()
    image_url_s = models.TextField(blank=True, default="")
    image_url_m = models.TextField(blank=True, default="")
    image_url_l = models.TextField(blank=True, default="")
    goodreads_rating = models.FloatField(null=True, blank=True)

    def __str__(self):
        return self.title


class User(models.Model):
    user_id = models.IntegerField(unique=True, primary_key=True)
    location = models.TextField()
    age = models.IntegerField(null=True, blank=True)

    def __str__(self):
        return f'User {self.user_id} from {self.location}'


class Rating(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    book = models.ForeignKey(Book, on_delete=models.CASCADE, db_index=True)
    book_rating = models.IntegerField()

    def __str__(self):
        return f'User {self.user.user_id} rated {self.book.isbn} with {self.book_rating}'
