from django.db import migrations, models

class Migration(migrations.Migration):

    dependencies = [
        ("bookflix", "0001_initial"),
    ]

    operations = [
        migrations.AlterField(
            model_name="book",
            name="isbn",
            field=models.CharField(db_index=True, max_length=200),
        ),
        migrations.AlterField(
            model_name="book",
            name="title",
            field=models.CharField(db_index=True, max_length=255),
        ),
    ]
