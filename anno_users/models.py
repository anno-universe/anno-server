from django.contrib.auth.models import AbstractUser


class User(AbstractUser):
    class Meta:
        db_table = "anno_users"
        verbose_name = "user"
        verbose_name_plural = "users"

    def __str__(self):
        return self.username
