from django.apps import AppConfig


class AnnoUsersConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "anno_users"

    def ready(self):
        import anno_users.signals  # noqa: F401
