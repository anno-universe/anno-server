from django.apps import AppConfig


class AnnoImagesConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "anno_images"

    def ready(self):
        import anno_images.signals  # noqa: F401
