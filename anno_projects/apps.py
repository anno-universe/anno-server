from django.apps import AppConfig


class AnnoProjectsConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "anno_projects"

    def ready(self):
        import anno_projects.signals  # noqa: F401
