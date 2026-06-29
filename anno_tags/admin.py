from django.contrib import admin

from .models import ProjectTag, ImageTag


@admin.register(ProjectTag)
class ProjectTagAdmin(admin.ModelAdmin):
    list_display = [
        "id",
        "name",
        "display_name",
        "project",
        "color",
        "is_active",
        "created_by",
        "created_at",
    ]
    list_filter = ["project", "is_active", "created_at"]
    search_fields = ["name", "display_name", "project__name"]
    readonly_fields = ["created_at", "updated_at"]


@admin.register(ImageTag)
class ImageTagAdmin(admin.ModelAdmin):
    list_display = ["id", "image", "tag", "applied_by", "created_at"]
    list_filter = ["tag", "created_at"]
    search_fields = ["image__file_name", "tag__name"]
    readonly_fields = ["created_at"]
