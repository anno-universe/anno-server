from django.contrib import admin

from .models import (
    Annotation2D,
    Box2D,
    Image2D,
    Keypoint2D,
    Polygon2D,
    Operation,
)


@admin.register(Image2D)
class Image2DAdmin(admin.ModelAdmin):
    list_display = ["id", "file_name", "project", "width", "height", "created_at"]
    list_filter = ["project", "created_at"]
    search_fields = ["file_name"]
    readonly_fields = ["created_at", "updated_at"]


@admin.register(Annotation2D)
class Annotation2DAdmin(admin.ModelAdmin):
    list_display = ["id", "image", "project", "annotation_type", "label", "is_active", "created_at"]
    list_filter = ["annotation_type", "is_active", "project", "created_at"]
    readonly_fields = ["created_at", "updated_at"]


@admin.register(Polygon2D)
class Polygon2DAdmin(admin.ModelAdmin):
    list_display = ["annotation_id", "annotation"]


@admin.register(Box2D)
class Box2DAdmin(admin.ModelAdmin):
    list_display = ["annotation_id", "annotation", "x", "y", "width", "height", "rotation"]


@admin.register(Keypoint2D)
class Keypoint2DAdmin(admin.ModelAdmin):
    list_display = ["annotation_id", "annotation"]


@admin.register(Operation)
class OperationAdmin(admin.ModelAdmin):
    list_display = ["id", "image", "action", "from_annotation", "to_annotation", "performed_by", "created_at"]
    list_filter = ["action", "created_at"]
    readonly_fields = ["created_at"]
