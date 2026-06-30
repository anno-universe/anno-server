from django.contrib import admin

from .models import InferenceJob, InferenceJobItem, InferenceServiceProvider


@admin.register(InferenceServiceProvider)
class InferenceServiceProviderAdmin(admin.ModelAdmin):
    # auth_secret is deliberately excluded from list_display; it is the
    # plaintext outbound credential and must not be surfaced casually.
    list_display = (
        "id",
        "name",
        "model_name",
        "project",
        "auth_type",
        "is_active",
        "created_at",
    )
    list_filter = ("is_active", "auth_type", "project")
    search_fields = ("name", "model_name", "inference_url")
    raw_id_fields = ("project", "created_by")


class InferenceJobItemInline(admin.TabularInline):
    model = InferenceJobItem
    extra = 0
    can_delete = False
    readonly_fields = ("image", "status", "annotations_created", "attempts", "error")
    fields = readonly_fields


@admin.register(InferenceJob)
class InferenceJobAdmin(admin.ModelAdmin):
    list_display = (
        "id",
        "project",
        "provider",
        "status",
        "total_items",
        "completed_items",
        "failed_items",
        "annotations_created",
        "created_at",
    )
    list_filter = ("status", "project")
    raw_id_fields = ("project", "provider", "created_by")
    readonly_fields = ("created_at", "started_at", "finished_at")
    inlines = [InferenceJobItemInline]


@admin.register(InferenceJobItem)
class InferenceJobItemAdmin(admin.ModelAdmin):
    list_display = ("id", "job", "image", "status", "annotations_created", "attempts")
    list_filter = ("status",)
    raw_id_fields = ("job", "image")
