from django.contrib import admin

from .models import (
    InferenceResult,
    InferenceRun,
    InferenceServiceProvider,
    InferenceTask,
    InteractiveInferenceOperation,
    InteractiveInferenceServiceProvider,
    InteractiveInferenceSession,
)


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


class InferenceTaskInline(admin.TabularInline):
    model = InferenceTask
    extra = 0
    can_delete = False
    readonly_fields = ("image", "status", "annotations_created", "attempts", "error")
    fields = readonly_fields


@admin.register(InferenceRun)
class InferenceRunAdmin(admin.ModelAdmin):
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
    inlines = [InferenceTaskInline]


class InferenceResultInline(admin.TabularInline):
    model = InferenceResult
    extra = 0
    can_delete = False
    readonly_fields = ("result_index", "result_type", "label", "score", "status", "annotation")
    fields = readonly_fields


@admin.register(InferenceTask)
class InferenceTaskAdmin(admin.ModelAdmin):
    list_display = ("id", "run", "image", "status", "annotations_created", "attempts")
    list_filter = ("status",)
    raw_id_fields = ("run", "image")
    inlines = [InferenceResultInline]


@admin.register(InferenceResult)
class InferenceResultAdmin(admin.ModelAdmin):
    list_display = ("id", "task", "result_index", "result_type", "status", "annotation")
    list_filter = ("status", "result_type")
    raw_id_fields = ("task", "annotation")


@admin.register(InteractiveInferenceServiceProvider)
class InteractiveInferenceServiceProviderAdmin(admin.ModelAdmin):
    # auth_secret is deliberately excluded from list_display.
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


class InteractiveInferenceOperationInline(admin.TabularInline):
    model = InteractiveInferenceOperation
    extra = 0
    can_delete = False
    readonly_fields = ("step_index", "result_type", "result", "error", "created_at")
    fields = readonly_fields


@admin.register(InteractiveInferenceSession)
class InteractiveInferenceSessionAdmin(admin.ModelAdmin):
    list_display = (
        "id",
        "project",
        "image",
        "provider",
        "status",
        "final_annotation",
        "created_at",
    )
    list_filter = ("status", "project")
    raw_id_fields = ("project", "image", "provider", "performed_by", "from_annotation", "final_annotation")
    readonly_fields = ("created_at", "committed_at", "discarded_at")
    inlines = [InteractiveInferenceOperationInline]
