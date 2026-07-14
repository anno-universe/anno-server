from django.contrib import admin, messages

from anno.admin import SoftDeleteAdminMixin

from .models import (
    InferenceResult,
    InferenceRun,
    InferenceServiceProvider,
    InferenceTask,
    InteractiveInferenceOperation,
    InteractiveInferenceServiceProvider,
    InteractiveInferenceSession,
)
from .services import complete_interactive_session


@admin.register(InferenceServiceProvider)
class InferenceServiceProviderAdmin(SoftDeleteAdminMixin, admin.ModelAdmin):
    # auth_secret is deliberately excluded from list_display; it is the
    # plaintext outbound credential and must not be surfaced casually.
    list_display = (
        "id",
        "name",
        "model_name",
        "project",
        "auth_type",
        "is_active",
        "deleted_at",
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
class InteractiveInferenceServiceProviderAdmin(SoftDeleteAdminMixin, admin.ModelAdmin):
    # auth_secret is deliberately excluded from list_display.
    list_display = (
        "id",
        "name",
        "model_name",
        "project",
        "auth_type",
        "is_active",
        "deleted_at",
        "created_at",
    )
    list_filter = ("is_active", "auth_type", "project")
    search_fields = ("name", "model_name", "inference_url", "public_url")
    raw_id_fields = ("project", "created_by")


class InteractiveInferenceOperationInline(admin.TabularInline):
    model = InteractiveInferenceOperation
    extra = 0
    can_delete = False
    readonly_fields = ("step_index", "result_type", "result", "annotation", "error", "created_at")
    fields = readonly_fields


@admin.register(InteractiveInferenceSession)
class InteractiveInferenceSessionAdmin(admin.ModelAdmin):
    list_display = (
        "id",
        "project",
        "image",
        "provider",
        "status",
        "created_at",
    )
    list_filter = ("status", "project")
    raw_id_fields = ("project", "image", "provider", "performed_by")
    readonly_fields = ("created_at", "updated_at")
    inlines = [InteractiveInferenceOperationInline]
    actions = ["force_discard_selected"]

    @admin.action(description="Force discard selected sessions")
    def force_discard_selected(self, request, queryset):
        count = 0
        for session in queryset.filter(status=InteractiveInferenceSession.STATUS_EDITING):
            session.status = InteractiveInferenceSession.STATUS_DISCARDED
            session.save(update_fields=["status", "updated_at"])
            complete_interactive_session(
                provider=session.provider, session_id=session.id
            )
            count += 1
        self.message_user(
            request,
            f"Force-discarded {count} session(s).",
            messages.SUCCESS,
        )
