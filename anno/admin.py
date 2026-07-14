"""Admin support for :class:`anno.models.SoftDeleteModel`.

Mixing ``SoftDeleteAdminMixin`` into a ``ModelAdmin`` makes soft-deleted rows
visible in the changelist (via ``all_objects``) and adds explicit ``restore``
and permanent ``hard delete`` actions. The default "Delete selected" action
still works but soft-deletes (it calls ``QuerySet.delete()``).
"""

from django.contrib import admin


class SoftDeleteAdminMixin:
    def get_queryset(self, request):
        # Show soft-deleted rows too, so admins can see / restore them.
        return self.model.all_objects.all()

    @admin.action(description="Restore selected (undo soft delete)")
    def restore_selected(self, request, queryset):
        queryset.restore()

    @admin.action(description="Hard delete selected (permanent, irreversible)")
    def hard_delete_selected(self, request, queryset):
        queryset.hard_delete()

    actions = ["restore_selected", "hard_delete_selected"]
