"""Admin support for :class:`anno.models.SoftDeleteModel`.

Mixing ``SoftDeleteAdminMixin`` into a ``ModelAdmin`` makes soft-deleted rows
visible in the changelist (via ``all_objects``), surfaces the ``deleted_at``
timestamp read-only on the change form (the field is ``editable=False`` so it is
otherwise dropped from the form), adds a "deleted / not deleted" list filter, and
adds explicit ``restore`` and permanent ``hard delete`` actions. The default
"Delete selected" action still works but soft-deletes (it calls
``QuerySet.delete()``).
"""

from django.contrib import admin
from django.utils.translation import gettext_lazy as _


class DeletedListFilter(admin.SimpleListFilter):
    """Filter the changelist by soft-delete state."""

    title = _("deleted")
    parameter_name = "deleted"

    def lookups(self, request, model_admin):
        return (("1", _("Deleted")), ("0", _("Not deleted")))

    def queryset(self, request, queryset):
        if self.value() == "1":
            return queryset.filter(deleted_at__isnull=False)
        if self.value() == "0":
            return queryset.filter(deleted_at__isnull=True)
        return queryset


class SoftDeleteAdminMixin:
    def get_queryset(self, request):
        # Show soft-deleted rows too, so admins can see / restore them.
        return self.model.all_objects.all()

    def get_list_filter(self, request):
        filters = list(super().get_list_filter(request))
        if DeletedListFilter not in filters:
            filters = [DeletedListFilter, *filters]
        return filters

    def get_readonly_fields(self, request, obj=None):
        # deleted_at is editable=False, so it is excluded from the form unless
        # listed here — surface it read-only so admins can see when a row was
        # soft-deleted.
        readonly = list(super().get_readonly_fields(request, obj))
        if "deleted_at" not in readonly:
            readonly.append("deleted_at")
        return readonly

    @admin.action(description="Restore selected (undo soft delete)")
    def restore_selected(self, request, queryset):
        queryset.restore()

    @admin.action(description="Hard delete selected (permanent, irreversible)")
    def hard_delete_selected(self, request, queryset):
        queryset.hard_delete()

    actions = ["restore_selected", "hard_delete_selected"]
