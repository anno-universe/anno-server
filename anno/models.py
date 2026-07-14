"""Reusable soft-delete foundation.

``SoftDeleteModel`` is an abstract base that turns a physical ``DELETE`` into a
logical one: instead of issuing SQL ``DELETE`` (which would fire ``on_delete``
cascades and raise ``ProtectedError`` for ``PROTECT`` relations), it stamps a
``deleted_at`` timestamp and hides the row from the default manager. History is
preserved and referential integrity is untouched.

This is distinct from any ``is_active`` flag a model may carry: ``is_active`` is
an enable/disable toggle ("usable"), whereas ``deleted_at`` means "removed from
the API". The two are orthogonal and coexist.

Every concrete model that subclasses ``SoftDeleteModel`` and declares its own
``Meta`` MUST inherit this base's ``Meta`` (``class Meta(SoftDeleteModel.Meta)``)
or re-declare ``base_manager_name = "all_objects"``. Django uses ``_base_manager``
for related-object access (``run.provider``) and the deletion collector; pointing
it at the unfiltered manager keeps those internals seeing soft-deleted rows while
public ``.objects`` queries hide them.
"""

from django.db import models
from django.utils import timezone


class SoftDeleteQuerySet(models.QuerySet):
    def alive(self):
        return self.filter(deleted_at__isnull=True)

    def dead(self):
        return self.filter(deleted_at__isnull=False)

    def delete(self):
        """Bulk soft delete: a single UPDATE, no cascade / PROTECT evaluation."""
        return self.update(deleted_at=timezone.now()), {}

    def hard_delete(self):
        """Physically delete the rows (fires cascades / PROTECT / signals)."""
        return super().delete()

    def restore(self):
        return self.update(deleted_at=None)


class SoftDeleteManager(models.Manager.from_queryset(SoftDeleteQuerySet)):
    """Default manager: excludes soft-deleted rows."""

    def get_queryset(self):
        return super().get_queryset().filter(deleted_at__isnull=True)


class AllObjectsManager(models.Manager.from_queryset(SoftDeleteQuerySet)):
    """Unfiltered manager: includes soft-deleted rows. Used as _base_manager."""


class SoftDeleteModel(models.Model):
    deleted_at = models.DateTimeField(
        null=True, blank=True, default=None, db_index=True, editable=False
    )

    objects = SoftDeleteManager()
    all_objects = AllObjectsManager()

    class Meta:
        abstract = True
        base_manager_name = "all_objects"

    @property
    def is_deleted(self) -> bool:
        return self.deleted_at is not None

    def delete(self, using=None, keep_parents=False):
        """Soft delete: stamp ``deleted_at``. No SQL DELETE is issued, so
        ``on_delete`` cascades and ``PROTECT`` are never evaluated."""
        self.deleted_at = timezone.now()
        self.save(using=using, update_fields=["deleted_at"])
        return 1, {self._meta.label: 1}

    def hard_delete(self, using=None, keep_parents=False):
        return super().delete(using=using, keep_parents=keep_parents)

    def restore(self, using=None):
        self.deleted_at = None
        self.save(using=using, update_fields=["deleted_at"])
