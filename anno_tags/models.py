from django.conf import settings
from django.db import models

from anno.models import SoftDeleteModel


class ProjectTag(SoftDeleteModel):
    """A per-project tag definition.

    Supervisors define tags that workers and supervisors can apply to
    images to track annotation progress (e.g. ``worker_finish``,
    ``verify_finish``, ``worker_question``, ``supervisor_question``).
    """

    # Who may apply this tag. Values mirror ``ProjectMembership.ROLE_CHOICES``
    # ("worker"/"supervisor") so the apply check can compare a member's role
    # directly against ``permission_level``. "common" means any project member.
    PERMISSION_COMMON = "common"
    PERMISSION_WORKER = "worker"
    PERMISSION_SUPERVISOR = "supervisor"
    PERMISSION_CHOICES = [
        (PERMISSION_COMMON, "Common"),
        (PERMISSION_WORKER, "Worker"),
        (PERMISSION_SUPERVISOR, "Supervisor"),
    ]

    project = models.ForeignKey(
        "anno_projects.Project",
        on_delete=models.CASCADE,
        related_name="tags",
    )
    name = models.CharField(
        max_length=64,
        help_text="Unique slug per project, e.g. 'worker_finish'.",
    )
    display_name = models.CharField(
        max_length=128,
        help_text="Human-readable label, e.g. 'Worker Finish'.",
    )
    color = models.CharField(
        max_length=7,
        default="#6366F1",
        help_text="Hex color code for UI badges, e.g. '#22C55E'.",
    )
    description = models.TextField(
        blank=True,
        default="",
        help_text="Optional explanation of this tag's meaning.",
    )
    is_active = models.BooleanField(
        default=True,
        help_text="Inactive tags are hidden from the apply UI but remain on existing images.",
    )
    permission_level = models.CharField(
        max_length=20,
        choices=PERMISSION_CHOICES,
        default=PERMISSION_COMMON,
        help_text=(
            "Which project role may apply this tag: 'common' = any member; "
            "'worker'/'supervisor' = only that role."
        ),
    )
    created_by = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.PROTECT,
        related_name="created_tags",
    )
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta(SoftDeleteModel.Meta):
        db_table = "anno_project_tag"
        ordering = ["name"]
        verbose_name = "project tag"
        verbose_name_plural = "project tags"
        constraints = [
            # Only alive tags are unique per project, so a soft-deleted name
            # can be reused.
            models.UniqueConstraint(
                fields=["project", "name"],
                condition=models.Q(deleted_at__isnull=True),
                name="uniq_active_project_tag_name",
            ),
        ]

    def __str__(self):
        return f"{self.name} ({self.project.name})"


class ImageTag(SoftDeleteModel):
    """Associates a tag with a specific image.

    Each image can have at most one instance of each tag.  Tags track
    annotation workflow progress.
    """

    image = models.ForeignKey(
        "anno_images.Image2D",
        on_delete=models.CASCADE,
        related_name="tags",
    )
    tag = models.ForeignKey(
        ProjectTag,
        on_delete=models.CASCADE,
        related_name="image_tags",
    )
    applied_by = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.PROTECT,
        related_name="applied_image_tags",
    )
    note = models.TextField(
        blank=True,
        default="",
        help_text="Optional comment, e.g. explanation for a 'worker question' tag.",
    )
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta(SoftDeleteModel.Meta):
        db_table = "anno_image_tag"
        ordering = ["-created_at"]
        verbose_name = "image tag"
        verbose_name_plural = "image tags"
        constraints = [
            # Only alive rows are unique, so re-applying a soft-deleted tag works.
            models.UniqueConstraint(
                fields=["image", "tag"],
                condition=models.Q(deleted_at__isnull=True),
                name="uniq_active_image_tag",
            ),
        ]
        indexes = [
            models.Index(fields=["image"]),
            models.Index(fields=["tag"]),
        ]

    def __str__(self):
        return f"{self.tag.name} on Image #{self.image_id}"
