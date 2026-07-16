import os

from django.conf import settings
from django.db import models

from anno.models import SoftDeleteModel


def _export_upload_to(instance, filename):
    return f"exports/{instance.task.project_id}/{filename}"


class ExportTask(SoftDeleteModel):
    STATUS_PENDING = "pending"
    STATUS_RUNNING = "running"
    STATUS_COMPLETED = "completed"
    STATUS_FAILED = "failed"
    STATUS_CANCELLED = "cancelled"
    STATUS_EXPIRED = "expired"
    STATUS_CHOICES = [
        (STATUS_PENDING, "Pending"),
        (STATUS_RUNNING, "Running"),
        (STATUS_COMPLETED, "Completed"),
        (STATUS_FAILED, "Failed"),
        (STATUS_CANCELLED, "Cancelled"),
        (STATUS_EXPIRED, "Expired"),
    ]

    FORMAT_COCO = "coco"
    FORMAT_YOLO = "yolo"
    FORMAT_CHOICES = [
        (FORMAT_COCO, "COCO JSON"),
        (FORMAT_YOLO, "YOLO"),
    ]

    project = models.ForeignKey(
        "anno_projects.Project",
        on_delete=models.CASCADE,
        related_name="export_tasks",
    )
    created_by = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.PROTECT,
        related_name="export_tasks",
    )
    format = models.CharField(max_length=10, choices=FORMAT_CHOICES)
    include_images = models.BooleanField(default=False)
    status = models.CharField(
        max_length=20, choices=STATUS_CHOICES, default=STATUS_PENDING, db_index=True
    )
    expires_at = models.DateTimeField(
        null=True,
        blank=True,
        default=None,
        help_text="When the export file should be cleaned up (null = never expire).",
    )
    error = models.TextField(blank=True, default="")
    created_at = models.DateTimeField(auto_now_add=True)
    started_at = models.DateTimeField(null=True, blank=True)
    finished_at = models.DateTimeField(null=True, blank=True)

    class Meta(SoftDeleteModel.Meta):
        db_table = "anno_export_task"
        ordering = ["-created_at"]
        indexes = [
            models.Index(fields=["project", "status"]),
            models.Index(fields=["status", "expires_at"]),
        ]

    def __str__(self) -> str:
        return f"ExportTask #{self.pk} ({self.format}, {self.status})"


class ExportTaskResult(models.Model):
    task = models.OneToOneField(
        ExportTask,
        on_delete=models.CASCADE,
        primary_key=True,
        related_name="result",
    )
    export_file = models.FileField(
        upload_to=_export_upload_to,
        null=True,
        blank=True,
        default=None,
    )
    file_size = models.BigIntegerField(default=0)
    file_deleted_at = models.DateTimeField(null=True, blank=True, default=None)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = "anno_export_task_result"
        ordering = ["-created_at"]

    def __str__(self) -> str:
        return f"ExportTaskResult for task #{self.task_id}"

    @property
    def file_available(self) -> bool:
        return bool(self.export_file) and self.file_deleted_at is None

    @property
    def file_name(self) -> str:
        if self.export_file and self.export_file.name:
            return os.path.basename(self.export_file.name)
        return ""
