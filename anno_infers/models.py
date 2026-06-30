"""Models for server-driven auto-annotation (Flow B).

``InferenceServiceProvider`` is the server-side registry of long-running
inference services. ``InferenceJob`` / ``InferenceJobItem`` track a
supervisor-triggered auto-annotation run; their statuses, ``cancel_requested``
flag, ``deadline`` and per-item ``attempts`` give persistence, cooperative
cancellation, timeout and replay at the application layer — independent of the
task executor that happens to run the job.
"""

from django.conf import settings
from django.db import models

# Annotation geometry types an inference service may declare / return.
RESULT_TYPE_CHOICES = [
    ("polygon", "Polygon"),
    ("box", "Box"),
    ("keypoint", "Keypoint"),
]
VALID_RESULT_TYPES = frozenset(t for t, _ in RESULT_TYPE_CHOICES)


class InferenceServiceProvider(models.Model):
    """A registered inference service the server can call to auto-annotate.

    ``project`` is nullable: a ``null`` row is a *global* provider (admin
    managed) usable by any project; a non-null row is scoped to one project.

    Auth is the credential the *anno server* presents to the service so the
    service accepts the request. ``auth_secret`` is stored in plaintext (the
    service-side credential, replayed outbound on every call) and must never be
    serialized in an API response.
    """

    AUTH_NONE = "none"
    AUTH_HEADER = "header"
    AUTH_QUERY = "query"
    AUTH_CHOICES = [
        (AUTH_NONE, "None"),
        (AUTH_HEADER, "Header"),
        (AUTH_QUERY, "Query param"),
    ]

    name = models.CharField(max_length=255, help_text="Human label, e.g. 'SAM-2 box->mask'.")
    model_name = models.CharField(max_length=255, blank=True, default="")
    description = models.TextField(blank=True, default="")
    inference_url = models.URLField(max_length=1024, help_text="The service's predict endpoint.")

    supported_result_types = models.JSONField(
        default=list,
        help_text="Subset of {'polygon','box','keypoint'} this service can return.",
    )

    # --- auth (server -> service) ---
    auth_type = models.CharField(max_length=10, choices=AUTH_CHOICES, default=AUTH_NONE)
    auth_param_name = models.CharField(
        max_length=128,
        blank=True,
        default="",
        help_text="Header name (e.g. 'Authorization', 'X-API-Key') or query key (e.g. 'api_key').",
    )
    auth_secret = models.CharField(
        max_length=512,
        blank=True,
        default="",
        help_text="Plaintext credential value presented to the service. Never serialized in API output.",
    )

    timeout_seconds = models.PositiveIntegerField(
        default=60, help_text="Per-image HTTP timeout when calling the service."
    )
    is_active = models.BooleanField(default=True)

    project = models.ForeignKey(
        "anno_projects.Project",
        null=True,
        blank=True,
        on_delete=models.CASCADE,
        related_name="inference_providers",
        help_text="Null => global provider usable by any project.",
    )

    created_by = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.PROTECT,
        related_name="created_inference_providers",
    )
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        db_table = "anno_inference_provider"
        ordering = ["-created_at"]
        indexes = [
            models.Index(fields=["project", "is_active"]),
            models.Index(fields=["is_active"]),
        ]

    def __str__(self) -> str:
        scope = "global" if self.project_id is None else f"project={self.project_id}"
        return f"{self.name} ({scope})"


class InferenceJob(models.Model):
    """A supervisor-triggered auto-annotation run over a set of images."""

    STATUS_PENDING = "pending"
    STATUS_RUNNING = "running"
    STATUS_COMPLETED = "completed"
    STATUS_FAILED = "failed"
    STATUS_CANCELLING = "cancelling"
    STATUS_CANCELLED = "cancelled"
    STATUS_CHOICES = [
        (STATUS_PENDING, "Pending"),
        (STATUS_RUNNING, "Running"),
        (STATUS_COMPLETED, "Completed"),
        (STATUS_FAILED, "Failed"),
        (STATUS_CANCELLING, "Cancelling"),
        (STATUS_CANCELLED, "Cancelled"),
    ]

    project = models.ForeignKey(
        "anno_projects.Project", on_delete=models.CASCADE, related_name="inference_jobs"
    )
    provider = models.ForeignKey(
        InferenceServiceProvider, on_delete=models.PROTECT, related_name="jobs"
    )
    status = models.CharField(
        max_length=20, choices=STATUS_CHOICES, default=STATUS_PENDING, db_index=True
    )

    total_items = models.PositiveIntegerField(default=0)
    completed_items = models.PositiveIntegerField(default=0)
    failed_items = models.PositiveIntegerField(default=0)
    annotations_created = models.PositiveIntegerField(default=0)

    cancel_requested = models.BooleanField(default=False)
    deadline = models.DateTimeField(null=True, blank=True)
    error = models.TextField(blank=True, default="")

    created_by = models.ForeignKey(
        settings.AUTH_USER_MODEL, on_delete=models.PROTECT, related_name="inference_jobs"
    )
    created_at = models.DateTimeField(auto_now_add=True)
    started_at = models.DateTimeField(null=True, blank=True)
    finished_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        db_table = "anno_inference_job"
        ordering = ["-created_at"]
        indexes = [
            models.Index(fields=["project", "status"]),
            models.Index(fields=["status", "created_at"]),
        ]

    def __str__(self) -> str:
        return f"InferenceJob #{self.pk} ({self.status})"


class InferenceJobItem(models.Model):
    """One image's unit of work within an :class:`InferenceJob`."""

    STATUS_PENDING = "pending"
    STATUS_RUNNING = "running"
    STATUS_DONE = "done"
    STATUS_FAILED = "failed"
    STATUS_SKIPPED = "skipped"
    STATUS_CHOICES = [
        (STATUS_PENDING, "Pending"),
        (STATUS_RUNNING, "Running"),
        (STATUS_DONE, "Done"),
        (STATUS_FAILED, "Failed"),
        (STATUS_SKIPPED, "Skipped"),
    ]

    job = models.ForeignKey(InferenceJob, on_delete=models.CASCADE, related_name="items")
    image = models.ForeignKey(
        "anno_images.Image2D", on_delete=models.CASCADE, related_name="inference_job_items"
    )
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default=STATUS_PENDING)
    annotations_created = models.PositiveIntegerField(default=0)
    error = models.TextField(blank=True, default="")
    attempts = models.PositiveIntegerField(default=0)
    started_at = models.DateTimeField(null=True, blank=True)
    finished_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        db_table = "anno_inference_job_item"
        ordering = ["id"]
        constraints = [
            models.UniqueConstraint(fields=["job", "image"], name="uniq_job_image"),
        ]
        indexes = [models.Index(fields=["job", "status"])]

    def __str__(self) -> str:
        return f"JobItem #{self.pk} (job={self.job_id}, image={self.image_id}, {self.status})"
