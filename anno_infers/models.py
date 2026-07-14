"""Models for server-driven auto-annotation (Flow B).

``InferenceServiceProvider`` is the server-side registry of long-running
inference services. A triggered run is a three-layer structure::

    InferenceServiceProvider
      └── InferenceRun       one run over a set of images (1..N)
            └── InferenceTask   one image's unit of work within the run
                  └── InferenceResult   one candidate result the model returned

A single-image inference is just a run with one task. The run/task statuses,
``cancel_requested`` flag, ``deadline`` and per-task ``attempts`` give
persistence, cooperative cancellation, timeout and replay at the application
layer — independent of the task executor that runs the run. ``InferenceResult``
records each candidate the model returned and, once committed, links one-to-one
to the ``Annotation2D`` it became (the reverse-lookup target for
``Operation.source == "inference"``).
"""

from django.conf import settings
from django.db import models

from anno.models import SoftDeleteModel

# Annotation geometry types an inference service may declare / return.
RESULT_TYPE_CHOICES = [
    ("polygon", "Polygon"),
    ("box", "Box"),
    ("keypoint", "Keypoint"),
]
VALID_RESULT_TYPES = frozenset(t for t, _ in RESULT_TYPE_CHOICES)

# Prompt types an interactive inference service (SAM/SAM2/MedSAM style) accepts.
PROMPT_TYPE_CHOICES = [
    ("box", "Box"),
    ("positive_point", "Positive point"),
    ("negative_point", "Negative point"),
    ("mask", "Mask"),
    ("text", "Text"),
]
VALID_PROMPT_TYPES = frozenset(t for t, _ in PROMPT_TYPE_CHOICES)


class InferenceServiceProvider(SoftDeleteModel):
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
    inference_url = models.CharField(max_length=1024, help_text="The service's base URL (e.g., https://infer.example.com or http://anno-sam:8422 for an internal Docker service). The platform appends standard SDK paths (/predict, /session, etc.) automatically.")

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

    class Meta(SoftDeleteModel.Meta):
        db_table = "anno_inference_provider"
        ordering = ["-created_at"]
        indexes = [
            models.Index(fields=["project", "is_active"]),
            models.Index(fields=["is_active"]),
        ]

    def __str__(self) -> str:
        scope = "global" if self.project_id is None else f"project={self.project_id}"
        return f"{self.name} ({scope})"


class InferenceRun(models.Model):
    """A triggered auto-annotation run over a set of images (1..N).

    A batch run targets many images; a single-image inference is a run with one
    task. ``provider`` and ``created_by`` are carried here and shared by every
    task in the run.
    """

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
        "anno_projects.Project", on_delete=models.CASCADE, related_name="inference_runs"
    )
    provider = models.ForeignKey(
        InferenceServiceProvider, on_delete=models.PROTECT, related_name="runs"
    )
    status = models.CharField(
        max_length=20, choices=STATUS_CHOICES, default=STATUS_PENDING, db_index=True
    )

    provider_snapshot = models.JSONField(
        default=dict,
        blank=True,
        help_text="Provider config snapshot at creation time (never includes auth_secret).",
    )

    total_items = models.PositiveIntegerField(default=0)
    completed_items = models.PositiveIntegerField(default=0)
    failed_items = models.PositiveIntegerField(default=0)
    annotations_created = models.PositiveIntegerField(default=0)

    cancel_requested = models.BooleanField(default=False)
    deadline = models.DateTimeField(null=True, blank=True)
    error = models.TextField(blank=True, default="")

    created_by = models.ForeignKey(
        settings.AUTH_USER_MODEL, on_delete=models.PROTECT, related_name="inference_runs"
    )
    created_at = models.DateTimeField(auto_now_add=True)
    started_at = models.DateTimeField(null=True, blank=True)
    finished_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        db_table = "anno_inference_run"
        ordering = ["-created_at"]
        indexes = [
            models.Index(fields=["project", "status"]),
            models.Index(fields=["status", "created_at"]),
        ]

    def __str__(self) -> str:
        return f"InferenceRun #{self.pk} ({self.status})"


class InferenceTask(models.Model):
    """One image's unit of work within a run."""

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

    run = models.ForeignKey(
        InferenceRun,
        on_delete=models.CASCADE,
        related_name="tasks",
    )
    image = models.ForeignKey(
        "anno_images.Image2D", on_delete=models.CASCADE, related_name="inference_tasks"
    )
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default=STATUS_PENDING)
    annotations_created = models.PositiveIntegerField(default=0)
    error = models.TextField(blank=True, default="")
    attempts = models.PositiveIntegerField(default=0)
    started_at = models.DateTimeField(null=True, blank=True)
    finished_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        db_table = "anno_inference_task"
        ordering = ["id"]
        constraints = [
            models.UniqueConstraint(fields=["run", "image"], name="uniq_run_image"),
        ]
        indexes = [
            models.Index(fields=["run", "status"]),
        ]

    def __str__(self) -> str:
        return (
            f"InferenceTask #{self.pk} "
            f"(run={self.run_id}, image={self.image_id}, {self.status})"
        )


class InferenceResult(models.Model):
    """One candidate annotation the model returned for a single image.

    Records the raw and normalized result plus, once committed, the
    ``Annotation2D`` it became (one-to-one). Given an ``Operation`` with
    ``source == "inference"``, the result is reverse-traceable via
    ``InferenceResult.objects.get(annotation_id=operation.to_annotation_id)``.
    """

    STATUS_PROPOSED = "proposed"
    STATUS_COMMITTED = "committed"
    STATUS_REJECTED = "rejected"
    STATUS_CHOICES = [
        (STATUS_PROPOSED, "Proposed"),
        (STATUS_COMMITTED, "Committed"),
        (STATUS_REJECTED, "Rejected"),
    ]

    task = models.ForeignKey(InferenceTask, on_delete=models.CASCADE, related_name="results")
    result_index = models.PositiveIntegerField(
        help_text="Position of this result in the model's returned list."
    )
    result_type = models.CharField(max_length=20, choices=RESULT_TYPE_CHOICES)
    label = models.IntegerField(null=True, blank=True)
    score = models.FloatField(null=True, blank=True)
    result_data = models.JSONField(
        default=dict,
        blank=True,
        help_text="Normalized geometry, e.g. {'points': [...]} or box fields.",
    )
    raw_result = models.JSONField(
        default=dict, blank=True, help_text="The model's raw returned result."
    )
    annotation = models.OneToOneField(
        "anno_images.Annotation2D",
        null=True,
        blank=True,
        on_delete=models.SET_NULL,
        related_name="inference_result",
        help_text="The Annotation2D this result was committed as (null until committed).",
    )
    status = models.CharField(
        max_length=20, choices=STATUS_CHOICES, default=STATUS_PROPOSED, db_index=True
    )
    created_at = models.DateTimeField(auto_now_add=True)
    committed_at = models.DateTimeField(null=True, blank=True)
    rejected_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        db_table = "anno_inference_result"
        ordering = ["task_id", "result_index"]
        constraints = [
            models.UniqueConstraint(
                fields=["task", "result_index"], name="uniq_task_result_index"
            ),
        ]
        indexes = [
            models.Index(fields=["task", "status"]),
        ]

    def __str__(self) -> str:
        return f"InferenceResult #{self.pk} (task={self.task_id}, {self.status})"


# ---------------------------------------------------------------------------
# Interactive inference (SAM/SAM2/MedSAM style)
#
# A user iteratively supplies prompts (box / positive-point / negative-point /
# mask / text) and the model returns one refined candidate each step. The
# candidate is recorded on an :class:`InteractiveInferenceOperation` and becomes
# a real ``Annotation2D`` only when the user commits the session.
# ---------------------------------------------------------------------------


class InteractiveInferenceServiceProvider(SoftDeleteModel):
    """A registered interactive inference service the server can call.

    Like :class:`InferenceServiceProvider` but the service is prompt-driven:
    ``supported_prompt_types`` declares which prompt kinds it accepts. ``project``
    is nullable (``null`` => global provider). ``auth_secret`` is stored in
    plaintext and must never be serialized in an API response.
    """

    AUTH_NONE = "none"
    AUTH_HEADER = "header"
    AUTH_QUERY = "query"
    AUTH_CHOICES = [
        (AUTH_NONE, "None"),
        (AUTH_HEADER, "Header"),
        (AUTH_QUERY, "Query param"),
    ]

    name = models.CharField(max_length=255, help_text="Human label, e.g. 'SAM-2 interactive'.")
    model_name = models.CharField(max_length=255, blank=True, default="")
    description = models.TextField(blank=True, default="")
    inference_url = models.CharField(max_length=1024, help_text="The service's base URL (e.g., https://infer.example.com or http://anno-sam:8422 for an internal Docker service). The platform appends standard SDK paths (/predict, /session, etc.) automatically.")
    public_url = models.CharField(
        max_length=1024,
        blank=True,
        default="",
        help_text=(
            "Browser-reachable base URL for the frontend's direct predict calls, e.g. "
            "'/_interactive_infer' (a same-origin path the reverse proxy forwards to the "
            "service) or 'https://host/_interactive_infer'. Distinct from inference_url, "
            "which is the server-side address used for the handshake. Blank => fall back "
            "to the service handshake, then inference_url."
        ),
    )

    supported_prompt_types = models.JSONField(
        default=list,
        help_text="Subset of {'box','positive_point','negative_point','mask','text'} accepted.",
    )
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
        default=60, help_text="Per-step HTTP timeout when calling the service."
    )
    is_active = models.BooleanField(default=True)

    project = models.ForeignKey(
        "anno_projects.Project",
        null=True,
        blank=True,
        on_delete=models.CASCADE,
        related_name="interactive_inference_providers",
        help_text="Null => global provider usable by any project.",
    )

    created_by = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.PROTECT,
        related_name="created_interactive_inference_providers",
    )
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta(SoftDeleteModel.Meta):
        db_table = "anno_interactive_inference_provider"
        ordering = ["-created_at"]
        indexes = [
            models.Index(fields=["project", "is_active"]),
            models.Index(fields=["is_active"]),
        ]

    def __str__(self) -> str:
        scope = "global" if self.project_id is None else f"project={self.project_id}"
        return f"{self.name} ({scope})"


class InteractiveInferenceSession(models.Model):
    """One user's interactive prompting session over a single image.

    One session can produce multiple annotations — commit does *not* end the
    session; only an explicit discard does. Each commit creates an
    ``Annotation2D`` and an ``InteractiveInferenceOperation`` linked to it.

    To reverse-trace from an audit ``Operation`` (``source == "interactive"``)
    back to the session that produced it, use
    ``InteractiveInferenceOperation.objects.get(
    annotation_id=op.to_annotation_id).session``.
    """

    STATUS_EDITING = "editing"
    STATUS_DISCARDED = "discarded"
    STATUS_FAILED = "failed"
    STATUS_CHOICES = [
        (STATUS_EDITING, "Editing"),
        (STATUS_DISCARDED, "Discarded"),
        (STATUS_FAILED, "Failed"),
    ]

    project = models.ForeignKey(
        "anno_projects.Project",
        on_delete=models.CASCADE,
        related_name="interactive_inference_sessions",
    )
    image = models.ForeignKey(
        "anno_images.Image2D",
        on_delete=models.CASCADE,
        related_name="interactive_inference_sessions",
    )
    provider = models.ForeignKey(
        InteractiveInferenceServiceProvider,
        on_delete=models.PROTECT,
        related_name="sessions",
    )
    performed_by = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.PROTECT,
        related_name="interactive_inference_sessions",
    )
    status = models.CharField(
        max_length=12, choices=STATUS_CHOICES, default=STATUS_EDITING, db_index=True
    )
    session_token = models.CharField(
        max_length=128,
        blank=True,
        null=True,
        unique=True,
        default=None,
        help_text="Short-lived token the service minted for browser→service calls. "
        "NULL until the handshake succeeds; unique thereafter.",
    )
    error = models.TextField(blank=True, default="")
    created_at = models.DateTimeField(auto_now_add=True)
    # ``updated_at`` is bumped on every commit and on discard, serving as the
    # ended-at timestamp only when ``status`` is terminal (discarded/failed).
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        db_table = "anno_interactive_session"
        ordering = ["-created_at"]
        indexes = [
            models.Index(fields=["project", "status"]),
        ]

    def __str__(self) -> str:
        return f"InteractiveInferenceSession #{self.pk} ({self.status})"


class InteractiveInferenceOperation(models.Model):
    """One step within a session: a set of prompts and the model's candidate.

    This is *not* an audit :class:`~anno_images.models.Operation`; it records the
    interaction only. When a step is committed, ``annotation`` links to the
    resulting ``Annotation2D``, providing a reverse-trace from audit ``Operation``
    back to the session:
    ``InteractiveInferenceOperation.objects.get(
    annotation_id=audit_op.to_annotation_id).session``.
    """

    session = models.ForeignKey(
        InteractiveInferenceSession, on_delete=models.CASCADE, related_name="operations"
    )
    annotation = models.ForeignKey(
        "anno_images.Annotation2D",
        null=True,
        blank=True,
        on_delete=models.SET_NULL,
        related_name="interactive_operations",
        help_text="The Annotation2D created by this operation step (null until committed).",
    )
    step_index = models.PositiveIntegerField(help_text="Step number within the session, from 1.")
    prompt = models.JSONField(
        default=dict, blank=True, help_text="The prompts the user supplied this step."
    )
    result = models.JSONField(
        default=dict,
        blank=True,
        help_text="Summary of the model output, e.g. score / bbox / area.",
    )
    result_type = models.CharField(
        max_length=20, choices=RESULT_TYPE_CHOICES, blank=True, default=""
    )
    result_data = models.JSONField(
        default=dict,
        blank=True,
        help_text="Normalized candidate geometry, e.g. {'points': [...]}.",
    )
    raw_result = models.JSONField(
        default=dict, blank=True, help_text="The service's raw returned result."
    )
    error = models.TextField(blank=True, default="")
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = "anno_interactive_operation"
        ordering = ["session_id", "step_index"]
        constraints = [
            models.UniqueConstraint(
                fields=["session", "step_index"], name="uniq_session_step_index"
            ),
        ]
        indexes = [models.Index(fields=["session", "step_index"])]

    def __str__(self) -> str:
        return (
            f"InteractiveInferenceOperation #{self.pk} "
            f"(session={self.session_id}, step={self.step_index})"
        )
