from datetime import datetime

from django.conf import settings
from ninja import Field, Schema
from pydantic import field_validator

from anno_images.schemas import Box2DDataInput, Keypoint2DDataInput, Polygon2DDataInput
from anno_infers.models import VALID_PROMPT_TYPES, VALID_RESULT_TYPES

# ---------- Project meta ----------


class ProjectMetaOutput(Schema):
    id: int
    name: str
    description: str
    meta_info: dict
    label_mapping: dict
    created_at: datetime
    updated_at: datetime

    @staticmethod
    def from_project(project) -> "ProjectMetaOutput":
        return ProjectMetaOutput(
            id=project.id,
            name=project.name,
            description=project.description,
            meta_info=project.meta_info,
            label_mapping=project.label_mapping,
            created_at=project.created_at,
            updated_at=project.updated_at,
        )


# ---------- Project images ----------


class ProjectImageOutput(Schema):
    id: int
    file_name: str
    width: int | None
    height: int | None
    file_url: str

    @staticmethod
    def from_image(img) -> "ProjectImageOutput":
        base = settings.INFERS_BASE_URL.rstrip("/")
        return ProjectImageOutput(
            id=img.id,
            file_name=img.file_name,
            width=img.width,
            height=img.height,
            file_url=f"{base}/api/infers/project/images/{img.id}/original_file",
        )


# ---------- Project annotation submission ----------


class ProjectAnnotationResultItem(Schema):
    client_ref: str | None = None
    image_id: int
    annotation_id: int | None = None
    status: str  # "created" | "error"
    error: str | None = None


class ProjectAnnotationBatchOutput(Schema):
    created: int
    failed: int
    results: list[ProjectAnnotationResultItem]


# ---------- Per-image annotation submission ----------


class ProjectImageAnnotationInput(Schema):
    """Single annotation for a specific image (image_id comes from the URL)."""

    annotation_type: str  # "polygon" | "box" | "keypoint"
    label: int | None = None
    polygon: Polygon2DDataInput | None = None
    box: Box2DDataInput | None = None
    keypoint: Keypoint2DDataInput | None = None
    client_ref: str | None = None


class ProjectImageAnnotationBatchInput(Schema):
    """Wrapper for per-image annotation submission."""

    annotations: list[ProjectImageAnnotationInput]


# ---------- Per-image annotation modify ----------


class ProjectAnnotationModifyOutput(Schema):
    """Returned after modifying an annotation via the infer API."""

    id: int
    image_id: int
    annotation_type: str
    label: int | None
    data: dict
    is_active: bool
    created_at: datetime
    modified_at: datetime

    @staticmethod
    def from_annotation(annotation) -> "ProjectAnnotationModifyOutput":
        data: dict = {}
        try:
            subtype = annotation.polygon
            data = {"points": subtype.points}
        except Exception:
            pass
        try:
            subtype = annotation.box
            data = {
                "x": subtype.x,
                "y": subtype.y,
                "width": subtype.width,
                "height": subtype.height,
                "rotation": subtype.rotation,
            }
        except Exception:
            pass
        try:
            subtype = annotation.keypoint
            data = {"points": subtype.points}
        except Exception:
            pass

        return ProjectAnnotationModifyOutput(
            id=annotation.id,
            image_id=annotation.image_id,
            annotation_type=annotation.annotation_type,
            label=annotation.label,
            data=data,
            is_active=annotation.is_active,
            created_at=annotation.created_at,
            modified_at=annotation.updated_at,
        )


# ---------- Inference service providers (Flow B) ----------


def _validate_result_types(value: list[str]) -> list[str]:
    invalid = [t for t in value if t not in VALID_RESULT_TYPES]
    if invalid:
        allowed = ", ".join(sorted(VALID_RESULT_TYPES))
        raise ValueError(f"Invalid result types {invalid}; allowed: {allowed}.")
    return value


class ProviderCreateInput(Schema):
    name: str
    inference_url: str = Field(description="The service's base URL (e.g., https://infer.example.com). /predict will be appended by the platform.")
    supported_result_types: list[str]
    model_name: str = ""
    description: str = ""
    auth_type: str = "none"  # "none" | "header" | "query"
    auth_param_name: str = ""
    auth_secret: str = ""  # plaintext credential presented to the service
    timeout_seconds: int = 60
    is_active: bool = True

    @field_validator("supported_result_types")
    @classmethod
    def _check_result_types(cls, v: list[str]) -> list[str]:
        return _validate_result_types(v)

    @field_validator("auth_type")
    @classmethod
    def _check_auth_type(cls, v: str) -> str:
        if v not in ("none", "header", "query"):
            raise ValueError("auth_type must be 'none', 'header' or 'query'.")
        return v


class ProviderUpdateInput(Schema):
    """Partial update; all fields optional. ``auth_secret`` is write-only."""

    name: str | None = None
    inference_url: str | None = Field(default=None, description="The service's base URL (e.g., https://infer.example.com). /predict will be appended by the platform.")
    supported_result_types: list[str] | None = None
    model_name: str | None = None
    description: str | None = None
    auth_type: str | None = None
    auth_param_name: str | None = None
    auth_secret: str | None = None
    timeout_seconds: int | None = None
    is_active: bool | None = None

    @field_validator("supported_result_types")
    @classmethod
    def _check_result_types(cls, v: list[str] | None) -> list[str] | None:
        return None if v is None else _validate_result_types(v)

    @field_validator("auth_type")
    @classmethod
    def _check_auth_type(cls, v: str | None) -> str | None:
        if v is not None and v not in ("none", "header", "query"):
            raise ValueError("auth_type must be 'none', 'header' or 'query'.")
        return v


class ProviderOutput(Schema):
    """Provider representation for API responses.

    Deliberately omits ``auth_secret``. ``has_auth_secret`` lets clients show
    whether a credential is configured without exposing it. ``is_global`` is
    true for admin-managed providers shared across projects.
    """

    id: int
    name: str
    model_name: str
    description: str
    inference_url: str = Field(description="The service's base URL (e.g., https://infer.example.com). /predict will be appended by the platform.")
    supported_result_types: list[str]
    auth_type: str
    auth_param_name: str
    has_auth_secret: bool
    timeout_seconds: int
    is_active: bool
    is_global: bool
    created_at: datetime
    updated_at: datetime

    @staticmethod
    def from_provider(p) -> "ProviderOutput":
        return ProviderOutput(
            id=p.id,
            name=p.name,
            model_name=p.model_name,
            description=p.description,
            inference_url=p.inference_url,
            supported_result_types=p.supported_result_types,
            auth_type=p.auth_type,
            auth_param_name=p.auth_param_name,
            has_auth_secret=bool(p.auth_secret),
            timeout_seconds=p.timeout_seconds,
            is_active=p.is_active,
            is_global=p.project_id is None,
            created_at=p.created_at,
            updated_at=p.updated_at,
        )


# ---------- Auto-annotation jobs (Flow B) ----------


class AutoAnnotateInput(Schema):
    """Kick off an auto-annotation job for ALL images in the project.

    The endpoint always targets every image in the project. If you need
    to process only a subset, create a separate project for those images.
    """

    provider_id: int


class ResultOutput(Schema):
    """One candidate result the model returned for a single image."""

    id: int
    result_index: int
    result_type: str
    label: int | None
    score: float | None
    result_data: dict
    annotation_id: int | None
    status: str
    created_at: datetime
    committed_at: datetime | None
    rejected_at: datetime | None

    @staticmethod
    def from_result(r) -> "ResultOutput":
        return ResultOutput(
            id=r.id,
            result_index=r.result_index,
            result_type=r.result_type,
            label=r.label,
            score=r.score,
            result_data=r.result_data,
            annotation_id=r.annotation_id,
            status=r.status,
            created_at=r.created_at,
            committed_at=r.committed_at,
            rejected_at=r.rejected_at,
        )


class TaskOutput(Schema):
    """One image's unit of work within a run."""

    id: int
    run_id: int
    image_id: int
    status: str
    annotations_created: int
    attempts: int
    error: str

    @staticmethod
    def from_task(task) -> "TaskOutput":
        return TaskOutput(
            id=task.id,
            run_id=task.run_id,
            image_id=task.image_id,
            status=task.status,
            annotations_created=task.annotations_created,
            attempts=task.attempts,
            error=task.error,
        )


class TaskDetailOutput(TaskOutput):
    """Per-image task with its candidate results."""

    results: list[ResultOutput] = []

    @staticmethod
    def from_task_with_results(task) -> "TaskDetailOutput":
        base = TaskOutput.from_task(task)
        return TaskDetailOutput(
            **base.dict(),
            results=[ResultOutput.from_result(r) for r in task.results.all()],
        )


class RunOutput(Schema):
    """A triggered auto-annotation run over a set of images (1..N)."""

    id: int
    project_id: int
    provider_id: int
    status: str
    provider_snapshot: dict
    total_items: int
    completed_items: int
    failed_items: int
    annotations_created: int
    cancel_requested: bool
    error: str
    created_by_id: int
    created_at: datetime
    started_at: datetime | None
    finished_at: datetime | None

    @staticmethod
    def from_run(run) -> "RunOutput":
        return RunOutput(
            id=run.id,
            project_id=run.project_id,
            provider_id=run.provider_id,
            status=run.status,
            provider_snapshot=run.provider_snapshot,
            total_items=run.total_items,
            completed_items=run.completed_items,
            failed_items=run.failed_items,
            annotations_created=run.annotations_created,
            cancel_requested=run.cancel_requested,
            error=run.error,
            created_by_id=run.created_by_id,
            created_at=run.created_at,
            started_at=run.started_at,
            finished_at=run.finished_at,
        )


class RunDetailOutput(RunOutput):
    tasks: list[TaskOutput] = []

    @staticmethod
    def from_run_with_tasks(run) -> "RunDetailOutput":
        base = RunOutput.from_run(run)
        return RunDetailOutput(
            **base.dict(),
            tasks=[TaskOutput.from_task(t) for t in run.tasks.all()],
        )


# ---------- Single-image auto-annotation (Flow B) ----------


class ImageAutoAnnotateInput(Schema):
    """Trigger server-driven inference for a single image asynchronously.

    Creates a single-item ``InferenceJob`` and enqueues it for the background
    worker, returning the job details immediately — same pattern as the batch
    endpoint. Use ``GET /jobs/{job_id}`` to track progress and results.
    """

    provider_id: int


# ---------- Interactive inference providers ----------


def _validate_prompt_types(value: list[str]) -> list[str]:
    invalid = [t for t in value if t not in VALID_PROMPT_TYPES]
    if invalid:
        allowed = ", ".join(sorted(VALID_PROMPT_TYPES))
        raise ValueError(f"Invalid prompt types {invalid}; allowed: {allowed}.")
    return value


class InteractiveProviderCreateInput(Schema):
    name: str
    inference_url: str = Field(description="The service's base URL (e.g., https://infer.example.com). /session will be appended by the platform.")
    supported_prompt_types: list[str]
    supported_result_types: list[str]
    model_name: str = ""
    description: str = ""
    auth_type: str = "none"  # "none" | "header" | "query"
    auth_param_name: str = ""
    auth_secret: str = ""  # plaintext credential presented to the service
    timeout_seconds: int = 60
    is_active: bool = True

    @field_validator("supported_prompt_types")
    @classmethod
    def _check_prompt_types(cls, v: list[str]) -> list[str]:
        return _validate_prompt_types(v)

    @field_validator("supported_result_types")
    @classmethod
    def _check_result_types(cls, v: list[str]) -> list[str]:
        return _validate_result_types(v)

    @field_validator("auth_type")
    @classmethod
    def _check_auth_type(cls, v: str) -> str:
        if v not in ("none", "header", "query"):
            raise ValueError("auth_type must be 'none', 'header' or 'query'.")
        return v


class InteractiveProviderUpdateInput(Schema):
    """Partial update; all fields optional. ``auth_secret`` is write-only."""

    name: str | None = None
    inference_url: str | None = Field(default=None, description="The service's base URL (e.g., https://infer.example.com). /session will be appended by the platform.")
    supported_prompt_types: list[str] | None = None
    supported_result_types: list[str] | None = None
    model_name: str | None = None
    description: str | None = None
    auth_type: str | None = None
    auth_param_name: str | None = None
    auth_secret: str | None = None
    timeout_seconds: int | None = None
    is_active: bool | None = None

    @field_validator("supported_prompt_types")
    @classmethod
    def _check_prompt_types(cls, v: list[str] | None) -> list[str] | None:
        return None if v is None else _validate_prompt_types(v)

    @field_validator("supported_result_types")
    @classmethod
    def _check_result_types(cls, v: list[str] | None) -> list[str] | None:
        return None if v is None else _validate_result_types(v)

    @field_validator("auth_type")
    @classmethod
    def _check_auth_type(cls, v: str | None) -> str | None:
        if v is not None and v not in ("none", "header", "query"):
            raise ValueError("auth_type must be 'none', 'header' or 'query'.")
        return v


class InteractiveProviderOutput(Schema):
    """Interactive provider representation; omits ``auth_secret`` like ``ProviderOutput``."""

    id: int
    name: str
    model_name: str
    description: str
    inference_url: str = Field(description="The service's base URL (e.g., https://infer.example.com). /session will be appended by the platform.")
    supported_prompt_types: list[str]
    supported_result_types: list[str]
    auth_type: str
    auth_param_name: str
    has_auth_secret: bool
    timeout_seconds: int
    is_active: bool
    is_global: bool
    created_at: datetime
    updated_at: datetime

    @staticmethod
    def from_provider(p) -> "InteractiveProviderOutput":
        return InteractiveProviderOutput(
            id=p.id,
            name=p.name,
            model_name=p.model_name,
            description=p.description,
            inference_url=p.inference_url,
            supported_prompt_types=p.supported_prompt_types,
            supported_result_types=p.supported_result_types,
            auth_type=p.auth_type,
            auth_param_name=p.auth_param_name,
            has_auth_secret=bool(p.auth_secret),
            timeout_seconds=p.timeout_seconds,
            is_active=p.is_active,
            is_global=p.project_id is None,
            created_at=p.created_at,
            updated_at=p.updated_at,
        )


# ---------- Interactive inference sessions ----------


class InteractiveSessionStartInput(Schema):
    """Open an interactive session on an image (image_id comes from the URL)."""

    provider_id: int


class InteractiveCommitInput(Schema):
    """Commit the user's chosen candidate as a real annotation.

    In the direct-call flow the per-prompt loop runs browser -> service, so the
    server never saw the intermediate steps; the frontend sends the final
    prompts (for audit) plus the chosen geometry here. Exactly one geometry
    field must match ``annotation_type``.
    """

    annotation_type: str
    label: int | None = None
    polygon: Polygon2DDataInput | None = None
    box: Box2DDataInput | None = None
    keypoint: Keypoint2DDataInput | None = None
    prompts: list[dict] = []
    score: float | None = None
    model_version: str = ""


class InteractiveStepOutput(Schema):
    id: int
    session_id: int
    step_index: int
    prompt: dict
    result: dict
    result_type: str
    result_data: dict
    annotation_id: int | None
    error: str
    created_at: datetime

    @staticmethod
    def from_operation(op) -> "InteractiveStepOutput":
        return InteractiveStepOutput(
            id=op.id,
            session_id=op.session_id,
            step_index=op.step_index,
            prompt=op.prompt,
            result=op.result,
            result_type=op.result_type,
            result_data=op.result_data,
            annotation_id=op.annotation_id,
            error=op.error,
            created_at=op.created_at,
        )


class InteractiveSessionOutput(Schema):
    id: int
    project_id: int
    image_id: int
    provider_id: int
    performed_by_id: int
    status: str
    error: str
    created_at: datetime
    updated_at: datetime

    @staticmethod
    def from_session(s) -> "InteractiveSessionOutput":
        return InteractiveSessionOutput(
            id=s.id,
            project_id=s.project_id,
            image_id=s.image_id,
            provider_id=s.provider_id,
            performed_by_id=s.performed_by_id,
            status=s.status,
            error=s.error,
            created_at=s.created_at,
            updated_at=s.updated_at,
        )


class InteractiveSessionStartOutput(InteractiveSessionOutput):
    """Session record plus the short-lived credential for the direct calls.

    The frontend presents ``token`` in the ``token_header`` header on its direct
    calls to the service — ``{predict_url}/{id}/infer_image`` (upload the image
    once) and ``{predict_url}/{id}/predict`` (per prompt). ``token_expires_at`` is
    the service-reported ISO-8601 expiry.
    """

    token: str
    token_header: str
    token_expires_at: str | None = None
    predict_url: str | None = None
    session_ref: str | None = None
    supported_prompt_types: list[str] = []
    supported_result_types: list[str] = []


class InteractiveSessionDetailOutput(InteractiveSessionOutput):
    steps: list[InteractiveStepOutput] = []

    @staticmethod
    def from_session_with_steps(s) -> "InteractiveSessionDetailOutput":
        base = InteractiveSessionOutput.from_session(s)
        return InteractiveSessionDetailOutput(
            **base.dict(),
            steps=[InteractiveStepOutput.from_operation(op) for op in s.operations.all()],
        )
