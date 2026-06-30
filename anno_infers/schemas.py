from datetime import datetime

from django.conf import settings
from ninja import Schema
from pydantic import field_validator

from anno_images.schemas import Box2DDataInput, Keypoint2DDataInput, Polygon2DDataInput
from anno_infers.models import VALID_RESULT_TYPES

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
    inference_url: str
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
    inference_url: str | None = None
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
    inference_url: str
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


class JobItemOutput(Schema):
    id: int
    image_id: int
    status: str
    annotations_created: int
    attempts: int
    error: str

    @staticmethod
    def from_item(item) -> "JobItemOutput":
        return JobItemOutput(
            id=item.id,
            image_id=item.image_id,
            status=item.status,
            annotations_created=item.annotations_created,
            attempts=item.attempts,
            error=item.error,
        )


class JobOutput(Schema):
    id: int
    project_id: int
    provider_id: int
    status: str
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
    def from_job(job) -> "JobOutput":
        return JobOutput(
            id=job.id,
            project_id=job.project_id,
            provider_id=job.provider_id,
            status=job.status,
            total_items=job.total_items,
            completed_items=job.completed_items,
            failed_items=job.failed_items,
            annotations_created=job.annotations_created,
            cancel_requested=job.cancel_requested,
            error=job.error,
            created_by_id=job.created_by_id,
            created_at=job.created_at,
            started_at=job.started_at,
            finished_at=job.finished_at,
        )


class JobDetailOutput(JobOutput):
    items: list[JobItemOutput] = []

    @staticmethod
    def from_job_with_items(job) -> "JobDetailOutput":
        base = JobOutput.from_job(job)
        return JobDetailOutput(
            **base.dict(),
            items=[JobItemOutput.from_item(i) for i in job.items.all()],
        )


# ---------- Single-image auto-annotation (Flow B) ----------


class ImageAutoAnnotateInput(Schema):
    """Trigger server-driven inference for a single image asynchronously.

    Creates a single-item ``InferenceJob`` and enqueues it for the background
    worker, returning the job details immediately — same pattern as the batch
    endpoint. Use ``GET /jobs/{job_id}`` to track progress and results.
    """

    provider_id: int
