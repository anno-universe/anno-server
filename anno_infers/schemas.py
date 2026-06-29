from datetime import datetime

from django.conf import settings
from ninja import Schema

from anno_images.schemas import Box2DDataInput, Keypoint2DDataInput, Polygon2DDataInput

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
