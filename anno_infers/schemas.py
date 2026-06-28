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


# ---------- Project annotation submission (the Flow-A protocol) ----------


class ProjectAnnotationItemInput(Schema):
    image_id: int
    annotation_type: str  # "polygon" | "box" | "keypoint"
    label: int | None = None
    polygon: Polygon2DDataInput | None = None
    box: Box2DDataInput | None = None
    keypoint: Keypoint2DDataInput | None = None
    client_ref: str | None = None  # echoed back so the edge side can correlate


class ProjectAnnotationBatchInput(Schema):
    items: list[ProjectAnnotationItemInput]


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
