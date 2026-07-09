from datetime import datetime

from ninja import Schema

from anno_tags.schemas import ImageTagOutput

# ---------- Image2D ----------


class Image2DOutput(Schema):
    id: int
    project_id: int
    image_url: str
    thumbnail_url: str
    file_name: str
    width: int | None
    height: int | None
    annotation_count: int = 0
    tags: list[ImageTagOutput] = []
    created_at: datetime
    updated_at: datetime

    @staticmethod
    def from_image(img) -> "Image2DOutput":
        tags = []
        # Populated when prefetch_related("tags__tag") was used on the queryset
        if hasattr(img, "tags") and hasattr(img.tags, "all"):
            tags = [ImageTagOutput.from_image_tag(t) for t in img.tags.all()]
        return Image2DOutput(
            id=img.id,
            project_id=img.project_id,
            image_url=f"/api/projects/{img.project_id}/images/{img.id}/original_image",
            thumbnail_url=f"/api/projects/{img.project_id}/images/{img.id}/thumbnail_image",
            file_name=img.file_name,
            width=img.width,
            height=img.height,
            annotation_count=getattr(img, "annotation_count", 0),
            tags=tags,
            created_at=img.created_at,
            updated_at=img.updated_at,
        )


# ---------- Annotation2D Subtype Inputs ----------


class Polygon2DDataInput(Schema):
    points: list[list[float]]


class Box2DDataInput(Schema):
    x: float
    y: float
    width: float
    height: float
    rotation: float = 0.0


class Keypoint2DDataInput(Schema):
    points: list[list[float]]


class Annotation2DCreateInput(Schema):
    annotation_type: str
    label: int | None = None
    polygon: Polygon2DDataInput | None = None
    box: Box2DDataInput | None = None
    keypoint: Keypoint2DDataInput | None = None


# ---------- Annotation2D Output ----------


class Annotation2DOutput(Schema):
    id: int
    image_id: int
    project_id: int
    annotation_type: str
    label: int | None
    is_active: bool
    data: dict
    created_at: datetime
    updated_at: datetime

    @staticmethod
    def from_annotation(annotation) -> "Annotation2DOutput":
        data = {}
        if annotation.annotation_type == "polygon" and hasattr(annotation, "polygon"):
            data = {"points": annotation.polygon.points}
        elif annotation.annotation_type == "box" and hasattr(annotation, "box"):
            box = annotation.box
            data = {
                "x": box.x,
                "y": box.y,
                "width": box.width,
                "height": box.height,
                "rotation": box.rotation,
            }
        elif annotation.annotation_type == "keypoint" and hasattr(
            annotation, "keypoint"
        ):
            data = {"points": annotation.keypoint.points}
        return Annotation2DOutput(
            id=annotation.id,
            image_id=annotation.image_id,
            project_id=annotation.project_id,
            annotation_type=annotation.annotation_type,
            label=annotation.label,
            is_active=annotation.is_active,
            data=data,
            created_at=annotation.created_at,
            updated_at=annotation.updated_at,
        )


# ---------- Operation ----------


class OperationOutput(Schema):
    id: int
    image_id: int
    from_annotation_id: int | None
    to_annotation_id: int | None
    action: str
    source: str
    performed_by_id: int
    created_at: datetime

    @staticmethod
    def from_operation(op) -> "OperationOutput":
        return OperationOutput(
            id=op.id,
            image_id=op.image_id,
            from_annotation_id=op.from_annotation_id,
            to_annotation_id=op.to_annotation_id,
            action=op.action,
            source=op.source,
            performed_by_id=op.performed_by_id,
            created_at=op.created_at,
        )
