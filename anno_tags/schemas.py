from datetime import datetime

from ninja import Schema


# ---------- ProjectTag ----------


class TagCreateInput(Schema):
    name: str
    display_name: str
    color: str = "#6366F1"
    description: str = ""


class TagUpdateInput(Schema):
    display_name: str | None = None
    color: str | None = None
    description: str | None = None
    is_active: bool | None = None


class TagOutput(Schema):
    id: int
    project_id: int
    name: str
    display_name: str
    color: str
    description: str
    is_active: bool
    created_by_id: int
    created_at: datetime
    updated_at: datetime

    @staticmethod
    def from_tag(tag) -> "TagOutput":
        return TagOutput(
            id=tag.id,
            project_id=tag.project_id,
            name=tag.name,
            display_name=tag.display_name,
            color=tag.color,
            description=tag.description,
            is_active=tag.is_active,
            created_by_id=tag.created_by_id,
            created_at=tag.created_at,
            updated_at=tag.updated_at,
        )


class TagStatItem(Schema):
    tag_id: int
    name: str
    display_name: str
    color: str
    image_count: int


class TagStatsOutput(Schema):
    project_id: int
    tags: list[TagStatItem]


# ---------- ImageTag ----------


class ImageTagApplyInput(Schema):
    tag_id: int
    note: str = ""


class ImageTagOutput(Schema):
    id: int
    image_id: int
    tag_id: int
    tag_name: str
    tag_display_name: str
    tag_color: str
    applied_by_id: int
    note: str
    created_at: datetime

    @staticmethod
    def from_image_tag(it) -> "ImageTagOutput":
        return ImageTagOutput(
            id=it.id,
            image_id=it.image_id,
            tag_id=it.tag_id,
            tag_name=it.tag.name,
            tag_display_name=it.tag.display_name,
            tag_color=it.tag.color,
            applied_by_id=it.applied_by_id,
            note=it.note,
            created_at=it.created_at,
        )
