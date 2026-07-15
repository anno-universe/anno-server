from datetime import datetime

from ninja import Field, Schema
from pydantic import field_validator

from anno_exports.models import ExportTask


class ExportCreateInput(Schema):
    format: str
    include_images: bool = False
    expires_at: datetime | None = Field(
        default=None,
        description="When the export file should be cleaned up. None = system default (24h). Pass null explicitly for no expiry.",
    )

    @field_validator("format")
    @classmethod
    def _check_format(cls, v: str) -> str:
        if v not in ("coco", "yolo"):
            raise ValueError("format must be 'coco' or 'yolo'.")
        return v


class ExportTaskResultOutput(Schema):
    file_name: str
    file_size: int
    file_available: bool
    file_deleted_at: datetime | None
    created_at: datetime

    @staticmethod
    def from_result(result) -> "ExportTaskResultOutput":
        if result is None:
            return None
        return ExportTaskResultOutput(
            file_name=result.file_name,
            file_size=result.file_size,
            file_available=result.file_available,
            file_deleted_at=result.file_deleted_at,
            created_at=result.created_at,
        )


class ExportTaskOutput(Schema):
    id: int
    project_id: int
    created_by_id: int
    format: str
    include_images: bool
    status: str
    expires_at: datetime | None
    error: str
    created_at: datetime
    started_at: datetime | None
    finished_at: datetime | None

    @staticmethod
    def from_task(task: "ExportTask") -> "ExportTaskOutput":
        return ExportTaskOutput(
            id=task.id,
            project_id=task.project_id,
            created_by_id=task.created_by_id,
            format=task.format,
            include_images=task.include_images,
            status=task.status,
            expires_at=task.expires_at,
            error=task.error,
            created_at=task.created_at,
            started_at=task.started_at,
            finished_at=task.finished_at,
        )


class ExportTaskDetailOutput(ExportTaskOutput):
    result: ExportTaskResultOutput | None = None

    @staticmethod
    def from_task_with_result(task: "ExportTask") -> "ExportTaskDetailOutput":
        base = ExportTaskOutput.from_task(task)
        result = None
        if hasattr(task, "result") and task.result is not None:
            result = ExportTaskResultOutput.from_result(task.result)
        return ExportTaskDetailOutput(
            **base.dict(),
            result=result,
        )
