import logging
import os
from datetime import timedelta

from django.conf import settings as django_settings
from django.db import transaction
from django.http import FileResponse, Http404
from django.shortcuts import get_object_or_404
from django.utils import timezone

logger = logging.getLogger(__name__)
from ninja_extra import api_controller, http_delete, http_get, http_post
from ninja_extra.permissions import IsAuthenticated
from ninja_jwt.authentication import JWTAuth

from anno.pagination import PaginatedResponse, paginate_queryset
from anno_projects.models import Project
from anno_projects.permissions import IsProjectMemberOrAdmin, IsProjectSupervisorOrAdmin

from .models import ExportTask
from .schemas import ExportCreateInput, ExportTaskDetailOutput, ExportTaskOutput
from .tasks import run_export


@api_controller("/projects/{project_id}/exports", tags=["exports"])
class ExportController:

    @http_post(
        "/",
        permissions=[IsAuthenticated, IsProjectSupervisorOrAdmin],
        auth=JWTAuth(),
        response={201: ExportTaskOutput},
        url_name="export_create",
    )
    def create(self, request, project_id: int, payload: ExportCreateInput):
        project = get_object_or_404(Project, id=project_id)

        payload_dict = payload.dict(exclude_unset=True)
        if "expires_at" not in payload_dict:
            expires_at = timezone.now() + timedelta(
                hours=getattr(django_settings, "EXPORT_DEFAULT_RETENTION_HOURS", 24)
            )
        else:
            expires_at = payload_dict["expires_at"]

        with transaction.atomic():
            task = ExportTask.objects.create(
                project=project,
                created_by=request.user,
                format=payload.format,
                include_images=payload.include_images,
                expires_at=expires_at,
            )
            transaction.on_commit(lambda: run_export.enqueue(task.id))

        logger.info(
            "Export task %d created and enqueued: project=%s format=%s include_images=%s expires=%s",
            task.id, project.name, task.format, task.include_images, task.expires_at,
        )
        return 201, ExportTaskOutput.from_task(task)

    @http_get(
        "/",
        permissions=[IsAuthenticated, IsProjectMemberOrAdmin],
        auth=JWTAuth(),
        response={200: PaginatedResponse[ExportTaskOutput]},
        url_name="export_list",
    )
    def list_tasks(self, request, project_id: int, limit: int = 100, offset: int = 0):
        qs = ExportTask.objects.filter(project_id=project_id).order_by("-created_at")
        count, limit, offset, rows = paginate_queryset(qs, limit, offset)
        return 200, PaginatedResponse(
            count=count,
            limit=limit,
            offset=offset,
            items=[ExportTaskOutput.from_task(t) for t in rows],
        )

    @http_get(
        "/{task_id}",
        permissions=[IsAuthenticated, IsProjectMemberOrAdmin],
        auth=JWTAuth(),
        response={200: ExportTaskDetailOutput},
        url_name="export_detail",
    )
    def detail(self, request, project_id: int, task_id: int):
        task = get_object_or_404(
            ExportTask.objects.select_related("result"),
            id=task_id,
            project_id=project_id,
        )
        return 200, ExportTaskDetailOutput.from_task_with_result(task)

    @http_get(
        "/{task_id}/download",
        permissions=[IsAuthenticated, IsProjectSupervisorOrAdmin],
        auth=JWTAuth(),
        url_name="export_download",
    )
    def download(self, request, project_id: int, task_id: int):
        task = get_object_or_404(
            ExportTask.objects.select_related("result"),
            id=task_id,
            project_id=project_id,
        )
        if task.status != ExportTask.STATUS_COMPLETED:
            raise Http404("Export not completed.")
        if not hasattr(task, "result") or task.result is None:
            raise Http404("No export file available.")
        if not task.result.file_available:
            raise Http404("Export file has been deleted or expired.")

        file_path = task.result.export_file.path
        if not os.path.isfile(file_path):
            raise Http404("Export file not found on disk.")

        return FileResponse(
            open(file_path, "rb"),
            as_attachment=True,
            filename=task.result.file_name,
        )

    @http_delete(
        "/{task_id}/file",
        permissions=[IsAuthenticated, IsProjectSupervisorOrAdmin],
        auth=JWTAuth(),
        response={204: None},
        url_name="export_delete_file",
    )
    def delete_file(self, request, project_id: int, task_id: int):
        task = get_object_or_404(
            ExportTask.objects.select_related("result"),
            id=task_id,
            project_id=project_id,
        )
        if not hasattr(task, "result") or task.result is None:
            raise Http404("No export file to delete.")
        if not task.result.export_file:
            raise Http404("Export file already deleted.")

        result = task.result
        result.export_file.delete(save=False)
        result.export_file = None
        result.file_deleted_at = timezone.now()
        result.save(update_fields=["export_file", "file_deleted_at"])

        logger.info("Export file for task %d manually deleted by user %d", task_id, request.user.id)
        return 204, None
