from django.conf import settings as django_settings
from django.db import transaction
from django.http import StreamingHttpResponse
from django.shortcuts import get_object_or_404
from ninja_extra import api_controller, http_get, http_post
from ninja_extra.permissions import AllowAny

from anno.pagination import PaginatedResponse, paginate_queryset
from anno_images.controllers import _presigned_redirect
from anno_images.models import Image2D

from .auth import ProjectAPIKeyAuth
from .schemas import (
    ProjectAnnotationBatchInput,
    ProjectAnnotationBatchOutput,
    ProjectAnnotationResultItem,
    ProjectImageOutput,
    ProjectMetaOutput,
)
from .services import create_ai_annotation


# ---------------------------------------------------------------------------
# Project inference endpoints (API-key auth; project implied by the key)
#
# The edge side authenticates with a per-project API key and accesses the
# project's resources: metadata (label mapping, etc.), images, and annotation
# submission.
# ---------------------------------------------------------------------------


@api_controller("/infers/project", tags=["infer-project"])
class ProjectInferController:

    # -- project meta --------------------------------------------------------

    @http_get(
        "/meta",
        permissions=[AllowAny],
        auth=ProjectAPIKeyAuth(),
        response={200: ProjectMetaOutput},
        url_name="infer_project_meta",
    )
    def project_meta(self, request):
        return 200, ProjectMetaOutput.from_project(request.project)

    # -- images --------------------------------------------------------------

    @http_get(
        "/images",
        permissions=[AllowAny],
        auth=ProjectAPIKeyAuth(),
        response={200: PaginatedResponse[ProjectImageOutput]},
        url_name="infer_project_image_list",
    )
    def list_images(
        self,
        request,
        limit: int = 100,
        offset: int = 0,
        has_active_annotations: bool | None = None,
    ):
        qs = Image2D.objects.filter(project=request.project)
        if has_active_annotations is True:
            qs = qs.filter(annotations__is_active=True).distinct()
        elif has_active_annotations is False:
            qs = qs.exclude(annotations__is_active=True)
        qs = qs.order_by("id")
        count, limit, offset, rows = paginate_queryset(qs, limit, offset)
        return 200, PaginatedResponse(
            count=count,
            limit=limit,
            offset=offset,
            items=[ProjectImageOutput.from_image(img) for img in rows],
        )

    @http_get(
        "/images/{image_id}",
        permissions=[AllowAny],
        auth=ProjectAPIKeyAuth(),
        response={200: ProjectImageOutput},
        url_name="infer_project_image_detail",
    )
    def image_detail(self, request, image_id: int):
        img = get_object_or_404(Image2D, id=image_id, project=request.project)
        return 200, ProjectImageOutput.from_image(img)

    @http_get(
        "/images/{image_id}/original_file",
        permissions=[AllowAny],
        auth=ProjectAPIKeyAuth(),
        url_name="infer_project_image_file",
    )
    def image_file(self, request, image_id: int):
        img = get_object_or_404(Image2D, id=image_id, project=request.project)
        if django_settings.DEBUG:
            return StreamingHttpResponse(img.image.open(), content_type="image/png")
        return _presigned_redirect(img.image.name, django_settings.IMAGE_PROXY_PREFIX)

    # -- annotations ---------------------------------------------------------

    @http_post(
        "/annotations",
        permissions=[AllowAny],
        auth=ProjectAPIKeyAuth(),
        response={200: ProjectAnnotationBatchOutput},
        url_name="infer_project_annotation_submit",
    )
    def submit_annotations(self, request, payload: ProjectAnnotationBatchInput):
        project = request.project
        performed_by = request.api_key.created_by

        # Resolve all referenced images up front, scoped to the key's project;
        # any id not in this map is rejected per-item below.
        wanted_ids = {item.image_id for item in payload.items}
        images = {
            img.id: img
            for img in Image2D.objects.filter(project=project, id__in=wanted_ids)
        }

        results: list[ProjectAnnotationResultItem] = []
        created = 0
        failed = 0
        for item in payload.items:
            try:
                img = images.get(item.image_id)
                if img is None:
                    raise ValueError("Image not found in this project.")
                # Per-item savepoint: one bad item rolls back only itself.
                with transaction.atomic():
                    annotation = create_ai_annotation(
                        image=img,
                        project=project,
                        annotation_type=item.annotation_type,
                        label=item.label,
                        polygon=item.polygon,
                        box=item.box,
                        keypoint=item.keypoint,
                        performed_by=performed_by,
                    )
                results.append(
                    ProjectAnnotationResultItem(
                        client_ref=item.client_ref,
                        image_id=item.image_id,
                        annotation_id=annotation.id,
                        status="created",
                    )
                )
                created += 1
            except Exception as exc:
                results.append(
                    ProjectAnnotationResultItem(
                        client_ref=item.client_ref,
                        image_id=item.image_id,
                        annotation_id=None,
                        status="error",
                        error=str(exc),
                    )
                )
                failed += 1

        return 200, ProjectAnnotationBatchOutput(
            created=created, failed=failed, results=results
        )
