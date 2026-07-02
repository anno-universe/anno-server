import logging
from datetime import timedelta

from django.conf import settings as django_settings
from django.db import transaction
from django.db.models import Q
from django.http import StreamingHttpResponse
from django.shortcuts import get_object_or_404
from django.utils import timezone

logger = logging.getLogger(__name__)
from ninja_extra import api_controller, http_delete, http_get, http_patch, http_post
from ninja_extra.exceptions import HttpError
from ninja_extra.permissions import AllowAny, IsAuthenticated
from ninja_jwt.authentication import JWTAuth

from anno.pagination import PaginatedResponse, paginate_queryset
from anno_images.controllers import _presigned_redirect
from anno_images.models import Annotation2D, Image2D
from anno_projects.models import Project
from anno_projects.permissions import IsProjectMemberOrAdmin, IsProjectSupervisorOrAdmin

from .auth import ProjectAPIKeyAuth
from .models import (
    InferenceRun,
    InferenceServiceProvider,
    InferenceTask,
    InteractiveInferenceServiceProvider,
    InteractiveInferenceSession,
)
from .schemas import (
    AutoAnnotateInput,
    ImageAutoAnnotateInput,
    InteractiveCommitInput,
    InteractiveProviderCreateInput,
    InteractiveProviderOutput,
    InteractiveProviderUpdateInput,
    InteractiveSessionDetailOutput,
    InteractiveSessionOutput,
    InteractiveSessionStartInput,
    InteractiveStepInput,
    InteractiveStepOutput,
    ProjectAnnotationBatchOutput,
    ProjectAnnotationModifyOutput,
    ProjectAnnotationResultItem,
    ProjectImageAnnotationBatchInput,
    ProjectImageAnnotationInput,
    ProjectImageOutput,
    ProjectMetaOutput,
    ProviderCreateInput,
    ProviderOutput,
    ProviderUpdateInput,
    RunDetailOutput,
    RunOutput,
    TaskDetailOutput,
)
from .services import (
    commit_interactive_session,
    create_ai_annotation,
    discard_interactive_session,
    modify_ai_annotation,
    provider_snapshot,
    run_interactive_step,
    start_interactive_session,
)
from .tasks import run_inference_run

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
        "/images/{image_id}/annotations",
        permissions=[AllowAny],
        auth=ProjectAPIKeyAuth(),
        response={200: ProjectAnnotationBatchOutput},
        url_name="infer_project_image_annotation_submit",
    )
    def submit_image_annotations(
        self,
        request,
        image_id: int,
        payload: ProjectImageAnnotationBatchInput,
    ):
        """Submit AI annotations for a single image.

        Each annotation in the list is processed independently;
        one failure does not affect the others."""
        project = request.project
        performed_by = request.api_key.created_by

        image = get_object_or_404(Image2D, project=project, id=image_id)

        results: list[ProjectAnnotationResultItem] = []
        created = 0
        failed = 0
        for item in payload.annotations:
            client_ref = item.client_ref
            try:
                with transaction.atomic():
                    annotation = create_ai_annotation(
                        image=image,
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
                        client_ref=client_ref,
                        image_id=image_id,
                        annotation_id=annotation.id,
                        status="created",
                    )
                )
                created += 1
            except Exception as exc:
                results.append(
                    ProjectAnnotationResultItem(
                        client_ref=client_ref,
                        image_id=image_id,
                        annotation_id=None,
                        status="error",
                        error=str(exc),
                    )
                )
                failed += 1

        return 200, ProjectAnnotationBatchOutput(
            created=created, failed=failed, results=results
        )

    @http_patch(
        "/images/{image_id}/annotations/{annotation_id}",
        permissions=[AllowAny],
        auth=ProjectAPIKeyAuth(),
        response={200: ProjectAnnotationModifyOutput},
        url_name="infer_project_image_annotation_modify",
    )
    @transaction.atomic
    def modify_image_annotation(
        self,
        request,
        image_id: int,
        annotation_id: int,
        payload: ProjectImageAnnotationInput,
    ):
        """Modify an existing AI annotation (immutable pattern).

        Creates a new annotation with the updated data and deactivates
        the old one. Supports SAM and other refinement models."""
        project = request.project
        performed_by = request.api_key.created_by

        old = get_object_or_404(
            Annotation2D.objects.select_related("polygon", "box", "keypoint"),
            id=annotation_id,
            image_id=image_id,
            project=project,
            is_active=True,
        )

        new = modify_ai_annotation(
            old_annotation=old,
            annotation_type=payload.annotation_type,
            label=payload.label,
            polygon=payload.polygon,
            box=payload.box,
            keypoint=payload.keypoint,
            performed_by=performed_by,
        )

        return 200, ProjectAnnotationModifyOutput.from_annotation(new)


# ---------------------------------------------------------------------------
# Inference service provider registry (JWT; Flow B)
#
# Providers are either global (project=null, admin-managed via Django admin and
# read-only here) or project-scoped (created/edited by supervisors). Members can
# list what's usable; only supervisors can mutate project-scoped providers.
# ---------------------------------------------------------------------------


def _visible_providers(project_id: int):
    """Providers usable by a project: its own plus global ones."""
    return InferenceServiceProvider.objects.filter(
        Q(project_id=project_id) | Q(project__isnull=True)
    )


@api_controller("/projects/{project_id}/inference-providers", tags=["infer-providers"])
class InferenceProviderController:

    @http_get(
        "/",
        permissions=[IsAuthenticated, IsProjectMemberOrAdmin],
        auth=JWTAuth(),
        response={200: PaginatedResponse[ProviderOutput]},
        url_name="infer_provider_list",
    )
    def list_providers(
        self,
        request,
        project_id: int,
        limit: int = 100,
        offset: int = 0,
        is_active: bool | None = None,
    ):
        qs = _visible_providers(project_id)
        if is_active is not None:
            qs = qs.filter(is_active=is_active)
        qs = qs.order_by("-created_at")
        count, limit, offset, rows = paginate_queryset(qs, limit, offset)
        return 200, PaginatedResponse(
            count=count,
            limit=limit,
            offset=offset,
            items=[ProviderOutput.from_provider(p) for p in rows],
        )

    @http_post(
        "/",
        permissions=[IsAuthenticated, IsProjectSupervisorOrAdmin],
        auth=JWTAuth(),
        response={201: ProviderOutput},
        url_name="infer_provider_create",
    )
    def create(self, request, project_id: int, payload: ProviderCreateInput):
        project = get_object_or_404(Project, id=project_id)
        provider = InferenceServiceProvider.objects.create(
            project=project,
            created_by=request.user,
            **payload.dict(),
        )
        return 201, ProviderOutput.from_provider(provider)

    @http_get(
        "/{provider_id}",
        permissions=[IsAuthenticated, IsProjectSupervisorOrAdmin],
        auth=JWTAuth(),
        response={200: ProviderOutput},
        url_name="infer_provider_detail",
    )
    def detail(self, request, project_id: int, provider_id: int):
        provider = get_object_or_404(_visible_providers(project_id), id=provider_id)
        return 200, ProviderOutput.from_provider(provider)

    @http_patch(
        "/{provider_id}",
        permissions=[IsAuthenticated, IsProjectSupervisorOrAdmin],
        auth=JWTAuth(),
        response={200: ProviderOutput},
        url_name="infer_provider_update",
    )
    def update(
        self, request, project_id: int, provider_id: int, payload: ProviderUpdateInput
    ):
        # Only project-scoped providers are editable here; globals are admin-managed.
        provider = get_object_or_404(
            InferenceServiceProvider, id=provider_id, project_id=project_id
        )
        for attr, value in payload.dict(exclude_unset=True).items():
            setattr(provider, attr, value)
        provider.save()
        return 200, ProviderOutput.from_provider(provider)

    @http_delete(
        "/{provider_id}",
        permissions=[IsAuthenticated, IsProjectSupervisorOrAdmin],
        auth=JWTAuth(),
        response={204: None},
        url_name="infer_provider_delete",
    )
    def delete(self, request, project_id: int, provider_id: int):
        provider = get_object_or_404(
            InferenceServiceProvider, id=provider_id, project_id=project_id
        )
        provider.delete()
        return 204, None


# ---------------------------------------------------------------------------
# Auto-annotation jobs (JWT, supervisor-triggered; Flow B)
# ---------------------------------------------------------------------------


@api_controller("/projects/{project_id}/auto-annotate", tags=["infer-auto"])
class AutoAnnotateController:

    @http_post(
        "/",
        permissions=[IsAuthenticated, IsProjectSupervisorOrAdmin],
        auth=JWTAuth(),
        response={201: RunOutput},
        url_name="infer_auto_annotate",
    )
    def start(self, request, project_id: int, payload: AutoAnnotateInput):
        project = get_object_or_404(Project, id=project_id)

        provider = (
            _visible_providers(project_id)
            .filter(id=payload.provider_id, is_active=True)
            .first()
        )
        if provider is None:
            raise HttpError(404, "Active provider not found for this project.")

        # Target ALL images in the project.
        image_ids = list(
            Image2D.objects.filter(project=project)
            .order_by("id")
            .values_list("id", flat=True)
        )
        if not image_ids:
            raise HttpError(400, "No images in project.")

        # Bound the whole run: time for each image plus a small buffer.
        deadline = timezone.now() + timedelta(
            seconds=(len(image_ids) + 1) * provider.timeout_seconds
        )

        logger.info(
            "Creating auto-annotation run: project_id=%d provider_id=%d image_count=%d",
            project.id,
            provider.id,
            len(image_ids),
        )
        with transaction.atomic():
            run = InferenceRun.objects.create(
                project=project,
                provider=provider,
                created_by=request.user,
                total_items=len(image_ids),
                deadline=deadline,
                provider_snapshot=provider_snapshot(provider),
            )
            InferenceTask.objects.bulk_create(
                [InferenceTask(run=run, image_id=iid) for iid in image_ids]
            )
            transaction.on_commit(lambda: run_inference_run.enqueue(run.id))

        logger.info(
            "Auto-annotation run %d created and task enqueued: %d items, deadline=%s",
            run.id,
            len(image_ids),
            deadline.isoformat(),
        )
        return 201, RunOutput.from_run(run)

    @http_get(
        "/runs",
        permissions=[IsAuthenticated, IsProjectMemberOrAdmin],
        auth=JWTAuth(),
        response={200: PaginatedResponse[RunOutput]},
        url_name="infer_auto_run_list",
    )
    def list_runs(self, request, project_id: int, limit: int = 100, offset: int = 0):
        qs = InferenceRun.objects.filter(project_id=project_id).order_by("-created_at")
        count, limit, offset, rows = paginate_queryset(qs, limit, offset)
        return 200, PaginatedResponse(
            count=count,
            limit=limit,
            offset=offset,
            items=[RunOutput.from_run(r) for r in rows],
        )

    @http_get(
        "/runs/{run_id}",
        permissions=[IsAuthenticated, IsProjectMemberOrAdmin],
        auth=JWTAuth(),
        response={200: RunDetailOutput},
        url_name="infer_auto_run_detail",
    )
    def run_detail(self, request, project_id: int, run_id: int):
        run = get_object_or_404(
            InferenceRun.objects.prefetch_related("tasks"),
            id=run_id,
            project_id=project_id,
        )
        return 200, RunDetailOutput.from_run_with_tasks(run)

    @http_get(
        "/runs/{run_id}/tasks/{task_id}",
        permissions=[IsAuthenticated, IsProjectMemberOrAdmin],
        auth=JWTAuth(),
        response={200: TaskDetailOutput},
        url_name="infer_auto_task_detail",
    )
    def task_detail(self, request, project_id: int, run_id: int, task_id: int):
        """Single-image ``InferenceTask`` detail with candidate results."""
        task = get_object_or_404(
            InferenceTask.objects.prefetch_related("results"),
            id=task_id,
            run_id=run_id,
            run__project_id=project_id,
        )
        return 200, TaskDetailOutput.from_task_with_results(task)

    @http_post(
        "/runs/{run_id}/cancel",
        permissions=[IsAuthenticated, IsProjectSupervisorOrAdmin],
        auth=JWTAuth(),
        response={200: RunOutput},
        url_name="infer_auto_run_cancel",
    )
    def cancel(self, request, project_id: int, run_id: int):
        run = get_object_or_404(InferenceRun, id=run_id, project_id=project_id)
        run.cancel_requested = True
        if run.status in (InferenceRun.STATUS_PENDING, InferenceRun.STATUS_RUNNING):
            run.status = InferenceRun.STATUS_CANCELLING
        run.save(update_fields=["cancel_requested", "status"])
        return 200, RunOutput.from_run(run)

    @http_post(
        "/runs/{run_id}/retry",
        permissions=[IsAuthenticated, IsProjectSupervisorOrAdmin],
        auth=JWTAuth(),
        response={200: RunOutput},
        url_name="infer_auto_run_retry",
    )
    def retry(self, request, project_id: int, run_id: int):
        run = get_object_or_404(InferenceRun, id=run_id, project_id=project_id)

        with transaction.atomic():
            reset = run.tasks.filter(
                status__in=[
                    InferenceTask.STATUS_FAILED,
                    InferenceTask.STATUS_SKIPPED,
                ]
            ).update(status=InferenceTask.STATUS_PENDING)
            run.failed_items = 0
            run.cancel_requested = False
            run.status = InferenceRun.STATUS_PENDING
            run.error = ""
            # Refresh the deadline relative to the work remaining.
            run.deadline = timezone.now() + timedelta(
                seconds=(reset + 1) * run.provider.timeout_seconds
            )
            run.save(
                update_fields=[
                    "failed_items",
                    "cancel_requested",
                    "status",
                    "error",
                    "deadline",
                ]
            )
            transaction.on_commit(lambda: run_inference_run.enqueue(run.id))

        return 200, RunOutput.from_run(run)


# ---------------------------------------------------------------------------
# Single-image auto-annotation (JWT, supervisor-triggered; Flow B)
# ---------------------------------------------------------------------------


@api_controller(
    "/projects/{project_id}/images/{image_id}/auto-annotate", tags=["infer-auto"]
)
class ImageAutoAnnotateController:
    """Trigger server-driven inference for a single image asynchronously.

    Creates an ``InferenceRun`` with a single ``InferenceTask`` and enqueues it
    for the background worker, then returns immediately. Use
    ``GET /runs/{run_id}`` to track progress and results.
    """

    @http_post(
        "/",
        permissions=[IsAuthenticated, IsProjectMemberOrAdmin],
        auth=JWTAuth(),
        response={201: RunOutput},
        url_name="infer_image_auto_annotate",
    )
    def auto_annotate_image(
        self,
        request,
        project_id: int,
        image_id: int,
        payload: ImageAutoAnnotateInput,
    ):
        project = get_object_or_404(Project, id=project_id)
        image = get_object_or_404(Image2D, id=image_id, project=project)

        provider = (
            _visible_providers(project_id)
            .filter(id=payload.provider_id, is_active=True)
            .first()
        )
        if provider is None:
            raise HttpError(404, "Active provider not found for this project.")

        # Bound the single-image run: one image plus a small buffer.
        deadline = timezone.now() + timedelta(seconds=2 * provider.timeout_seconds)

        logger.info(
            "Creating single-image inference run: project_id=%d image_id=%d provider_id=%d",
            project.id,
            image.id,
            provider.id,
        )
        with transaction.atomic():
            run = InferenceRun.objects.create(
                project=project,
                provider=provider,
                created_by=request.user,
                total_items=1,
                deadline=deadline,
                provider_snapshot=provider_snapshot(provider),
            )
            InferenceTask.objects.create(run=run, image=image)
            transaction.on_commit(lambda: run_inference_run.enqueue(run.id))

        logger.info(
            "Single-image inference run %d created and task enqueued: image_id=%d",
            run.id,
            image.id,
        )
        return 201, RunOutput.from_run(run)

    @http_get(
        "/runs/{run_id}",
        permissions=[IsAuthenticated, IsProjectMemberOrAdmin],
        auth=JWTAuth(),
        response={200: RunDetailOutput},
        url_name="infer_image_run_detail",
    )
    def run_detail(self, request, project_id: int, image_id: int, run_id: int):
        run = get_object_or_404(
            InferenceRun.objects.prefetch_related("tasks"),
            id=run_id,
            project_id=project_id,
            tasks__image_id=image_id,
        )
        return 200, RunDetailOutput.from_run_with_tasks(run)


# ---------------------------------------------------------------------------
# Interactive inference provider registry (JWT)
#
# Same scoping model as InferenceProviderController: global (project=null,
# admin-managed) or project-scoped (supervisor-managed). Members can list.
# ---------------------------------------------------------------------------


def _visible_interactive_providers(project_id: int):
    """Interactive providers usable by a project: its own plus global ones."""
    return InteractiveInferenceServiceProvider.objects.filter(
        Q(project_id=project_id) | Q(project__isnull=True)
    )


@api_controller(
    "/projects/{project_id}/interactive-providers", tags=["interactive-providers"]
)
class InteractiveProviderController:

    @http_get(
        "/",
        permissions=[IsAuthenticated, IsProjectMemberOrAdmin],
        auth=JWTAuth(),
        response={200: PaginatedResponse[InteractiveProviderOutput]},
        url_name="interactive_provider_list",
    )
    def list_providers(
        self,
        request,
        project_id: int,
        limit: int = 100,
        offset: int = 0,
        is_active: bool | None = None,
    ):
        qs = _visible_interactive_providers(project_id)
        if is_active is not None:
            qs = qs.filter(is_active=is_active)
        qs = qs.order_by("-created_at")
        count, limit, offset, rows = paginate_queryset(qs, limit, offset)
        return 200, PaginatedResponse(
            count=count,
            limit=limit,
            offset=offset,
            items=[InteractiveProviderOutput.from_provider(p) for p in rows],
        )

    @http_post(
        "/",
        permissions=[IsAuthenticated, IsProjectSupervisorOrAdmin],
        auth=JWTAuth(),
        response={201: InteractiveProviderOutput},
        url_name="interactive_provider_create",
    )
    def create(self, request, project_id: int, payload: InteractiveProviderCreateInput):
        project = get_object_or_404(Project, id=project_id)
        provider = InteractiveInferenceServiceProvider.objects.create(
            project=project,
            created_by=request.user,
            **payload.dict(),
        )
        return 201, InteractiveProviderOutput.from_provider(provider)

    @http_get(
        "/{provider_id}",
        permissions=[IsAuthenticated, IsProjectSupervisorOrAdmin],
        auth=JWTAuth(),
        response={200: InteractiveProviderOutput},
        url_name="interactive_provider_detail",
    )
    def detail(self, request, project_id: int, provider_id: int):
        provider = get_object_or_404(
            _visible_interactive_providers(project_id), id=provider_id
        )
        return 200, InteractiveProviderOutput.from_provider(provider)

    @http_patch(
        "/{provider_id}",
        permissions=[IsAuthenticated, IsProjectSupervisorOrAdmin],
        auth=JWTAuth(),
        response={200: InteractiveProviderOutput},
        url_name="interactive_provider_update",
    )
    def update(
        self, request, project_id: int, provider_id: int, payload: InteractiveProviderUpdateInput
    ):
        # Only project-scoped providers are editable here; globals are admin-managed.
        provider = get_object_or_404(
            InteractiveInferenceServiceProvider, id=provider_id, project_id=project_id
        )
        for attr, value in payload.dict(exclude_unset=True).items():
            setattr(provider, attr, value)
        provider.save()
        return 200, InteractiveProviderOutput.from_provider(provider)

    @http_delete(
        "/{provider_id}",
        permissions=[IsAuthenticated, IsProjectSupervisorOrAdmin],
        auth=JWTAuth(),
        response={204: None},
        url_name="interactive_provider_delete",
    )
    def delete(self, request, project_id: int, provider_id: int):
        provider = get_object_or_404(
            InteractiveInferenceServiceProvider, id=provider_id, project_id=project_id
        )
        provider.delete()
        return 204, None


# ---------------------------------------------------------------------------
# Interactive inference sessions (JWT, member-level)
#
# A user opens a session on an image, submits prompt steps (each a synchronous
# provider call returning a candidate), and finally commits a chosen step's
# candidate as a real Annotation2D (Operation.source="interactive") or discards.
# ---------------------------------------------------------------------------


@api_controller(
    "/projects/{project_id}/images/{image_id}/interactive-sessions",
    tags=["interactive-sessions"],
)
class InteractiveSessionController:

    def _get_session(self, project_id, image_id, session_id, *, editing=False):
        qs = InteractiveInferenceSession.objects.select_related(
            "project", "image", "provider", "from_annotation"
        )
        session = get_object_or_404(
            qs, id=session_id, project_id=project_id, image_id=image_id
        )
        if editing and session.status != InteractiveInferenceSession.STATUS_EDITING:
            raise HttpError(409, f"Session is {session.status}, not editing.")
        return session

    @http_post(
        "/",
        permissions=[IsAuthenticated, IsProjectMemberOrAdmin],
        auth=JWTAuth(),
        response={201: InteractiveSessionOutput},
        url_name="interactive_session_start",
    )
    def start(self, request, project_id: int, image_id: int, payload: InteractiveSessionStartInput):
        project = get_object_or_404(Project, id=project_id)
        image = get_object_or_404(Image2D, id=image_id, project=project)

        provider = (
            _visible_interactive_providers(project_id)
            .filter(id=payload.provider_id, is_active=True)
            .first()
        )
        if provider is None:
            raise HttpError(404, "Active interactive provider not found for this project.")

        from_annotation = None
        if payload.from_annotation_id is not None:
            from_annotation = get_object_or_404(
                Annotation2D,
                id=payload.from_annotation_id,
                image=image,
                project=project,
                is_active=True,
            )

        session = start_interactive_session(
            project=project,
            image=image,
            provider=provider,
            performed_by=request.user,
            from_annotation=from_annotation,
        )
        return 201, InteractiveSessionOutput.from_session(session)

    @http_post(
        "/{session_id}/steps",
        permissions=[IsAuthenticated, IsProjectMemberOrAdmin],
        auth=JWTAuth(),
        response={201: InteractiveStepOutput},
        url_name="interactive_session_step",
    )
    def add_step(
        self, request, project_id: int, image_id: int, session_id: int, payload: InteractiveStepInput
    ):
        session = self._get_session(project_id, image_id, session_id, editing=True)
        try:
            operation = run_interactive_step(session, payload.prompts)
        except ValueError as exc:
            raise HttpError(422, str(exc))
        except Exception as exc:
            # Provider call failed; the step was recorded with its error.
            raise HttpError(502, f"Interactive provider call failed: {exc}")
        return 201, InteractiveStepOutput.from_operation(operation)

    @http_get(
        "/{session_id}",
        permissions=[IsAuthenticated, IsProjectMemberOrAdmin],
        auth=JWTAuth(),
        response={200: InteractiveSessionDetailOutput},
        url_name="interactive_session_detail",
    )
    def detail(self, request, project_id: int, image_id: int, session_id: int):
        session = get_object_or_404(
            InteractiveInferenceSession.objects.prefetch_related("operations"),
            id=session_id,
            project_id=project_id,
            image_id=image_id,
        )
        return 200, InteractiveSessionDetailOutput.from_session_with_steps(session)

    @http_post(
        "/{session_id}/commit",
        permissions=[IsAuthenticated, IsProjectMemberOrAdmin],
        auth=JWTAuth(),
        response={200: InteractiveSessionOutput},
        url_name="interactive_session_commit",
    )
    def commit(
        self, request, project_id: int, image_id: int, session_id: int, payload: InteractiveCommitInput
    ):
        session = self._get_session(project_id, image_id, session_id, editing=True)
        operation = get_object_or_404(
            session.operations, id=payload.step_id
        )
        try:
            commit_interactive_session(session, operation)
        except ValueError as exc:
            raise HttpError(422, str(exc))
        return 200, InteractiveSessionOutput.from_session(session)

    @http_post(
        "/{session_id}/discard",
        permissions=[IsAuthenticated, IsProjectMemberOrAdmin],
        auth=JWTAuth(),
        response={200: InteractiveSessionOutput},
        url_name="interactive_session_discard",
    )
    def discard(self, request, project_id: int, image_id: int, session_id: int):
        session = self._get_session(project_id, image_id, session_id, editing=True)
        discard_interactive_session(session)
        return 200, InteractiveSessionOutput.from_session(session)
