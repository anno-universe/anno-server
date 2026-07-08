"""Background worker for server-driven auto-annotation (Flow B).

``run_inference_run`` is enqueued by the auto-annotate endpoints and executed by
the ``db_worker`` process. For each pending image it sends the raw image bytes
plus a JSON metadata block to the provider's inference URL, parses the response
with the shared anno-sdk contract, validates the returned geometry against the
provider's declared result types, and writes annotations through the existing
``create_ai_annotation`` write-path (which records an ``Operation``).

Persistence, cooperative cancellation, the whole-run deadline and per-task
replay all live in the run/task rows, so this worker is just a driver. A
single-image inference is simply a run with one task and needs no special path.
"""

from __future__ import annotations

import json
import logging
import mimetypes
import os
from datetime import timedelta

import httpx
from django.db import transaction
from django.db.models import F
from django.utils import timezone
from django_tasks import task

from anno_images.models import Operation
from anno_images.schemas import Box2DDataInput, Keypoint2DDataInput, Polygon2DDataInput
from anno_sdk import InferenceRequestMeta, InferenceResponse

from .models import InferenceResult, InferenceRun, InferenceTask
from .services import create_ai_annotation

logger = logging.getLogger(__name__)


def _annotation_to_kwargs(annotation) -> dict:
    """Map an anno-sdk ``Annotation`` to ``create_ai_annotation`` kwargs.

    The SDK ``Box2D`` has no ``rotation`` attribute (it is implied 0.0); the
    ``*DataInput`` schemas supply the default, so we route through them.
    """
    geo = annotation.geometry
    atype = geo.annotation_type
    kwargs: dict = {
        "annotation_type": atype,
        "label": annotation.label,
        "polygon": None,
        "box": None,
        "keypoint": None,
    }
    if atype == "polygon":
        kwargs["polygon"] = Polygon2DDataInput(points=geo.points)
    elif atype == "keypoint":
        kwargs["keypoint"] = Keypoint2DDataInput(points=geo.points)
    elif atype == "box":
        rotation = getattr(geo, "rotation", 0.0) or 0.0
        kwargs["box"] = Box2DDataInput(
            x=geo.x, y=geo.y, width=geo.width, height=geo.height, rotation=rotation
        )
    else:  # pragma: no cover - guarded by supported-type filtering upstream
        raise ValueError(f"Unsupported annotation_type: {atype!r}")
    return kwargs


def _build_auth(provider) -> tuple[dict, dict]:
    """Return ``(headers, params)`` carrying the provider credential."""
    headers: dict = {}
    params: dict = {}
    if provider.auth_type == provider.AUTH_HEADER and provider.auth_param_name:
        headers[provider.auth_param_name] = provider.auth_secret
    elif provider.auth_type == provider.AUTH_QUERY and provider.auth_param_name:
        params[provider.auth_param_name] = provider.auth_secret
    return headers, params


def _call_provider(provider, image_bytes: bytes, file_name: str, meta: InferenceRequestMeta):
    """POST image bytes + metadata to the provider; return an InferenceResponse."""
    headers, params = _build_auth(provider)
    content_type = mimetypes.guess_type(file_name or "")[0] or "application/octet-stream"
    files = {"image": (file_name or "image", image_bytes, content_type)}
    data = {"metadata": json.dumps(meta.to_dict())}

    logger.debug(
        "Calling provider %s at %s: image_id=%s task_id=%s timeout=%ds size=%d",
        provider.name,
        provider.inference_url,
        meta.image_id,
        meta.task_id,
        provider.timeout_seconds,
        len(image_bytes),
    )
    resp = httpx.post(
        provider.inference_url.rstrip("/") + "/predict",
        files=files,
        data=data,
        headers=headers,
        params=params,
        timeout=provider.timeout_seconds,
    )
    resp.raise_for_status()
    response = InferenceResponse.from_dict(resp.json())
    logger.debug(
        "Provider %s returned %d annotations for image_id=%s",
        provider.name,
        len(response.annotations),
        meta.image_id,
    )
    return response


def _process_task(task: InferenceTask, *, provider, project, performed_by) -> None:
    """Run one image through the provider and persist its annotations.

    ``provider``, ``project`` and ``performed_by`` are passed explicitly by the
    caller (all carried on the run). For each returned candidate an
    ``InferenceResult`` is recorded and then auto-committed. Raises on any
    failure so the caller can mark the task failed; all DB writes for the task
    are wrapped in a single atomic block.
    """
    image = task.image

    logger.info(
        "Processing task %d: image_id=%d file=%s (attempt %d)",
        task.id,
        image.id,
        image.file_name,
        task.attempts + 1,
    )

    with image.image.open("rb") as fh:
        image_bytes = fh.read()
    file_name = os.path.basename(image.image.name or "") or image.file_name

    meta = InferenceRequestMeta(
        image_id=image.id,
        task_id=task.id,
        label_mapping=project.label_mapping,
        requested_types=list(provider.supported_result_types),
        width=image.width,
        height=image.height,
    )

    response = _call_provider(provider, image_bytes, file_name, meta)

    supported = set(provider.supported_result_types)
    created = 0
    with transaction.atomic():
        for idx, ann in enumerate(response.annotations):
            result_type = ann.geometry.annotation_type
            if result_type not in supported:
                logger.warning(
                    "Task %d: provider returned unsupported type %r for image_id=%d, skipping",
                    task.id,
                    result_type,
                    image.id,
                )
                continue

            result = InferenceResult.objects.create(
                task=task,
                result_index=idx,
                result_type=result_type,
                label=ann.label,
                result_data=ann.geometry.to_dict(),
                raw_result=ann.to_dict(),
                status=InferenceResult.STATUS_PROPOSED,
            )
            annotation = create_ai_annotation(
                image=image,
                project=project,
                performed_by=performed_by,
                source=Operation.SOURCE_INFERENCE,
                **_annotation_to_kwargs(ann),
            )
            result.annotation = annotation
            result.status = InferenceResult.STATUS_COMMITTED
            result.committed_at = timezone.now()
            result.save(update_fields=["annotation", "status", "committed_at"])
            created += 1

    task.annotations_created = created
    task.status = InferenceTask.STATUS_DONE
    task.error = ""
    task.finished_at = timezone.now()
    task.save(update_fields=["annotations_created", "status", "error", "finished_at"])

    logger.info(
        "Task %d done: image_id=%d annotations_created=%d",
        task.id,
        image.id,
        created,
    )


@task()
def run_inference_run(run_id: int) -> None:
    """Task entrypoint enqueued by the auto-annotate endpoints / db_worker."""
    execute_inference_run(run_id)


def execute_inference_run(run_id: int) -> None:
    """Execute an auto-annotation run: one provider call per pending image.

    Handles both batch runs and single-image runs (a run with one task).
    Plain function (no task wrapper) so it can be driven directly in tests.
    """
    logger.info("Worker picked up inference run %d", run_id)

    run = (
        InferenceRun.objects.select_related("provider", "project", "created_by")
        .filter(pk=run_id)
        .first()
    )
    if run is None or run.status not in (
        InferenceRun.STATUS_PENDING,
        InferenceRun.STATUS_RUNNING,
    ):
        logger.info(
            "Inference run %d skipped: run=%s status=%s",
            run_id,
            "found" if run else "not_found",
            run.status if run else "n/a",
        )
        return

    logger.info(
        "Inference run %d starting: project=%s provider=%s total_items=%d",
        run_id,
        run.project.name,
        run.provider.name,
        run.total_items,
    )

    run.status = InferenceRun.STATUS_RUNNING
    run.started_at = run.started_at or timezone.now()
    run.error = ""

    # Recompute deadline from *now* so it reflects actual processing time,
    # not wall-clock time since run creation (worker may have been delayed).
    run.deadline = timezone.now() + timedelta(
        seconds=(run.total_items + 1) * run.provider.timeout_seconds
    )
    run.save(update_fields=["status", "started_at", "error", "deadline"])

    logger.info(
        "Inference run %d deadline set to %s (%d items × %ds timeout)",
        run_id,
        run.deadline.isoformat(),
        run.total_items,
        run.provider.timeout_seconds,
    )

    tasks = (
        run.tasks.filter(
            status__in=[InferenceTask.STATUS_PENDING, InferenceTask.STATUS_FAILED]
        )
        .select_related("image")
        .order_by("id")
    )

    task_count = len(tasks)
    logger.info("Inference run %d: %d images to process", run_id, task_count)

    final_status = InferenceRun.STATUS_COMPLETED
    error_message = ""
    for idx, task in enumerate(tasks, 1):
        # Cooperative cancel: re-read the flag fresh each iteration.
        if InferenceRun.objects.filter(pk=run.pk, cancel_requested=True).exists():
            logger.info(
                "Inference run %d: cancel requested, skipping remaining images (processed %d/%d)",
                run_id,
                idx - 1,
                task_count,
            )
            run.tasks.filter(
                status__in=[InferenceTask.STATUS_PENDING, InferenceTask.STATUS_FAILED]
            ).update(status=InferenceTask.STATUS_SKIPPED)
            final_status = InferenceRun.STATUS_CANCELLED
            break

        # Whole-run wall-clock deadline.
        if run.deadline and timezone.now() > run.deadline:
            logger.warning(
                "Inference run %d: deadline %s exceeded (processed %d/%d)",
                run_id,
                run.deadline.isoformat(),
                idx - 1,
                task_count,
            )
            run.tasks.filter(
                status__in=[InferenceTask.STATUS_PENDING, InferenceTask.STATUS_FAILED]
            ).update(status=InferenceTask.STATUS_SKIPPED)
            final_status = InferenceRun.STATUS_FAILED
            error_message = "deadline exceeded"
            break

        task.status = InferenceTask.STATUS_RUNNING
        task.attempts = F("attempts") + 1
        task.started_at = timezone.now()
        task.save(update_fields=["status", "attempts", "started_at"])
        task.refresh_from_db(fields=["attempts"])

        logger.info(
            "Inference run %d: processing image %d/%d (task_id=%d image_id=%d)",
            run_id,
            idx,
            task_count,
            task.id,
            task.image_id,
        )

        try:
            _process_task(
                task,
                provider=run.provider,
                project=run.project,
                performed_by=run.created_by,
            )
        except Exception as exc:  # one image's failure must not abort the run
            logger.error(
                "Inference run %d task %d failed: image_id=%d error=%s",
                run_id,
                task.id,
                task.image_id,
                exc,
                exc_info=True,
            )
            task.status = InferenceTask.STATUS_FAILED
            task.error = str(exc)
            task.finished_at = timezone.now()
            task.save(update_fields=["status", "error", "finished_at"])
            InferenceRun.objects.filter(pk=run.pk).update(
                failed_items=F("failed_items") + 1
            )
        else:
            InferenceRun.objects.filter(pk=run.pk).update(
                completed_items=F("completed_items") + 1,
                annotations_created=F("annotations_created") + task.annotations_created,
            )

    run.refresh_from_db()
    if (
        final_status == InferenceRun.STATUS_COMPLETED
        and run.completed_items == 0
        and run.failed_items > 0
    ):
        final_status = InferenceRun.STATUS_FAILED
    run.status = final_status
    run.error = error_message
    run.finished_at = timezone.now()
    run.save(update_fields=["status", "finished_at", "error"])

    logger.info(
        "Inference run %d finished: status=%s completed=%d failed=%d annotations=%d",
        run_id,
        final_status,
        run.completed_items,
        run.failed_items,
        run.annotations_created,
    )
