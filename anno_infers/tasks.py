"""Background worker for server-driven auto-annotation (Flow B).

``run_inference_job`` is enqueued by the auto-annotate endpoint and executed by
the ``db_worker`` process. For each pending image it sends the raw image bytes
plus a JSON metadata block to the provider's inference URL, parses the response
with the shared anno-sdk contract, validates the returned geometry against the
provider's declared result types, and writes annotations through the existing
``create_ai_annotation`` write-path (which records an ``Operation``).

Persistence, cooperative cancellation, the whole-job deadline and per-item
replay all live in the job/item rows, so this worker is just a driver.
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

from anno_images.schemas import Box2DDataInput, Keypoint2DDataInput, Polygon2DDataInput
from anno_sdk import InferenceRequestMeta, InferenceResponse

from .models import InferenceJob, InferenceJobItem
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
        "Calling provider %s at %s: image_id=%s job_id=%s timeout=%ds size=%d",
        provider.name,
        provider.inference_url,
        meta.image_id,
        meta.job_id,
        provider.timeout_seconds,
        len(image_bytes),
    )
    resp = httpx.post(
        provider.inference_url,
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


def _process_item(job, item: InferenceJobItem) -> None:
    """Run one image through the provider and persist its annotations.

    Raises on any failure so the caller can mark the item failed; all DB writes
    for the item are wrapped in a single atomic block.
    """
    provider = job.provider
    image = item.image

    logger.info(
        "Processing item %d: image_id=%d file=%s (attempt %d)",
        item.id,
        image.id,
        image.file_name,
        item.attempts + 1,
    )

    with image.image.open("rb") as fh:
        image_bytes = fh.read()
    file_name = os.path.basename(image.image.name or "") or image.file_name

    meta = InferenceRequestMeta(
        image_id=image.id,
        job_id=job.id,
        label_mapping=job.project.label_mapping,
        requested_types=list(provider.supported_result_types),
        width=image.width,
        height=image.height,
    )

    response = _call_provider(provider, image_bytes, file_name, meta)

    supported = set(provider.supported_result_types)
    created = 0
    with transaction.atomic():
        for ann in response.annotations:
            if ann.geometry.annotation_type not in supported:
                logger.warning(
                    "Item %d: provider returned unsupported type %r for image_id=%d, skipping",
                    item.id,
                    ann.geometry.annotation_type,
                    image.id,
                )
                continue
            create_ai_annotation(
                image=image,
                project=job.project,
                performed_by=job.created_by,
                **_annotation_to_kwargs(ann),
            )
            created += 1

    item.annotations_created = created
    item.status = InferenceJobItem.STATUS_DONE
    item.error = ""
    item.finished_at = timezone.now()
    item.save(update_fields=["annotations_created", "status", "error", "finished_at"])

    InferenceJob.objects.filter(pk=job.pk).update(
        completed_items=F("completed_items") + 1,
        annotations_created=F("annotations_created") + created,
    )

    logger.info(
        "Item %d done: image_id=%d annotations_created=%d",
        item.id,
        image.id,
        created,
    )


@task()
def run_inference_job(job_id: int) -> None:
    """Task entrypoint enqueued by the auto-annotate endpoint / db_worker."""
    execute_inference_job(job_id)


def execute_inference_job(job_id: int) -> None:
    """Execute an auto-annotation job: one provider call per pending image.

    Plain function (no task wrapper) so it can be driven directly in tests.
    """
    logger.info("Worker picked up job %d", job_id)

    job = (
        InferenceJob.objects.select_related("provider", "project", "created_by")
        .filter(pk=job_id)
        .first()
    )
    if job is None or job.status not in (
        InferenceJob.STATUS_PENDING,
        InferenceJob.STATUS_RUNNING,
    ):
        logger.info(
            "Job %d skipped: job=%s status=%s",
            job_id,
            "found" if job else "not_found",
            job.status if job else "n/a",
        )
        return

    logger.info(
        "Job %d starting: project=%s provider=%s total_items=%d",
        job_id,
        job.project.name,
        job.provider.name,
        job.total_items,
    )

    job.status = InferenceJob.STATUS_RUNNING
    job.started_at = job.started_at or timezone.now()
    job.error = ""

    # Recompute deadline from *now* so it reflects actual processing time,
    # not wall-clock time since job creation (worker may have been delayed).
    job.deadline = timezone.now() + timedelta(
        seconds=(job.total_items + 1) * job.provider.timeout_seconds
    )
    job.save(update_fields=["status", "started_at", "error", "deadline"])

    logger.info(
        "Job %d deadline set to %s (%d items × %ds timeout)",
        job_id,
        job.deadline.isoformat(),
        job.total_items,
        job.provider.timeout_seconds,
    )

    items = (
        job.items.filter(
            status__in=[InferenceJobItem.STATUS_PENDING, InferenceJobItem.STATUS_FAILED]
        )
        .select_related("image")
        .order_by("id")
    )

    item_count = len(items)
    logger.info("Job %d: %d items to process", job_id, item_count)

    final_status = InferenceJob.STATUS_COMPLETED
    error_message = ""
    for idx, item in enumerate(items, 1):
        # Cooperative cancel: re-read the flag fresh each iteration.
        if InferenceJob.objects.filter(pk=job.pk, cancel_requested=True).exists():
            logger.info(
                "Job %d: cancel requested, skipping remaining items (processed %d/%d)",
                job_id,
                idx - 1,
                item_count,
            )
            job.items.filter(
                status__in=[InferenceJobItem.STATUS_PENDING, InferenceJobItem.STATUS_FAILED]
            ).update(status=InferenceJobItem.STATUS_SKIPPED)
            final_status = InferenceJob.STATUS_CANCELLED
            break

        # Whole-job wall-clock deadline.
        if job.deadline and timezone.now() > job.deadline:
            logger.warning(
                "Job %d: deadline %s exceeded (processed %d/%d)",
                job_id,
                job.deadline.isoformat(),
                idx - 1,
                item_count,
            )
            job.items.filter(
                status__in=[InferenceJobItem.STATUS_PENDING, InferenceJobItem.STATUS_FAILED]
            ).update(status=InferenceJobItem.STATUS_SKIPPED)
            final_status = InferenceJob.STATUS_FAILED
            error_message = "deadline exceeded"
            break

        item.status = InferenceJobItem.STATUS_RUNNING
        item.attempts = F("attempts") + 1
        item.started_at = timezone.now()
        item.save(update_fields=["status", "attempts", "started_at"])
        item.refresh_from_db(fields=["attempts"])

        logger.info(
            "Job %d: processing item %d/%d (item_id=%d image_id=%d)",
            job_id,
            idx,
            item_count,
            item.id,
            item.image_id,
        )

        try:
            _process_item(job, item)
        except Exception as exc:  # one item's failure must not abort the job
            logger.error(
                "Job %d item %d failed: image_id=%d error=%s",
                job_id,
                item.id,
                item.image_id,
                exc,
                exc_info=True,
            )
            item.status = InferenceJobItem.STATUS_FAILED
            item.error = str(exc)
            item.finished_at = timezone.now()
            item.save(update_fields=["status", "error", "finished_at"])
            InferenceJob.objects.filter(pk=job.pk).update(failed_items=F("failed_items") + 1)

    job.refresh_from_db()
    if final_status == InferenceJob.STATUS_COMPLETED and job.completed_items == 0 and job.failed_items > 0:
        final_status = InferenceJob.STATUS_FAILED
    job.status = final_status
    job.error = error_message
    job.finished_at = timezone.now()
    job.save(update_fields=["status", "finished_at", "error"])

    logger.info(
        "Job %d finished: status=%s completed=%d failed=%d annotations=%d",
        job_id,
        final_status,
        job.completed_items,
        job.failed_items,
        job.annotations_created,
    )
