import json
import logging
import mimetypes
import os

import httpx
from django.db import transaction
from django.db.models import Max
from django.utils import timezone

from anno_images.models import Annotation2D, Box2D, Keypoint2D, Polygon2D, Operation
from anno_images.schemas import Box2DDataInput, Keypoint2DDataInput, Polygon2DDataInput
from anno_sdk import InteractiveInferenceRequestMeta, InteractiveInferenceResponse

from .models import (
    InteractiveInferenceOperation,
    InteractiveInferenceSession,
    VALID_PROMPT_TYPES,
)

logger = logging.getLogger(__name__)


def provider_snapshot(provider) -> dict:
    """Serialize a provider's config for an audit snapshot, minus any secret.

    ``auth_secret`` is deliberately excluded — the snapshot records *how* a job
    was configured without persisting the outbound credential. Works for both
    ``InferenceServiceProvider`` and ``InteractiveInferenceServiceProvider``.
    """
    snapshot = {
        "id": provider.id,
        "name": provider.name,
        "model_name": provider.model_name,
        "inference_url": provider.inference_url,
        "supported_result_types": list(provider.supported_result_types),
        "auth_type": provider.auth_type,
        "auth_param_name": provider.auth_param_name,
        "timeout_seconds": provider.timeout_seconds,
    }
    supported_prompt_types = getattr(provider, "supported_prompt_types", None)
    if supported_prompt_types is not None:
        snapshot["supported_prompt_types"] = list(supported_prompt_types)
    return snapshot


def _create_subtype(annotation, *, polygon=None, box=None, keypoint=None):
    """Create the subtype row for an annotation, mirroring
    ``Annotation2DController._create_subtype`` in anno_images."""
    if annotation.annotation_type == "polygon" and polygon is not None:
        Polygon2D.objects.create(annotation=annotation, points=polygon.points)
    elif annotation.annotation_type == "box" and box is not None:
        Box2D.objects.create(
            annotation=annotation,
            x=box.x,
            y=box.y,
            width=box.width,
            height=box.height,
            rotation=box.rotation,
        )
    elif annotation.annotation_type == "keypoint" and keypoint is not None:
        Keypoint2D.objects.create(annotation=annotation, points=keypoint.points)
    else:
        raise ValueError(
            "Missing or mismatched subtype data for "
            f"annotation_type='{annotation.annotation_type}'"
        )


def create_ai_annotation(
    *,
    image,
    project,
    annotation_type,
    label=None,
    polygon=None,
    box=None,
    keypoint=None,
    performed_by,
    source=Operation.SOURCE_INFERENCE,
):
    """Create one AI-contributed annotation, identical to the human write-path.

    Mirrors ``Annotation2DController.create``: creates the ``Annotation2D``, its
    subtype row, and an ``Operation(action="add")`` audit record. AI annotations
    are deliberately indistinguishable from human ones. ``source`` records the
    origin on the ``Operation`` (``inference`` for auto-annotation, ``interactive``
    for interactive-inference commits). The caller is responsible for the
    surrounding transaction / savepoint.
    """
    annotation = Annotation2D.objects.create(
        image=image,
        project=project,
        annotation_type=annotation_type,
        label=label,
    )
    _create_subtype(annotation, polygon=polygon, box=box, keypoint=keypoint)
    Operation.objects.create(
        image=image,
        to_annotation=annotation,
        action="add",
        source=source,
        performed_by=performed_by,
    )
    return annotation


def modify_ai_annotation(
    *,
    old_annotation,
    annotation_type,
    label=None,
    polygon=None,
    box=None,
    keypoint=None,
    performed_by,
    source=Operation.SOURCE_INFERENCE,
):
    """Modify an AI annotation using the immutable pattern.

    Takes the old annotation (already resolved and scoped by the caller),
    creates a new Annotation2D with the updated data, deactivates the old
    one, and records an Operation(action="modify"). ``source`` records the
    origin on the ``Operation``.

    The caller is responsible for the surrounding transaction and for
    resolving the old annotation (scoped to image + project + is_active).
    """
    new = Annotation2D.objects.create(
        image=old_annotation.image,
        project=old_annotation.project,
        annotation_type=annotation_type,
        label=label if label is not None else old_annotation.label,
    )
    _create_subtype(new, polygon=polygon, box=box, keypoint=keypoint)

    old_annotation.is_active = False
    old_annotation.save(update_fields=["is_active"])

    Operation.objects.create(
        image=old_annotation.image,
        from_annotation=old_annotation,
        to_annotation=new,
        action="modify",
        source=source,
        performed_by=performed_by,
    )

    return new


# ---------------------------------------------------------------------------
# Interactive inference (SAM/SAM2/MedSAM style): request-time provider calls
# ---------------------------------------------------------------------------


def _kwargs_from_result(result_type, result_data, label):
    """Build ``create_ai_annotation`` geometry kwargs from a stored candidate.

    ``result_data`` is the normalized geometry dict recorded on an
    :class:`InteractiveInferenceOperation` (``annotation.geometry.to_dict()``).
    """
    kwargs = {
        "annotation_type": result_type,
        "label": label,
        "polygon": None,
        "box": None,
        "keypoint": None,
    }
    if result_type == "polygon":
        kwargs["polygon"] = Polygon2DDataInput(points=result_data["points"])
    elif result_type == "keypoint":
        kwargs["keypoint"] = Keypoint2DDataInput(points=result_data["points"])
    elif result_type == "box":
        kwargs["box"] = Box2DDataInput(
            x=result_data["x"],
            y=result_data["y"],
            width=result_data["width"],
            height=result_data["height"],
            rotation=result_data.get("rotation") or 0.0,
        )
    else:
        raise ValueError(f"Cannot commit unsupported result_type {result_type!r}")
    return kwargs


def _interactive_auth(provider) -> tuple[dict, dict]:
    """Return ``(headers, params)`` carrying the provider credential."""
    headers: dict = {}
    params: dict = {}
    if provider.auth_type == provider.AUTH_HEADER and provider.auth_param_name:
        headers[provider.auth_param_name] = provider.auth_secret
    elif provider.auth_type == provider.AUTH_QUERY and provider.auth_param_name:
        params[provider.auth_param_name] = provider.auth_secret
    return headers, params


def _call_interactive_provider(
    provider, image_bytes: bytes, file_name: str, meta: InteractiveInferenceRequestMeta
) -> InteractiveInferenceResponse:
    """POST image bytes + prompt metadata to the provider; return its candidate."""
    headers, params = _interactive_auth(provider)
    content_type = mimetypes.guess_type(file_name or "")[0] or "application/octet-stream"
    files = {"image": (file_name or "image", image_bytes, content_type)}
    data = {"metadata": json.dumps(meta.to_dict())}

    resp = httpx.post(
        provider.inference_url,
        files=files,
        data=data,
        headers=headers,
        params=params,
        timeout=provider.timeout_seconds,
    )
    resp.raise_for_status()
    return InteractiveInferenceResponse.from_dict(resp.json())


def start_interactive_session(*, project, image, provider, performed_by, from_annotation=None):
    """Open an interactive session (status ``editing``)."""
    return InteractiveInferenceSession.objects.create(
        project=project,
        image=image,
        provider=provider,
        performed_by=performed_by,
        from_annotation=from_annotation,
    )


def run_interactive_step(session, prompts):
    """Run one interactive step: validate prompts, call the provider, record it.

    Records an :class:`InteractiveInferenceOperation` with the candidate the
    model returned. No ``Annotation2D`` / ``Operation`` is created here — that
    happens only on commit. On a provider failure the step is still recorded
    with its ``error`` populated, and the exception is re-raised so the caller
    can surface it (the session stays ``editing`` so the user may retry).
    """
    provider = session.provider
    supported = set(provider.supported_prompt_types)
    for p in prompts:
        ptype = p.get("type")
        if ptype not in VALID_PROMPT_TYPES:
            raise ValueError(f"Unknown prompt type {ptype!r}.")
        if ptype not in supported:
            raise ValueError(f"Provider does not support prompt type {ptype!r}.")

    step_index = (
        session.operations.aggregate(m=Max("step_index"))["m"] or 0
    ) + 1

    image = session.image
    with image.image.open("rb") as fh:
        image_bytes = fh.read()
    file_name = os.path.basename(image.image.name or "") or image.file_name

    meta = InteractiveInferenceRequestMeta(
        image_id=image.id,
        session_id=session.id,
        step_index=step_index,
        prompts=[dict(p) for p in prompts],
        label_mapping=session.project.label_mapping,
        requested_types=list(provider.supported_result_types),
        width=image.width,
        height=image.height,
    )

    try:
        response = _call_interactive_provider(provider, image_bytes, file_name, meta)
    except Exception as exc:
        logger.error(
            "Interactive session %d step %d failed: %s", session.id, step_index, exc
        )
        InteractiveInferenceOperation.objects.create(
            session=session,
            step_index=step_index,
            prompt={"prompts": meta.prompts},
            error=str(exc),
        )
        raise

    ann = response.annotation
    result_type = ann.geometry.annotation_type if ann is not None else ""
    result_data = ann.geometry.to_dict() if ann is not None else {}
    label = ann.label if ann is not None else None

    return InteractiveInferenceOperation.objects.create(
        session=session,
        step_index=step_index,
        prompt={"prompts": meta.prompts},
        result={"score": response.score, "label": label},
        result_type=result_type,
        result_data=result_data,
        raw_result=response.to_dict(),
    )


def commit_interactive_session(session, operation):
    """Commit a step's candidate as a real ``Annotation2D`` (``source=interactive``).

    When the session refines an existing annotation (``from_annotation`` set) the
    write is a ``modify`` (immutable pattern); otherwise an ``add``. Sets
    ``final_annotation`` and marks the session ``committed``.
    """
    if not operation.result_data:
        raise ValueError("Selected step has no candidate to commit.")

    label = (operation.result or {}).get("label")
    kwargs = _kwargs_from_result(operation.result_type, operation.result_data, label)

    with transaction.atomic():
        if session.from_annotation_id is not None:
            annotation = modify_ai_annotation(
                old_annotation=session.from_annotation,
                performed_by=session.performed_by,
                source=Operation.SOURCE_INTERACTIVE,
                **kwargs,
            )
        else:
            annotation = create_ai_annotation(
                image=session.image,
                project=session.project,
                performed_by=session.performed_by,
                source=Operation.SOURCE_INTERACTIVE,
                **kwargs,
            )
        session.final_annotation = annotation
        session.status = InteractiveInferenceSession.STATUS_COMMITTED
        session.committed_at = timezone.now()
        session.error = ""
        session.save(
            update_fields=["final_annotation", "status", "committed_at", "error"]
        )

    return annotation


def discard_interactive_session(session):
    """Abandon a session; no ``Annotation2D`` / ``Operation`` is created."""
    session.status = InteractiveInferenceSession.STATUS_DISCARDED
    session.discarded_at = timezone.now()
    session.save(update_fields=["status", "discarded_at"])
    return session
