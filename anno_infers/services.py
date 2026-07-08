import logging

import httpx
from anno_sdk import InteractiveSessionCreateRequest, InteractiveSessionCreateResponse

from anno_images.models import Annotation2D, Box2D, Keypoint2D, Operation, Polygon2D
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


# ---------------------------------------------------------------------------
# Interactive inference — server <-> service session handshake
#
# The latency-sensitive per-prompt loop runs browser -> service directly; the
# server only performs a one-time handshake to open the session (the service
# mints the short-lived browser token) and a best-effort call to release the
# service's seat when the session ends. Both are authenticated with the
# provider's outbound credential (``auth_type`` / ``auth_param_name`` /
# ``auth_secret``) — the same mechanism Flow B uses.
# ---------------------------------------------------------------------------


def _provider_auth(provider) -> tuple[dict, dict]:
    """Return ``(headers, params)`` carrying the provider's outbound credential."""
    headers: dict = {}
    params: dict = {}
    if provider.auth_type == provider.AUTH_HEADER and provider.auth_param_name:
        headers[provider.auth_param_name] = provider.auth_secret
    elif provider.auth_type == provider.AUTH_QUERY and provider.auth_param_name:
        params[provider.auth_param_name] = provider.auth_secret
    return headers, params


def open_interactive_session(
    *,
    provider,
    session,
    ttl_seconds,
    label_mapping,
    requested_types,
    width=None,
    height=None,
) -> InteractiveSessionCreateResponse:
    """Handshake with the service to open ``session``; return its token reply.

    POSTs an ``InteractiveSessionCreateRequest`` to ``{inference_url}/session``.
    The service mints a short-lived token bound to the session and returns an
    ``InteractiveSessionCreateResponse`` the caller relays to the browser.
    Raises on transport/HTTP error so the caller can fail the session.
    """
    headers, params = _provider_auth(provider)
    req = InteractiveSessionCreateRequest(
        session_id=session.id,
        image_id=session.image_id,
        requested_types=list(requested_types),
        label_mapping=label_mapping,
        ttl_seconds=ttl_seconds,
        width=width,
        height=height,
    )
    url = provider.inference_url.rstrip("/") + "/session"
    resp = httpx.post(
        url,
        json=req.to_dict(),
        headers=headers,
        params=params,
        timeout=provider.timeout_seconds,
    )
    resp.raise_for_status()
    return InteractiveSessionCreateResponse.from_dict(resp.json())


def complete_interactive_session(*, provider, session_id) -> None:
    """Best-effort release of the service's per-session seat.

    Called after a session is committed or discarded so the service frees its
    image cache / GPU seat promptly instead of waiting for token expiry. Any
    error is logged and swallowed — the session is already finalized locally.
    """
    headers, params = _provider_auth(provider)
    url = provider.inference_url.rstrip("/") + f"/session/{session_id}/complete"
    try:
        resp = httpx.post(
            url,
            headers=headers,
            params=params,
            timeout=provider.timeout_seconds,
        )
        resp.raise_for_status()
    except Exception:
        logger.warning(
            "Interactive session %s: complete handshake failed (ignored).",
            session_id,
            exc_info=True,
        )


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
