import logging
from io import BytesIO
from urllib.parse import urlparse

import boto3
from botocore.client import Config
from django.conf import settings as django_settings
from django.core.cache import caches
from django.db import transaction
from django.db.models import Count, Q
from django.http import HttpResponse, StreamingHttpResponse
from django.shortcuts import get_object_or_404
from ninja import File
from ninja.files import UploadedFile
from ninja_extra import api_controller, http_delete, http_get, http_patch, http_post
from ninja_extra.exceptions import HttpError
from ninja_extra.permissions import IsAuthenticated
from ninja_jwt.authentication import JWTAuth
from PIL import Image as PILImage

from anno.pagination import PaginatedResponse, paginate_queryset
from anno_projects.models import Project
from anno_projects.permissions import IsProjectMemberOrAdmin, IsProjectSupervisorOrAdmin
from anno_tags.models import ProjectTag

from .models import Annotation2D, Box2D, Image2D, Keypoint2D, Polygon2D, Operation
from .schemas import (
    Annotation2DCreateInput,
    Annotation2DOutput,
    Image2DOutput,
    ImageURLOutput,
    OperationOutput,
)

logger = logging.getLogger(__name__)


def _get_s3_client():
    """Return a boto3 S3 client configured for RustFS."""
    return boto3.client(
        "s3",
        endpoint_url=django_settings.AWS_S3_ENDPOINT_URL,
        aws_access_key_id=django_settings.AWS_ACCESS_KEY_ID,
        aws_secret_access_key=django_settings.AWS_SECRET_ACCESS_KEY,
        config=Config(signature_version="s3v4"),
        region_name=django_settings.AWS_S3_REGION_NAME,
    )


def _presigned_url(key: str, prefix: str) -> str:
    """Return a browser-usable URL for an S3 object: the internal proxy path
    (``prefix``) with a pre-signed S3 query string attached.  The URL is
    relative so it routes through Caddy (prod) / the Vite proxy (dev) to
    RustFS, which validates the signature and serves the object — RustFS is
    never addressed directly."""
    s3 = _get_s3_client()
    presigned = s3.generate_presigned_url(
        "get_object",
        Params={"Bucket": django_settings.AWS_STORAGE_BUCKET_NAME, "Key": key},
        ExpiresIn=3600,
    )
    query = urlparse(presigned).query
    return f"{prefix.rstrip('/')}/{key}?{query}"


def _presigned_redirect(key: str, prefix: str) -> HttpResponse:
    """Build a 307 redirect to the pre-signed internal proxy path.  Used by
    the machine/SDK image endpoint, whose callers expect raw bytes and follow
    the redirect."""
    response = HttpResponse(status=307)
    response["Location"] = _presigned_url(key, prefix)
    return response


# ---------------------------------------------------------------------------
# Images 2D
# ---------------------------------------------------------------------------


@api_controller("/projects/{project_id}/images", tags=["images"])
class Image2DController:

    @http_post(
        "/",
        permissions=[IsAuthenticated, IsProjectSupervisorOrAdmin],
        auth=JWTAuth(),
        response={201: Image2DOutput},
        url_name="image_upload",
    )
    def upload(self, request, project_id: int, file: File[UploadedFile]):
        project = get_object_or_404(Project, id=project_id)
        # Read the pixel dimensions from the in-memory upload before saving, so
        # the frontend renders annotations in the image's real coordinate space
        # (and AI inference meta carries the correct width/height). PIL.open is
        # lazy — it parses only the header, not the full image — and reading the
        # already-uploaded bytes avoids an extra round-trip to storage. Without
        # this, width/height stay null and the client falls back to a fake
        # extent, misplacing/scaling annotations.
        width = height = None
        try:
            with PILImage.open(file) as pil_img:
                width, height = pil_img.size
        except Exception:
            pass  # non-image or unreadable header — leave dimensions null
        finally:
            file.seek(0)

        try:
            img = Image2D.objects.create(
                project=project,
                image=file,
                file_name=file.name,
                width=width,
                height=height,
            )
        except Exception:
            logger.exception(
                "Image upload failed for project %d: %s", project_id, file.name
            )
            raise HttpError(500, "Image upload failed. Check that the storage service is running and the bucket exists.")
        return 201, Image2DOutput.from_image(img)

    @http_get(
        "/",
        permissions=[IsAuthenticated, IsProjectMemberOrAdmin],
        auth=JWTAuth(),
        response={200: PaginatedResponse[Image2DOutput]},
        url_name="image_list",
    )
    def list_images(
        self,
        request,
        project_id: int,
        limit: int = 100,
        offset: int = 0,
        tag: str | None = None,
    ):
        project = get_object_or_404(Project, id=project_id)
        qs = Image2D.objects.filter(project=project)

        # Tag filtering with AND semantics (comma-separated tag names)
        if tag:
            tag_names = [t.strip() for t in tag.split(",") if t.strip()]
            if tag_names:
                tag_ids = list(
                    ProjectTag.objects.filter(
                        project=project, name__in=tag_names, is_active=True
                    ).values_list("id", flat=True)
                )
                if len(tag_ids) != len(tag_names):
                    found = set(
                        ProjectTag.objects.filter(
                            project=project, name__in=tag_names, is_active=True
                        ).values_list("name", flat=True)
                    )
                    missing = [n for n in tag_names if n not in found]
                    raise HttpError(400, f"Unknown tag(s): {', '.join(missing)}")
                for tid in tag_ids:
                    qs = qs.filter(tags__tag_id=tid)

        qs = (
            qs.annotate(
                annotation_count=Count(
                    "annotations", filter=Q(annotations__is_active=True)
                )
            )
            .prefetch_related("tags__tag")
            .distinct()
        )
        count, limit, offset, rows = paginate_queryset(qs, limit, offset)
        return 200, PaginatedResponse(
            count=count,
            limit=limit,
            offset=offset,
            items=[Image2DOutput.from_image(img) for img in rows],
        )

    @http_get(
        "/{image_id}",
        permissions=[IsAuthenticated, IsProjectMemberOrAdmin],
        auth=JWTAuth(),
        response={200: Image2DOutput},
        url_name="image_detail",
    )
    def detail(self, request, project_id: int, image_id: int):
        img = get_object_or_404(Image2D, id=image_id, project_id=project_id)
        return 200, Image2DOutput.from_image(img)

    @http_get(
        "/{image_id}/original_image",
        permissions=[IsAuthenticated, IsProjectMemberOrAdmin],
        auth=JWTAuth(),
        response={200: ImageURLOutput},
        url_name="image_file",
    )
    def file(self, request, project_id: int, image_id: int):
        """Mint a pre-signed URL for the original image and hand it to the
        client, which loads the bytes directly from RustFS-behind-Caddy.
        Membership is checked here; the signed URL is the capability."""
        img = get_object_or_404(Image2D, id=image_id, project_id=project_id)
        return 200, ImageURLOutput(
            url=_presigned_url(img.image.name, django_settings.IMAGE_PROXY_PREFIX)
        )

    @http_get(
        "/{image_id}/thumbnail_image",
        permissions=[IsAuthenticated, IsProjectMemberOrAdmin],
        auth=JWTAuth(),
        url_name="image_thumbnail",
    )
    def thumbnail(
        self,
        request,
        project_id: int,
        image_id: int,
        w: int = 300,
        h: int = 300,
    ):
        w = max(50, min(w, 800))
        h = max(50, min(h, 800))

        cache_key = f"thumbnail_{image_id}_{w}x{h}"
        thumb_cache = caches["thumbnails"]

        cached = thumb_cache.get(cache_key)
        if cached is not None:
            if django_settings.DEBUG:
                return StreamingHttpResponse(
                    BytesIO(cached), content_type="image/jpeg"
                )
            response = HttpResponse(status=307)
            path = thumb_cache.filepath(cache_key)
            response["Location"] = (
                f"{django_settings.THUMB_CACHE_PREFIX.rstrip('/')}/{path.name}"
            )
            return response

        # Cache miss — resize and store
        img = get_object_or_404(Image2D, id=image_id, project_id=project_id)
        img.image.open()
        pil_img = PILImage.open(img.image)
        pil_img.thumbnail((w, h), PILImage.LANCZOS)
        buf = BytesIO()
        pil_img.save(buf, format="JPEG", quality=85)
        img.image.close()
        data = buf.getvalue()

        thumb_cache.set(cache_key, data)

        if django_settings.DEBUG:
            return StreamingHttpResponse(BytesIO(data), content_type="image/jpeg")
        response = HttpResponse(status=307)
        path = thumb_cache.filepath(cache_key)
        response["Location"] = (
            f"{django_settings.THUMB_CACHE_PREFIX.rstrip('/')}/{path.name}"
        )
        return response


# ---------------------------------------------------------------------------
# Annotations 2D (immutable pattern)
# ---------------------------------------------------------------------------


@api_controller(
    "/projects/{project_id}/images/{image_id}/annotations", tags=["annotations"]
)
class Annotation2DController:

    @http_post(
        "/",
        permissions=[IsAuthenticated, IsProjectMemberOrAdmin],
        auth=JWTAuth(),
        response={201: Annotation2DOutput},
        url_name="annotation_create",
    )
    @transaction.atomic
    def create(
        self, request, project_id: int, image_id: int, payload: Annotation2DCreateInput
    ):
        project = get_object_or_404(Project, id=project_id)
        img = get_object_or_404(Image2D, id=image_id, project_id=project_id)

        annotation = Annotation2D.objects.create(
            image=img,
            project=project,
            annotation_type=payload.annotation_type,
            label=payload.label,
        )

        self._create_subtype(annotation, payload)
        Operation.objects.create(
            image=img,
            to_annotation=annotation,
            action="add",
            source=Operation.SOURCE_HUMAN,
            performed_by=request.user,
        )

        return 201, Annotation2DOutput.from_annotation(annotation)

    @http_get(
        "/",
        permissions=[IsAuthenticated, IsProjectMemberOrAdmin],
        auth=JWTAuth(),
        response={200: PaginatedResponse[Annotation2DOutput]},
        url_name="annotation_list",
    )
    def list_annotations(
        self,
        request,
        project_id: int,
        image_id: int,
        limit: int = 100,
        offset: int = 0,
    ):
        qs = Annotation2D.objects.filter(
            image_id=image_id,
            project_id=project_id,
            is_active=True,
        ).select_related("polygon", "box", "keypoint")
        count, limit, offset, rows = paginate_queryset(qs, limit, offset)
        return 200, PaginatedResponse(
            count=count, limit=limit, offset=offset,
            items=[Annotation2DOutput.from_annotation(a) for a in rows],
        )

    @http_get(
        "/{annotation_id}",
        permissions=[IsAuthenticated, IsProjectMemberOrAdmin],
        auth=JWTAuth(),
        response={200: Annotation2DOutput},
        url_name="annotation_detail",
    )
    def detail(self, request, project_id: int, image_id: int, annotation_id: int):
        annotation = get_object_or_404(
            Annotation2D.objects.select_related("polygon", "box", "keypoint"),
            id=annotation_id,
            image_id=image_id,
            project_id=project_id,
        )
        return 200, Annotation2DOutput.from_annotation(annotation)

    @http_patch(
        "/{annotation_id}",
        permissions=[IsAuthenticated, IsProjectMemberOrAdmin],
        auth=JWTAuth(),
        response={200: Annotation2DOutput},
        url_name="annotation_modify",
    )
    @transaction.atomic
    def modify(
        self,
        request,
        project_id: int,
        image_id: int,
        annotation_id: int,
        payload: Annotation2DCreateInput,
    ):
        old = get_object_or_404(
            Annotation2D.objects.select_related("polygon", "box", "keypoint"),
            id=annotation_id,
            image_id=image_id,
            project_id=project_id,
            is_active=True,
        )

        new = Annotation2D.objects.create(
            image=old.image,
            project=old.project,
            annotation_type=payload.annotation_type,
            label=payload.label if payload.label is not None else old.label,
        )
        self._create_subtype(new, payload)

        old.is_active = False
        old.save(update_fields=["is_active"])

        Operation.objects.create(
            image=old.image,
            from_annotation=old,
            to_annotation=new,
            action="modify",
            source=Operation.SOURCE_HUMAN,
            performed_by=request.user,
        )

        return 200, Annotation2DOutput.from_annotation(new)

    @http_delete(
        "/{annotation_id}",
        permissions=[IsAuthenticated, IsProjectMemberOrAdmin],
        auth=JWTAuth(),
        response={204: None},
        url_name="annotation_delete",
    )
    @transaction.atomic
    def delete(self, request, project_id: int, image_id: int, annotation_id: int):
        annotation = get_object_or_404(
            Annotation2D,
            id=annotation_id,
            image_id=image_id,
            project_id=project_id,
            is_active=True,
        )
        annotation.is_active = False
        annotation.save(update_fields=["is_active"])

        Operation.objects.create(
            image=annotation.image,
            from_annotation=annotation,
            action="delete",
            source=Operation.SOURCE_HUMAN,
            performed_by=request.user,
        )

        return 204, None

    @staticmethod
    def _create_subtype(annotation: Annotation2D, payload: Annotation2DCreateInput):
        if annotation.annotation_type == "polygon" and payload.polygon:
            Polygon2D.objects.create(annotation=annotation, points=payload.polygon.points)
        elif annotation.annotation_type == "box" and payload.box:
            box_data = payload.box
            Box2D.objects.create(
                annotation=annotation,
                x=box_data.x,
                y=box_data.y,
                width=box_data.width,
                height=box_data.height,
                rotation=box_data.rotation,
            )
        elif annotation.annotation_type == "keypoint" and payload.keypoint:
            Keypoint2D.objects.create(
                annotation=annotation, points=payload.keypoint.points
            )
        else:
            raise ValueError(
                f"Missing subtype data for annotation_type='{annotation.annotation_type}'"
            )


# ---------------------------------------------------------------------------
# Operations (read-only audit trail)
# ---------------------------------------------------------------------------


@api_controller(
    "/projects/{project_id}/images/{image_id}/operations", tags=["operations"]
)
class OperationController:

    @http_get(
        "/",
        permissions=[IsAuthenticated, IsProjectMemberOrAdmin],
        auth=JWTAuth(),
        response={200: PaginatedResponse[OperationOutput]},
        url_name="operation_list",
    )
    def list_operations(
        self,
        request,
        project_id: int,
        image_id: int,
        limit: int = 100,
        offset: int = 0,
    ):
        qs = Operation.objects.filter(image_id=image_id)
        count, limit, offset, rows = paginate_queryset(qs, limit, offset)
        return 200, PaginatedResponse(
            count=count, limit=limit, offset=offset,
            items=[OperationOutput.from_operation(op) for op in rows],
        )
