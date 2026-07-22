import re

from django.db.models import Count, Q
from django.shortcuts import get_object_or_404
from ninja_extra import api_controller, http_delete, http_get, http_patch, http_post
from ninja_extra.exceptions import HttpError
from ninja_extra.permissions import IsAuthenticated
from ninja_jwt.authentication import JWTAuth

from anno.pagination import PaginatedResponse, paginate_queryset
from anno_images.models import Image2D
from anno_projects.models import Project
from anno_projects.permissions import IsProjectMemberOrAdmin, IsProjectSupervisorOrAdmin

from .models import ImageTag, ProjectTag
from .schemas import (
    ImageTagApplyInput,
    ImageTagOutput,
    TagCreateInput,
    TagOutput,
    TagStatItem,
    TagStatsOutput,
    TagUpdateInput,
)

# ---------------------------------------------------------------------------
# Project tags (supervisor-managed, member-visible)
# ---------------------------------------------------------------------------


@api_controller("/projects/{project_id}/tags", tags=["project-tags"])
class TagController:

    @http_get(
        "/",
        permissions=[IsAuthenticated, IsProjectMemberOrAdmin],
        auth=JWTAuth(),
        response={200: PaginatedResponse[TagOutput]},
        url_name="project_tag_list",
    )
    def list_tags(
        self,
        request,
        project_id: int,
        limit: int = 100,
        offset: int = 0,
        is_active: bool | None = None,
    ):
        qs = ProjectTag.objects.filter(project_id=project_id)
        if is_active is not None:
            qs = qs.filter(is_active=is_active)
        count, limit, offset, rows = paginate_queryset(qs, limit, offset)
        return 200, PaginatedResponse(
            count=count,
            limit=limit,
            offset=offset,
            items=[TagOutput.from_tag(t) for t in rows],
        )

    @http_get(
        "/stats",
        permissions=[IsAuthenticated, IsProjectMemberOrAdmin],
        auth=JWTAuth(),
        response={200: TagStatsOutput},
        url_name="project_tag_stats",
    )
    def tag_stats(self, request, project_id: int):
        qs = (
            ProjectTag.objects.filter(project_id=project_id, is_active=True)
            .annotate(
                image_count=Count(
                    "image_tags",
                    filter=Q(image_tags__deleted_at__isnull=True),
                )
            )
            .order_by("-image_count", "name")
        )
        tags = [
            TagStatItem(
                tag_id=t.id,
                name=t.name,
                display_name=t.display_name,
                color=t.color,
                image_count=t.image_count,
            )
            for t in qs
        ]
        return 200, TagStatsOutput(project_id=project_id, tags=tags)

    @http_get(
        "/{tag_id}",
        permissions=[IsAuthenticated, IsProjectMemberOrAdmin],
        auth=JWTAuth(),
        response={200: TagOutput},
        url_name="project_tag_detail",
    )
    def detail(self, request, project_id: int, tag_id: int):
        tag = get_object_or_404(ProjectTag, id=tag_id, project_id=project_id)
        return 200, TagOutput.from_tag(tag)

    @http_post(
        "/",
        permissions=[IsAuthenticated, IsProjectSupervisorOrAdmin],
        auth=JWTAuth(),
        response={201: TagOutput},
        url_name="project_tag_create",
    )
    def create(self, request, project_id: int, payload: TagCreateInput):
        project = get_object_or_404(Project, id=project_id)
        if not re.match(r"^[a-z0-9_]+$", payload.name):
            raise HttpError(
                400,
                "Tag name must contain only lowercase letters, digits, and underscores.",
            )
        if payload.permission_level not in dict(ProjectTag.PERMISSION_CHOICES):
            raise HttpError(
                400, "permission_level must be one of: common, worker, supervisor."
            )
        if ProjectTag.objects.filter(project=project, name=payload.name).exists():
            raise HttpError(
                409, f"Tag '{payload.name}' already exists in this project."
            )
        tag = ProjectTag.objects.create(
            project=project,
            name=payload.name,
            display_name=payload.display_name,
            color=payload.color,
            description=payload.description,
            permission_level=payload.permission_level,
            created_by=request.user,
        )
        return 201, TagOutput.from_tag(tag)

    @http_patch(
        "/{tag_id}",
        permissions=[IsAuthenticated, IsProjectSupervisorOrAdmin],
        auth=JWTAuth(),
        response={200: TagOutput},
        url_name="project_tag_update",
    )
    def update(self, request, project_id: int, tag_id: int, payload: TagUpdateInput):
        tag = get_object_or_404(ProjectTag, id=tag_id, project_id=project_id)
        data = payload.model_dump(exclude_unset=True)
        if (
            "permission_level" in data
            and data["permission_level"] not in dict(ProjectTag.PERMISSION_CHOICES)
        ):
            raise HttpError(
                400, "permission_level must be one of: common, worker, supervisor."
            )
        for attr, value in data.items():
            setattr(tag, attr, value)
        tag.save()
        return 200, TagOutput.from_tag(tag)

    @http_delete(
        "/{tag_id}",
        permissions=[IsAuthenticated, IsProjectSupervisorOrAdmin],
        auth=JWTAuth(),
        response={204: None},
        url_name="project_tag_delete",
    )
    def delete(self, request, project_id: int, tag_id: int):
        tag = get_object_or_404(ProjectTag, id=tag_id, project_id=project_id)
        # Soft delete does not cascade at the DB level, so soft-delete the tag's
        # applications explicitly — otherwise the tag would linger on images.
        tag.image_tags.all().delete()
        tag.delete()
        return 204, None


# ---------------------------------------------------------------------------
# Image tags (apply / remove / list)
# ---------------------------------------------------------------------------


@api_controller("/projects/{project_id}/images/{image_id}/tags", tags=["image-tags"])
class ImageTagController:

    @http_get(
        "/",
        permissions=[IsAuthenticated, IsProjectMemberOrAdmin],
        auth=JWTAuth(),
        response={200: list[ImageTagOutput]},
        url_name="image_tag_list",
    )
    def list_tags(self, request, project_id: int, image_id: int):
        img = get_object_or_404(Image2D, id=image_id, project_id=project_id)
        qs = ImageTag.objects.filter(image=img).select_related("tag")
        return 200, [ImageTagOutput.from_image_tag(it) for it in qs]

    @http_post(
        "/",
        permissions=[IsAuthenticated, IsProjectMemberOrAdmin],
        auth=JWTAuth(),
        response={201: ImageTagOutput},
        url_name="image_tag_apply",
    )
    def apply(
        self, request, project_id: int, image_id: int, payload: ImageTagApplyInput
    ):
        img = get_object_or_404(Image2D, id=image_id, project_id=project_id)
        tag = get_object_or_404(ProjectTag, id=payload.tag_id, project_id=project_id)
        if not tag.is_active:
            raise HttpError(400, "Cannot apply an inactive tag.")
        # Per-tag apply permission. "common" = any member; "worker"/"supervisor" =
        # only that membership role. Admins are treated by their actual project role;
        # a system admin who is not a project member (allowed here by
        # IsProjectMemberOrAdmin) has role None and falls through.
        if tag.permission_level != ProjectTag.PERMISSION_COMMON:
            role = img.project.get_membership_role(request.user)
            if role is not None and role != tag.permission_level:
                raise HttpError(
                    403,
                    f"Only {tag.permission_level}s can apply the '{tag.name}' tag.",
                )
        if ImageTag.objects.filter(image=img, tag=tag).exists():
            raise HttpError(409, f"Tag '{tag.name}' is already applied to this image.")
        image_tag = ImageTag.objects.create(
            image=img, tag=tag, applied_by=request.user, note=payload.note
        )
        # Re-fetch with joined tag for the output
        image_tag = ImageTag.objects.select_related("tag").get(pk=image_tag.pk)
        return 201, ImageTagOutput.from_image_tag(image_tag)

    @http_delete(
        "/{tag_id}",
        permissions=[IsAuthenticated, IsProjectMemberOrAdmin],
        auth=JWTAuth(),
        response={204: None},
        url_name="image_tag_remove",
    )
    def remove(self, request, project_id: int, image_id: int, tag_id: int):
        img = get_object_or_404(Image2D, id=image_id, project_id=project_id)
        image_tag = get_object_or_404(ImageTag, image=img, tag_id=tag_id)

        # Ownership check: workers can only remove their own tags.
        # Supervisors and admins can remove any tag in the project.
        role = img.project.get_user_role(request.user)
        if role not in ("admin", "supervisor"):
            if role != "worker" or image_tag.applied_by_id != request.user.id:
                raise HttpError(403, "You can only remove tags you applied yourself.")

        image_tag.delete()
        return 204, None
