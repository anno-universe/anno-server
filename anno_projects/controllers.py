from django.contrib.auth import get_user_model
from django.shortcuts import get_object_or_404
from ninja_extra import api_controller, http_delete, http_get, http_patch, http_post
from ninja_extra.exceptions import HttpError
from ninja_extra.permissions import IsAuthenticated
from ninja_jwt.authentication import JWTAuth

from anno.pagination import PaginatedResponse, paginate_queryset

from .models import Project, ProjectAPIKey, ProjectMembership
from .permissions import IsProjectMemberOrAdmin, IsProjectSupervisorOrAdmin
from .schemas import (
    AddProjectMemberInput,
    APIKeyCreatedOutput,
    APIKeyCreateInput,
    APIKeyOutput,
    APIKeyUpdateInput,
    ProjectCreateInput,
    ProjectMemberOutput,
    ProjectOutput,
    ProjectUpdateInput,
    UpdateMemberRoleInput,
)

User = get_user_model()


# ---------------------------------------------------------------------------
# Projects
# ---------------------------------------------------------------------------


@api_controller("/projects", tags=["projects"])
class ProjectController:

    @http_post(
        "/",
        permissions=[IsAuthenticated],
        auth=JWTAuth(),
        response={201: ProjectOutput},
        url_name="project_create",
    )
    def create(self, request, payload: ProjectCreateInput):
        project = Project.objects.create(
            name=payload.name,
            description=payload.description,
            meta_info=payload.meta_info,
            label_mapping=payload.label_mapping,
            created_by=request.user,
        )
        ProjectMembership.objects.get_or_create(
            project=project,
            user=request.user,
            defaults={"role": "supervisor", "added_by": request.user},
        )
        return 201, ProjectOutput.from_project(project, "supervisor")

    @http_get(
        "/",
        permissions=[IsAuthenticated],
        auth=JWTAuth(),
        response={200: PaginatedResponse[ProjectOutput]},
        url_name="project_list",
    )
    def list_projects(self, request, limit: int = 100, offset: int = 0):
        user = request.user
        if user.groups.filter(name="admin").exists():
            qs = Project.objects.all()
            count, limit, offset, rows = paginate_queryset(qs, limit, offset)
            return 200, PaginatedResponse(
                count=count, limit=limit, offset=offset,
                items=[ProjectOutput.from_project(p, "admin") for p in rows],
            )
        qs = ProjectMembership.objects.filter(user=user).select_related("project")
        count, limit, offset, rows = paginate_queryset(qs, limit, offset)
        return 200, PaginatedResponse(
            count=count, limit=limit, offset=offset,
            items=[ProjectOutput.from_project(m.project, m.role) for m in rows],
        )

    @http_get(
        "/{project_id}",
        permissions=[IsAuthenticated, IsProjectMemberOrAdmin],
        auth=JWTAuth(),
        response={200: ProjectOutput},
        url_name="project_detail",
    )
    def detail(self, request, project_id: int):
        project = get_object_or_404(Project, id=project_id)
        role = project.get_user_role(request.user)
        return 200, ProjectOutput.from_project(project, role)

    @http_patch(
        "/{project_id}",
        permissions=[IsAuthenticated, IsProjectSupervisorOrAdmin],
        auth=JWTAuth(),
        response={200: ProjectOutput},
        url_name="project_update",
    )
    def update(self, request, project_id: int, payload: ProjectUpdateInput):
        project = get_object_or_404(Project, id=project_id)
        for attr, value in payload.model_dump(exclude_unset=True).items():
            setattr(project, attr, value)
        project.save()
        role = project.get_user_role(request.user)
        return 200, ProjectOutput.from_project(project, role)

    @http_delete(
        "/{project_id}",
        permissions=[IsAuthenticated, IsProjectSupervisorOrAdmin],
        auth=JWTAuth(),
        response={204: None},
        url_name="project_delete",
    )
    def delete(self, request, project_id: int):
        project = get_object_or_404(Project, id=project_id)
        project.delete()
        return 204, None

    @http_get(
        "/{project_id}/members",
        permissions=[IsAuthenticated, IsProjectSupervisorOrAdmin],
        auth=JWTAuth(),
        response={200: PaginatedResponse[ProjectMemberOutput]},
        url_name="project_members_list",
    )
    def list_members(
        self, request, project_id: int, limit: int = 100, offset: int = 0
    ):
        project = get_object_or_404(Project, id=project_id)
        qs = ProjectMembership.objects.filter(project=project).select_related("user")
        count, limit, offset, rows = paginate_queryset(qs, limit, offset)
        return 200, PaginatedResponse(
            count=count, limit=limit, offset=offset,
            items=[ProjectMemberOutput.from_membership(m) for m in rows],
        )

    @http_post(
        "/{project_id}/members",
        permissions=[IsAuthenticated, IsProjectSupervisorOrAdmin],
        auth=JWTAuth(),
        response={201: ProjectMemberOutput},
        url_name="project_member_add",
    )
    def add_member(self, request, project_id: int, payload: AddProjectMemberInput):
        if payload.role not in ("worker", "supervisor"):
            raise HttpError(400, "Role must be 'worker' or 'supervisor'.")
        project = get_object_or_404(Project, id=project_id)
        user = get_object_or_404(User, id=payload.user_id)
        membership, created = ProjectMembership.objects.get_or_create(
            project=project,
            user=user,
            defaults={"role": payload.role, "added_by": request.user},
        )
        if not created:
            raise HttpError(409, "User is already a member of this project.")
        return 201, ProjectMemberOutput.from_membership(membership)

    @http_patch(
        "/{project_id}/members/{user_id}",
        permissions=[IsAuthenticated, IsProjectSupervisorOrAdmin],
        auth=JWTAuth(),
        response={200: ProjectMemberOutput},
        url_name="project_member_update_role",
    )
    def update_member_role(
        self, request, project_id: int, user_id: int, payload: UpdateMemberRoleInput
    ):
        if payload.role not in ("worker", "supervisor"):
            raise HttpError(400, "Role must be 'worker' or 'supervisor'.")
        membership = get_object_or_404(
            ProjectMembership, project_id=project_id, user_id=user_id
        )
        if membership.role == "supervisor" and payload.role == "worker":
            if membership.user_id == membership.project.created_by_id:
                raise HttpError(400, "Cannot demote the project creator to worker.")
        membership.role = payload.role
        membership.save(update_fields=["role", "updated_at"])
        return 200, ProjectMemberOutput.from_membership(membership)

    @http_delete(
        "/{project_id}/members/{user_id}",
        permissions=[IsAuthenticated, IsProjectSupervisorOrAdmin],
        auth=JWTAuth(),
        response={204: None},
        url_name="project_member_remove",
    )
    def remove_member(self, request, project_id: int, user_id: int):
        membership = get_object_or_404(
            ProjectMembership, project_id=project_id, user_id=user_id
        )
        if membership.user_id == membership.project.created_by_id:
            raise HttpError(400, "Cannot remove the project creator from the project.")
        membership.delete()
        return 204, None


# ---------------------------------------------------------------------------
# API key management (JWT, supervisor-only)
# ---------------------------------------------------------------------------


@api_controller("/projects/{project_id}/api-keys", tags=["infer-keys"])
class APIKeyController:

    @http_post(
        "/",
        permissions=[IsAuthenticated, IsProjectSupervisorOrAdmin],
        auth=JWTAuth(),
        response={201: APIKeyCreatedOutput},
        url_name="infer_apikey_create",
    )
    def create(self, request, project_id: int, payload: APIKeyCreateInput):
        project = get_object_or_404(Project, id=project_id)
        instance, token = ProjectAPIKey.generate(
            project=project,
            name=payload.name,
            created_by=request.user,
            expires_at=payload.expires_at,
        )
        instance.save()
        return 201, APIKeyCreatedOutput.from_api_key_with_token(instance, token)

    @http_get(
        "/",
        permissions=[IsAuthenticated, IsProjectSupervisorOrAdmin],
        auth=JWTAuth(),
        response={200: PaginatedResponse[APIKeyOutput]},
        url_name="infer_apikey_list",
    )
    def list_keys(
        self, request, project_id: int, limit: int = 100, offset: int = 0
    ):
        qs = ProjectAPIKey.objects.filter(project_id=project_id)
        count, limit, offset, rows = paginate_queryset(qs, limit, offset)
        return 200, PaginatedResponse(
            count=count, limit=limit, offset=offset,
            items=[APIKeyOutput.from_api_key(k) for k in rows],
        )

    @http_get(
        "/{key_id}",
        permissions=[IsAuthenticated, IsProjectSupervisorOrAdmin],
        auth=JWTAuth(),
        response={200: APIKeyOutput},
        url_name="infer_apikey_detail",
    )
    def detail(self, request, project_id: int, key_id: int):
        key = get_object_or_404(ProjectAPIKey, id=key_id, project_id=project_id)
        return 200, APIKeyOutput.from_api_key(key)

    @http_patch(
        "/{key_id}",
        permissions=[IsAuthenticated, IsProjectSupervisorOrAdmin],
        auth=JWTAuth(),
        response={200: APIKeyOutput},
        url_name="infer_apikey_update",
    )
    def update(self, request, project_id: int, key_id: int, payload: APIKeyUpdateInput):
        key = get_object_or_404(ProjectAPIKey, id=key_id, project_id=project_id)
        for attr, value in payload.model_dump(exclude_unset=True).items():
            setattr(key, attr, value)
        key.save()
        return 200, APIKeyOutput.from_api_key(key)

    @http_delete(
        "/{key_id}",
        permissions=[IsAuthenticated, IsProjectSupervisorOrAdmin],
        auth=JWTAuth(),
        response={204: None},
        url_name="infer_apikey_delete",
    )
    def delete(self, request, project_id: int, key_id: int):
        key = get_object_or_404(ProjectAPIKey, id=key_id, project_id=project_id)
        key.delete()
        return 204, None
