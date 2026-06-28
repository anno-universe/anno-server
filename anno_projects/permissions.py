from django.http import HttpRequest
from ninja_extra.permissions import BasePermission

from .models import ProjectMembership


def _is_admin(user) -> bool:
    return bool(
        user
        and user.is_authenticated
        and user.groups.filter(name="admin").exists()
    )


class IsProjectMemberOrAdmin(BasePermission):
    """Grants access if the user is a member of the project (any role)
    or is a system admin."""

    message = "You must be a member of this project."

    def has_permission(self, request: HttpRequest, controller) -> bool:
        user = request.user
        if not user or not user.is_authenticated:
            return False
        if _is_admin(user):
            return True
        project_id = request.resolver_match.kwargs.get("project_id")
        if project_id is None:
            return False
        return ProjectMembership.objects.filter(
            project_id=project_id, user=user,
        ).exists()


class IsProjectSupervisorOrAdmin(BasePermission):
    """Grants access if the user is a supervisor of the project
    or is a system admin."""

    message = "You must be a supervisor of this project."

    def has_permission(self, request: HttpRequest, controller) -> bool:
        user = request.user
        if not user or not user.is_authenticated:
            return False
        if _is_admin(user):
            return True
        project_id = request.resolver_match.kwargs.get("project_id")
        if project_id is None:
            return False
        return ProjectMembership.objects.filter(
            project_id=project_id, user=user, role="supervisor",
        ).exists()
