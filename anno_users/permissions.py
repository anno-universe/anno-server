from django.http import HttpRequest
from ninja_extra.permissions import BasePermission


class IsAdminGroup(BasePermission):
    message = "You must be a system administrator."

    def has_permission(self, request: HttpRequest, controller) -> bool:
        user = request.user
        return bool(
            user and user.is_authenticated and user.groups.filter(name="admin").exists()
        )
