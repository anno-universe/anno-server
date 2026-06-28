from datetime import timedelta

from django.utils import timezone
from ninja.security import APIKeyHeader

from anno_projects.models import ProjectAPIKey

# Avoid a DB write on every authenticated request — only refresh last_used_at
# when it is older than this.
_LAST_USED_THROTTLE = timedelta(seconds=60)


class ProjectAPIKeyAuth(APIKeyHeader):
    """Authenticate inference workers by a per-project API key (Flow A).

    The full token is sent in the ``X-API-Key`` header; we hash it and look the
    key up by hash. On success the resolved key and its project are stashed on
    the request (``request.api_key`` / ``request.project``) — ``request.user``
    stays anonymous, so endpoints must read those attributes, not ``request.user``.
    """

    param_name = "X-API-Key"

    def authenticate(self, request, key):
        if not key:
            return None
        digest = ProjectAPIKey.hash_token(key)
        api_key = (
            ProjectAPIKey.objects.select_related("project", "created_by")
            .filter(key_hash=digest, is_active=True)
            .first()
        )
        if api_key is None or not api_key.is_usable():
            return None

        now = timezone.now()
        if (
            api_key.last_used_at is None
            or now - api_key.last_used_at > _LAST_USED_THROTTLE
        ):
            ProjectAPIKey.objects.filter(pk=api_key.pk).update(last_used_at=now)
            api_key.last_used_at = now

        request.api_key = api_key
        request.project = api_key.project
        return api_key
