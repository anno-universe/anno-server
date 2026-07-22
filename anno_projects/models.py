import hashlib
import secrets

from django.conf import settings
from django.core.exceptions import ValidationError
from django.db import models
from django.utils import timezone

from anno.models import SoftDeleteModel


class Project(SoftDeleteModel):
    name = models.CharField(max_length=255)
    description = models.TextField(blank=True, default="")
    meta_info = models.JSONField(
        blank=True,
        default=dict,
        help_text="Arbitrary metadata about the dataset content.",
    )
    label_mapping = models.JSONField(
        blank=True,
        default=dict,
        help_text="Semantic label names to numeric class IDs, e.g. {'cat': 0, 'dog': 1}.",
    )
    created_by = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.PROTECT,
        related_name="projects",
    )
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta(SoftDeleteModel.Meta):
        db_table = "anno_project"
        ordering = ["-created_at"]
        verbose_name = "project"
        verbose_name_plural = "projects"

    def __str__(self):
        return self.name

    def get_user_role(self, user) -> str | None:
        """Return the role string for a user in this project, or None."""
        if user and user.is_authenticated and user.groups.filter(name="admin").exists():
            return "admin"
        if hasattr(self, "_prefetched_memberships_cache"):
            for m in self._prefetched_memberships_cache:
                if m.user_id == user.id:
                    return m.role
        try:
            return self.memberships.get(user=user).role
        except ProjectMembership.DoesNotExist:
            return None

    def get_membership_role(self, user) -> str | None:
        """Return the user's ProjectMembership role in this project, or None.

        Unlike :meth:`get_user_role`, this does NOT treat the Django ``admin``
        group as a role — an admin is handled by their actual project membership
        (or lack thereof) by the caller. Used by the tag-apply check so an admin
        who joined a project as a worker/supervisor is bound by that role.
        """
        try:
            return self.memberships.get(user=user).role
        except ProjectMembership.DoesNotExist:
            return None


class ProjectMembership(SoftDeleteModel):
    ROLE_CHOICES = [
        ("worker", "Worker"),
        ("supervisor", "Supervisor"),
    ]

    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="project_memberships",
    )
    project = models.ForeignKey(
        Project,
        on_delete=models.CASCADE,
        related_name="memberships",
    )
    role = models.CharField(max_length=20, choices=ROLE_CHOICES)
    added_by = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.PROTECT,
        related_name="added_memberships",
        null=True,
        blank=True,
    )
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta(SoftDeleteModel.Meta):
        db_table = "anno_project_membership"
        ordering = ["created_at"]
        verbose_name = "project membership"
        verbose_name_plural = "project memberships"
        constraints = [
            # Only alive memberships are unique, so a removed member can be
            # re-added.
            models.UniqueConstraint(
                fields=["user", "project"],
                condition=models.Q(deleted_at__isnull=True),
                name="uniq_active_membership",
            ),
        ]

    def __str__(self):
        return f"{self.user.username} is {self.role} in {self.project.name}"

    def delete(self, using=None, keep_parents=False):
        # Preserve the last-supervisor rule here (it used to live in a
        # pre_delete signal, which no longer fires on soft delete). Uses the
        # alive-only default manager so a previously removed supervisor does
        # not count as remaining.
        if self.role == "supervisor":
            remaining = (
                ProjectMembership.objects.filter(
                    project=self.project, role="supervisor"
                )
                .exclude(pk=self.pk)
                .exists()
            )
            if not remaining:
                raise ValidationError(
                    "Cannot remove the last supervisor from a project."
                )
        return super().delete(using=using, keep_parents=keep_parents)


class ProjectAPIKey(SoftDeleteModel):
    """A per-project API key used by external Project API clients.

    Workers authenticate with the plaintext token via the ``X-API-Key`` header;
    only a SHA-256 hash is persisted. The ``prefix`` is a non-secret leading
    segment kept in clear for fast lookup and display.
    """

    project = models.ForeignKey(
        Project,
        on_delete=models.CASCADE,
        related_name="api_keys",
    )
    name = models.CharField(
        max_length=255,
        help_text="Human label, e.g. 'gpu-box-01'.",
    )
    prefix = models.CharField(
        max_length=12,
        db_index=True,
        help_text="Non-secret leading segment of the token, for lookup and display.",
    )
    key_hash = models.CharField(
        max_length=64,
        unique=True,
        help_text="SHA-256 hex digest of the full token. The plaintext is never stored.",
    )
    created_by = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.PROTECT,
        related_name="created_api_keys",
    )
    is_active = models.BooleanField(default=True)
    expires_at = models.DateTimeField(null=True, blank=True)
    last_used_at = models.DateTimeField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta(SoftDeleteModel.Meta):
        db_table = "anno_project_api_key"
        ordering = ["-created_at"]
        verbose_name = "project API key"
        verbose_name_plural = "project API keys"
        indexes = [
            models.Index(fields=["project", "is_active"]),
            models.Index(fields=["prefix"]),
        ]

    def __str__(self):
        return f"{self.name} ({self.prefix}…)"

    # ----- token helpers -----

    @staticmethod
    def hash_token(token: str) -> str:
        return hashlib.sha256(token.encode()).hexdigest()

    @classmethod
    def generate(cls, *, project, name, created_by, expires_at=None):
        """Build (but do not save) a key, returning ``(instance, plaintext_token)``.

        The plaintext token is shown to the caller exactly once; only its hash is
        stored on the instance.
        """
        prefix = "ak_" + secrets.token_hex(4)
        secret = secrets.token_urlsafe(32)
        token = f"{prefix}.{secret}"
        instance = cls(
            project=project,
            name=name,
            prefix=prefix,
            key_hash=cls.hash_token(token),
            created_by=created_by,
            expires_at=expires_at,
        )
        return instance, token

    def is_usable(self) -> bool:
        """True if the key is active and not past its expiry."""
        if not self.is_active:
            return False
        if self.expires_at is not None and self.expires_at <= timezone.now():
            return False
        return True
