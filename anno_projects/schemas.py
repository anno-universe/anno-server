from datetime import datetime

from ninja import Schema

# ---------- Project ----------


class ProjectCreateInput(Schema):
    name: str
    description: str = ""
    meta_info: dict = {}
    label_mapping: dict = {}


class ProjectUpdateInput(Schema):
    name: str | None = None
    description: str | None = None
    meta_info: dict | None = None
    label_mapping: dict | None = None


class ProjectOutput(Schema):
    id: int
    name: str
    description: str
    meta_info: dict
    label_mapping: dict
    created_by_id: int
    my_role: str | None
    created_at: datetime
    updated_at: datetime

    @staticmethod
    def from_project(project, role: str | None = None) -> "ProjectOutput":
        return ProjectOutput(
            id=project.id,
            name=project.name,
            description=project.description,
            meta_info=project.meta_info,
            label_mapping=project.label_mapping,
            created_by_id=project.created_by_id,
            my_role=role,
            created_at=project.created_at,
            updated_at=project.updated_at,
        )


class ProjectMemberOutput(Schema):
    user_id: int
    username: str
    email: str
    role: str
    created_at: datetime

    @staticmethod
    def from_membership(m) -> "ProjectMemberOutput":
        return ProjectMemberOutput(
            user_id=m.user_id,
            username=m.user.username,
            email=m.user.email,
            role=m.role,
            created_at=m.created_at,
        )


class AddProjectMemberInput(Schema):
    user_id: int
    role: str


class UpdateMemberRoleInput(Schema):
    role: str


# ---------- API keys ----------


class APIKeyCreateInput(Schema):
    name: str
    expires_at: datetime | None = None


class APIKeyUpdateInput(Schema):
    name: str | None = None
    is_active: bool | None = None
    expires_at: datetime | None = None


class APIKeyOutput(Schema):
    id: int
    project_id: int
    name: str
    prefix: str
    is_active: bool
    expires_at: datetime | None
    last_used_at: datetime | None
    created_by_id: int
    created_at: datetime

    @staticmethod
    def from_api_key(key) -> "APIKeyOutput":
        return APIKeyOutput(
            id=key.id,
            project_id=key.project_id,
            name=key.name,
            prefix=key.prefix,
            is_active=key.is_active,
            expires_at=key.expires_at,
            last_used_at=key.last_used_at,
            created_by_id=key.created_by_id,
            created_at=key.created_at,
        )


class APIKeyCreatedOutput(APIKeyOutput):
    """Returned only at creation time — the one response that carries the
    plaintext token. Shown to the caller exactly once."""

    token: str

    @staticmethod
    def from_api_key_with_token(key, token: str) -> "APIKeyCreatedOutput":
        return APIKeyCreatedOutput(
            id=key.id,
            project_id=key.project_id,
            name=key.name,
            prefix=key.prefix,
            is_active=key.is_active,
            expires_at=key.expires_at,
            last_used_at=key.last_used_at,
            created_by_id=key.created_by_id,
            created_at=key.created_at,
            token=token,
        )
