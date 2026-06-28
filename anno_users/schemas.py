from pydantic import field_validator
from ninja import Schema
from django.contrib.auth import get_user_model

User = get_user_model()


class RegisterInput(Schema):
    username: str
    email: str
    password: str

    @field_validator("username")
    @classmethod
    def username_unique(cls, v: str) -> str:
        if User.objects.filter(username=v).exists():
            raise ValueError("A user with that username already exists.")
        return v

    @field_validator("email")
    @classmethod
    def email_unique(cls, v: str) -> str:
        if User.objects.filter(email=v).exists():
            raise ValueError("A user with that email already exists.")
        return v


class UserProfileOutput(Schema):
    id: int
    username: str
    email: str
    first_name: str
    last_name: str
    is_active: bool
    groups: list[str]


class ProfileUpdateInput(Schema):
    first_name: str | None = None
    last_name: str | None = None
    email: str | None = None


class UserSearchResult(Schema):
    id: int
    username: str
    email: str
