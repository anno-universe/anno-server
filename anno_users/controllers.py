from django.contrib.auth import get_user_model
from django.contrib.auth.models import Group
from ninja_extra import api_controller, http_get, http_patch, http_post
from ninja_extra.permissions import AllowAny, IsAuthenticated
from ninja_jwt.authentication import JWTAuth

from anno.pagination import PaginatedResponse, paginate_queryset

from .schemas import ProfileUpdateInput, RegisterInput, UserProfileOutput, UserSearchResult

User = get_user_model()


@api_controller("/users", tags=["users"])
class UserController:

    @http_post(
        "/register",
        permissions=[AllowAny],
        auth=None,
        response={201: UserProfileOutput},
        url_name="user_register",
    )
    def register(self, payload: RegisterInput):
        user = User.objects.create_user(
            username=payload.username,
            email=payload.email,
            password=payload.password,
        )
        user.groups.add(Group.objects.get(name="user"))
        return 201, self._user_to_profile(user)

    @http_get(
        "/me",
        permissions=[IsAuthenticated],
        auth=JWTAuth(),
        response={200: UserProfileOutput},
        url_name="user_profile",
    )
    def profile(self, request):
        return 200, self._user_to_profile(request.user)

    @http_patch(
        "/me",
        permissions=[IsAuthenticated],
        auth=JWTAuth(),
        response={200: UserProfileOutput},
        url_name="user_profile_update",
    )
    def update_profile(self, request, payload: ProfileUpdateInput):
        user = request.user
        for attr, value in payload.model_dump(exclude_unset=True).items():
            setattr(user, attr, value)
        user.save()
        return 200, self._user_to_profile(user)

    @http_get(
        "/search",
        permissions=[IsAuthenticated],
        auth=JWTAuth(),
        response={200: PaginatedResponse[UserSearchResult]},
        url_name="user_search",
    )
    def search_users(self, q: str, limit: int = 100, offset: int = 0):
        q = q.strip()
        if not q:
            return 200, PaginatedResponse(count=0, limit=limit, offset=offset, items=[])
        qs = User.objects.filter(username__icontains=q).order_by("username")
        count, limit, offset, rows = paginate_queryset(qs, limit, offset)
        return 200, PaginatedResponse(
            count=count,
            limit=limit,
            offset=offset,
            items=[UserSearchResult(id=u.id, username=u.username, email=u.email) for u in rows],
        )

    def _user_to_profile(self, user) -> dict:
        return {
            "id": user.id,
            "username": user.username,
            "email": user.email,
            "first_name": user.first_name,
            "last_name": user.last_name,
            "is_active": user.is_active,
            "groups": list(user.groups.values_list("name", flat=True)),
        }
