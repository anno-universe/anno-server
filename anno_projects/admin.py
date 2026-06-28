from django.contrib import admin, messages
from django.core.exceptions import PermissionDenied
from django.utils.html import format_html

from .models import Project, ProjectAPIKey, ProjectMembership


@admin.register(Project)
class ProjectAdmin(admin.ModelAdmin):
    list_display = ["id", "name", "created_by", "created_at", "updated_at"]
    list_filter = ["created_at"]
    search_fields = ["name", "description"]
    readonly_fields = ["created_at", "updated_at"]


@admin.register(ProjectMembership)
class ProjectMembershipAdmin(admin.ModelAdmin):
    list_display = ["id", "user", "project", "role", "added_by", "created_at"]
    list_filter = ["role", "project", "created_at"]
    search_fields = ["user__username", "project__name"]
    readonly_fields = ["created_at", "updated_at"]


@admin.register(ProjectAPIKey)
class ProjectAPIKeyAdmin(admin.ModelAdmin):
    list_display = [
        "id",
        "name",
        "project",
        "prefix",
        "key_hash",
        "is_active",
        "expires_at",
        "last_used_at",
        "created_by",
        "created_at",
        "updated_at",
    ]
    list_filter = ["is_active", "project", "created_at"]
    search_fields = ["name", "prefix", "project__name"]
    readonly_fields = [
        "prefix",
        "key_hash",
        "token_display",
        "last_used_at",
        "created_at",
        "updated_at",
    ]

    _request = None

    def get_readonly_fields(self, request, obj=None):
        if obj is None:
            return ["last_used_at", "created_at", "updated_at"]
        return self.readonly_fields + ["created_by"]

    def get_fields(self, request, obj=None):
        if obj is None:
            return ["project", "name", "created_by", "expires_at"]
        return super().get_fields(request, obj)

    def formfield_for_foreignkey(self, db_field, request, **kwargs):
        if db_field.name == "project" and not request.user.is_superuser:
            kwargs["queryset"] = Project.objects.filter(
                memberships__user=request.user,
                memberships__role="supervisor",
            ).distinct()
        return super().formfield_for_foreignkey(db_field, request, **kwargs)

    def save_model(self, request, obj, form, change):
        if not change:
            if not request.user.is_superuser and not request.user.groups.filter(name="admin").exists():
                is_supervisor = ProjectMembership.objects.filter(
                    project=obj.project,
                    user=request.user,
                    role="supervisor",
                ).exists()
                if not is_supervisor:
                    raise PermissionDenied("只有项目 supervisor 才能创建 API Key。")
            instance, token = ProjectAPIKey.generate(
                project=obj.project,
                name=obj.name,
                created_by=obj.created_by or request.user,
                expires_at=obj.expires_at,
            )
            instance.save()
            request.session[f"_api_key_token_{instance.pk}"] = token
            messages.success(request, "API Key 已创建，Token 见下方。")
        else:
            super().save_model(request, obj, form, change)

    def change_view(self, request, object_id, form_url="", extra_context=None):
        self._request = request
        return super().change_view(request, object_id, form_url, extra_context)

    def token_display(self, obj):
        token = None
        if self._request:
            token = self._request.session.pop(f"_api_key_token_{obj.pk}", None)
        if token:
            return format_html(
                '<code style="font-size:14px;user-select:all;">{}</code>'
                '<p style="color:red;">⚠ 此 Token 仅显示一次，请立即保存！刷新后消失。</p>',
                token,
            )
        return "（仅创建时显示一次，已不可见）"

    token_display.short_description = "完整 Token"
