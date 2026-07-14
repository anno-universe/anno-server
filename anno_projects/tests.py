import json
from datetime import timedelta

from django.contrib.auth import get_user_model
from django.core.exceptions import ValidationError
from django.test import Client, TestCase
from django.utils import timezone
from ninja_jwt.tokens import RefreshToken

from anno_projects.models import Project, ProjectAPIKey, ProjectMembership

User = get_user_model()


def _jwt_headers(user):
    access = RefreshToken.for_user(user).access_token
    return {"Authorization": f"Bearer {access}"}


class TokenHelperTests(TestCase):
    def test_generate_and_hash_roundtrip(self):
        user = User.objects.create_user(username="u1", password="x")
        project = Project.objects.create(name="P", created_by=user)
        inst, token = ProjectAPIKey.generate(project=project, name="k", created_by=user)
        inst.save()
        self.assertTrue(token.startswith(inst.prefix + "."))
        self.assertEqual(ProjectAPIKey.hash_token(token), inst.key_hash)
        self.assertEqual(len(inst.key_hash), 64)
        # The plaintext token is never persisted.
        self.assertNotEqual(token, inst.key_hash)

    def test_is_usable(self):
        user = User.objects.create_user(username="u2", password="x")
        project = Project.objects.create(name="P", created_by=user)
        inst, _ = ProjectAPIKey.generate(project=project, name="k", created_by=user)
        self.assertTrue(inst.is_usable())
        inst.is_active = False
        self.assertFalse(inst.is_usable())
        inst.is_active = True
        inst.expires_at = timezone.now() - timedelta(seconds=1)
        self.assertFalse(inst.is_usable())
        inst.expires_at = timezone.now() + timedelta(days=1)
        self.assertTrue(inst.is_usable())


class APIKeyManagementTests(TestCase):
    def setUp(self):
        self.client = Client()
        self.supervisor = User.objects.create_user(username="sup", password="x")
        self.outsider = User.objects.create_user(username="out", password="x")
        self.project = Project.objects.create(name="P", created_by=self.supervisor)

    def _url(self, suffix=""):
        return f"/api/projects/{self.project.id}/api-keys/{suffix}"

    def test_create_key_returns_token_once_then_hidden(self):
        res = self.client.post(
            self._url(),
            data=json.dumps({"name": "gpu"}),
            content_type="application/json",
            headers=_jwt_headers(self.supervisor),
        )
        self.assertEqual(res.status_code, 201)
        body = res.json()
        self.assertIn("token", body)
        self.assertTrue(body["token"].startswith(body["prefix"] + "."))

        # The token is never returned again on listing.
        res2 = self.client.get(self._url(), headers=_jwt_headers(self.supervisor))
        self.assertEqual(res2.status_code, 200)
        data = res2.json()
        self.assertEqual(data["count"], 1)
        listed = data["items"]
        self.assertEqual(len(listed), 1)
        self.assertNotIn("token", listed[0])

    def test_non_supervisor_cannot_create_key(self):
        res = self.client.post(
            self._url(),
            data=json.dumps({"name": "gpu"}),
            content_type="application/json",
            headers=_jwt_headers(self.outsider),
        )
        self.assertIn(res.status_code, (401, 403))
        self.assertEqual(ProjectAPIKey.objects.count(), 0)

    def test_revoke_then_worker_unauthorized(self):
        inst, token = ProjectAPIKey.generate(
            project=self.project, name="k", created_by=self.supervisor
        )
        inst.save()
        # Revoke via the management API.
        res = self.client.patch(
            self._url(f"{inst.id}"),
            data=json.dumps({"is_active": False}),
            content_type="application/json",
            headers=_jwt_headers(self.supervisor),
        )
        self.assertEqual(res.status_code, 200)
        self.assertFalse(res.json()["is_active"])
        # The worker can no longer authenticate.
        res2 = self.client.get(
            "/api/infers/project/images", headers={"X-API-Key": token}
        )
        self.assertEqual(res2.status_code, 401)


class ProjectSoftDeleteTests(TestCase):
    def setUp(self):
        self.client = Client()
        self.supervisor = User.objects.create_user(username="sup", password="x")
        self.worker = User.objects.create_user(username="wk", password="x")
        self.project = Project.objects.create(name="P", created_by=self.supervisor)
        ProjectMembership.objects.create(
            user=self.worker, project=self.project, role="worker"
        )

    def test_soft_delete_project_hides_and_blocks_access(self):
        res = self.client.delete(
            f"/api/projects/{self.project.id}",
            headers=_jwt_headers(self.supervisor),
        )
        self.assertEqual(res.status_code, 204)
        self.assertFalse(Project.objects.filter(pk=self.project.pk).exists())
        self.assertTrue(Project.all_objects.filter(pk=self.project.pk).exists())

        # Gone from the worker's project list.
        lst = self.client.get("/api/projects/", headers=_jwt_headers(self.worker))
        self.assertEqual(lst.status_code, 200)
        self.assertNotIn(
            self.project.id, [p["id"] for p in lst.json()["items"]]
        )
        # A member is denied on the project's child endpoints.
        detail = self.client.get(
            f"/api/projects/{self.project.id}",
            headers=_jwt_headers(self.worker),
        )
        self.assertIn(detail.status_code, (403, 404))

    def test_delete_last_supervisor_blocked(self):
        membership = ProjectMembership.objects.get(
            project=self.project, user=self.supervisor
        )
        with self.assertRaises(ValidationError):
            membership.delete()
        membership.refresh_from_db()
        self.assertIsNone(membership.deleted_at)

    def test_remove_non_last_supervisor_then_readd(self):
        sup2 = User.objects.create_user(username="sup2", password="x")
        ProjectMembership.objects.create(
            project=self.project, user=sup2, role="supervisor"
        )
        remove = self.client.delete(
            f"/api/projects/{self.project.id}/members/{sup2.id}",
            headers=_jwt_headers(self.supervisor),
        )
        self.assertEqual(remove.status_code, 204)
        self.assertFalse(
            ProjectMembership.objects.filter(project=self.project, user=sup2).exists()
        )
        # Re-adding the same user succeeds (partial-unique on alive rows).
        readd = self.client.post(
            f"/api/projects/{self.project.id}/members",
            data=json.dumps({"user_id": sup2.id, "role": "supervisor"}),
            content_type="application/json",
            headers=_jwt_headers(self.supervisor),
        )
        self.assertEqual(readd.status_code, 201)
        self.assertEqual(
            ProjectMembership.objects.filter(project=self.project, user=sup2).count(),
            1,
        )

    def test_soft_delete_api_key_hidden_and_unusable(self):
        inst, token = ProjectAPIKey.generate(
            project=self.project, name="k", created_by=self.supervisor
        )
        inst.save()
        res = self.client.delete(
            f"/api/projects/{self.project.id}/api-keys/{inst.id}",
            headers=_jwt_headers(self.supervisor),
        )
        self.assertEqual(res.status_code, 204)
        self.assertFalse(ProjectAPIKey.objects.filter(pk=inst.pk).exists())
        # A soft-deleted key can no longer authenticate.
        res2 = self.client.get(
            "/api/infers/project/images", headers={"X-API-Key": token}
        )
        self.assertEqual(res2.status_code, 401)
