"""Soft-delete behaviour for inference providers.

Covers both the reusable ``SoftDeleteModel`` base (exercised through the
concrete provider models) and the original bug: deleting a provider that has
protected history (``InferenceRun`` / ``InteractiveInferenceSession`` with
``on_delete=PROTECT``) used to raise ``ProtectedError`` → 500. With soft-delete
no SQL DELETE is issued, so it now returns 204 and the history is preserved.
"""

import json

from django.contrib.auth import get_user_model
from django.test import Client, TestCase
from ninja_jwt.tokens import RefreshToken

from anno_projects.models import Project, ProjectMembership
from anno_images.models import Image2D
from django.core.files.base import ContentFile

from anno_infers.models import (
    InferenceRun,
    InferenceServiceProvider,
    InteractiveInferenceServiceProvider,
    InteractiveInferenceSession,
)

User = get_user_model()


def _auth(user):
    return {"Authorization": f"Bearer {RefreshToken.for_user(user).access_token}"}


class SoftDeleteBaseTest(TestCase):
    """The SoftDeleteModel base contract, via InferenceServiceProvider."""

    def setUp(self):
        self.user = User.objects.create_user(username="u", password="x")

    def _make(self, **kw):
        defaults = dict(
            name="p",
            inference_url="http://svc.local",
            supported_result_types=["box"],
            created_by=self.user,
        )
        defaults.update(kw)
        return InferenceServiceProvider.objects.create(**defaults)

    def test_objects_excludes_all_objects_includes(self):
        p = self._make()
        p.delete()  # soft
        self.assertFalse(InferenceServiceProvider.objects.filter(pk=p.pk).exists())
        self.assertTrue(InferenceServiceProvider.all_objects.filter(pk=p.pk).exists())

    def test_delete_stamps_deleted_at_without_removing_row(self):
        p = self._make()
        result = p.delete()
        p.refresh_from_db()
        self.assertIsNotNone(p.deleted_at)
        self.assertTrue(p.is_deleted)
        self.assertEqual(result, (1, {InferenceServiceProvider._meta.label: 1}))

    def test_alive_and_dead_partition(self):
        alive = self._make()
        dead = self._make()
        dead.delete()
        self.assertEqual(
            set(InferenceServiceProvider.all_objects.alive().values_list("pk", flat=True)),
            {alive.pk},
        )
        self.assertEqual(
            set(InferenceServiceProvider.all_objects.dead().values_list("pk", flat=True)),
            {dead.pk},
        )

    def test_restore(self):
        p = self._make()
        p.delete()
        p.restore()
        self.assertTrue(InferenceServiceProvider.objects.filter(pk=p.pk).exists())
        p.refresh_from_db()
        self.assertIsNone(p.deleted_at)

    def test_queryset_bulk_soft_delete(self):
        a, b = self._make(), self._make()
        InferenceServiceProvider.objects.filter(pk__in=[a.pk, b.pk]).delete()
        self.assertEqual(InferenceServiceProvider.objects.count(), 0)
        self.assertEqual(InferenceServiceProvider.all_objects.count(), 2)

    def test_hard_delete_removes_row(self):
        p = self._make()
        p.hard_delete()
        self.assertFalse(InferenceServiceProvider.all_objects.filter(pk=p.pk).exists())


class ProviderDeleteEndpointTest(TestCase):
    """The original 500 bug, end to end, for both provider kinds."""

    def setUp(self):
        self.client = Client()
        self.supervisor = User.objects.create_user(username="sup", password="x")
        self.project = Project.objects.create(name="P", created_by=self.supervisor)
        # creator auto-added as supervisor member by signal.
        self.image = Image2D.objects.create(
            project=self.project,
            image=ContentFile(b"\x89PNG\r\n\x1a\n-x", name="a.png"),
            width=8,
            height=8,
        )

    def test_delete_provider_with_protected_run_returns_204(self):
        provider = InferenceServiceProvider.objects.create(
            project=self.project,
            name="p",
            inference_url="http://svc.local",
            supported_result_types=["box"],
            created_by=self.supervisor,
        )
        run = InferenceRun.objects.create(
            project=self.project, provider=provider, created_by=self.supervisor
        )

        res = self.client.delete(
            f"/api/projects/{self.project.id}/inference-providers/{provider.id}",
            headers=_auth(self.supervisor),
        )

        self.assertEqual(res.status_code, 204)
        # Soft-deleted: hidden from the API, still present for history.
        self.assertFalse(
            InferenceServiceProvider.objects.filter(pk=provider.pk).exists()
        )
        provider_all = InferenceServiceProvider.all_objects.get(pk=provider.pk)
        self.assertIsNotNone(provider_all.deleted_at)
        # The protected run's FK still resolves (proves base_manager_name fix).
        run.refresh_from_db()
        self.assertEqual(run.provider_id, provider.pk)
        self.assertEqual(run.provider.name, "p")
        # Gone from list; detail 404s.
        list_res = self.client.get(
            f"/api/projects/{self.project.id}/inference-providers/",
            headers=_auth(self.supervisor),
        )
        self.assertEqual(list_res.status_code, 200)
        self.assertNotIn(
            provider.id, [row["id"] for row in list_res.json()["items"]]
        )
        detail_res = self.client.get(
            f"/api/projects/{self.project.id}/inference-providers/{provider.id}",
            headers=_auth(self.supervisor),
        )
        self.assertEqual(detail_res.status_code, 404)

    def test_delete_interactive_provider_with_protected_session_returns_204(self):
        provider = InteractiveInferenceServiceProvider.objects.create(
            project=self.project,
            name="sam",
            inference_url="http://svc.local",
            supported_prompt_types=["box"],
            supported_result_types=["polygon"],
            created_by=self.supervisor,
        )
        session = InteractiveInferenceSession.objects.create(
            project=self.project,
            image=self.image,
            provider=provider,
            performed_by=self.supervisor,
            session_token="tok_sd_1",
        )

        res = self.client.delete(
            f"/api/projects/{self.project.id}/interactive-providers/{provider.id}",
            headers=_auth(self.supervisor),
        )

        self.assertEqual(res.status_code, 204)
        self.assertFalse(
            InteractiveInferenceServiceProvider.objects.filter(pk=provider.pk).exists()
        )
        session.refresh_from_db()
        self.assertEqual(session.provider_id, provider.pk)
        self.assertEqual(session.provider.name, "sam")
