"""Tests for server-driven auto-annotation (Flow B).

Covers the provider registry, the supervisor-triggered auto-annotate endpoints,
and the background worker (with the provider HTTP call mocked).
"""

import json
import tempfile
from unittest.mock import MagicMock, patch

from django.contrib.auth import get_user_model
from django.core.files.base import ContentFile
from django.test import Client, TestCase, override_settings

from anno_images.models import Annotation2D, Image2D, Operation
from anno_projects.models import Project, ProjectMembership
from anno_sdk import Annotation, Box2D, InferenceResponse, Mask2D
from ninja_jwt.tokens import RefreshToken

from anno_infers.models import (
    InferenceJob,
    InferenceJobItem,
    InferenceServiceProvider,
)
from anno_infers.tasks import execute_inference_job

User = get_user_model()

_TMP_MEDIA = tempfile.mkdtemp(prefix="anno-flowb-test-")

_TEST_STORAGES = {
    "default": {
        "BACKEND": "django.core.files.storage.FileSystemStorage",
        "OPTIONS": {"location": _TMP_MEDIA},
    },
    "staticfiles": {
        "BACKEND": "django.contrib.staticfiles.storage.StaticFilesStorage",
    },
}


def _auth(user):
    return {"Authorization": f"Bearer {RefreshToken.for_user(user).access_token}"}


def _fake_response(payload: dict) -> MagicMock:
    resp = MagicMock()
    resp.raise_for_status.return_value = None
    resp.json.return_value = payload
    return resp


@override_settings(STORAGES=_TEST_STORAGES)
class FlowBBaseTest(TestCase):
    def setUp(self):
        self.client = Client()
        self.supervisor = User.objects.create_user(username="sup", password="x")
        self.worker = User.objects.create_user(username="wrk", password="x")
        self.outsider = User.objects.create_user(username="out", password="x")

        self.project = Project.objects.create(
            name="P",
            label_mapping={"cat": 0, "dog": 1},
            created_by=self.supervisor,
        )
        # supervisor membership auto-created by signal; add a worker.
        ProjectMembership.objects.create(
            user=self.worker, project=self.project, role="worker"
        )
        self.other_project = Project.objects.create(name="Q", created_by=self.supervisor)

    def _make_image(self, project, name="a.png"):
        return Image2D.objects.create(
            project=project,
            image=ContentFile(b"\x89PNG\r\n\x1a\n-fake-bytes", name=name),
            width=64,
            height=64,
        )

    def _make_provider(self, *, project=None, types=("box", "polygon"), **kw):
        defaults = dict(
            name="prov",
            inference_url="http://svc.local/predict",
            supported_result_types=list(types),
            created_by=self.supervisor,
        )
        defaults.update(kw)
        return InferenceServiceProvider.objects.create(project=project, **defaults)


# ---------------------------------------------------------------------------
# Provider registry
# ---------------------------------------------------------------------------


class ProviderRegistryTests(FlowBBaseTest):
    def test_create_provider_supervisor(self):
        body = {
            "name": "SAM",
            "inference_url": "http://svc/predict",
            "supported_result_types": ["box"],
            "auth_type": "header",
            "auth_param_name": "X-API-Key",
            "auth_secret": "s3cr3t",
        }
        res = self.client.post(
            f"/api/projects/{self.project.id}/inference-providers/",
            data=json.dumps(body),
            content_type="application/json",
            headers=_auth(self.supervisor),
        )
        self.assertEqual(res.status_code, 201, res.content)
        data = res.json()
        # Secret is never serialized; a boolean flag reports its presence.
        self.assertNotIn("auth_secret", data)
        self.assertTrue(data["has_auth_secret"])
        self.assertFalse(data["is_global"])
        prov = InferenceServiceProvider.objects.get(id=data["id"])
        self.assertEqual(prov.auth_secret, "s3cr3t")
        self.assertEqual(prov.project_id, self.project.id)

    def test_create_provider_worker_forbidden(self):
        body = {
            "name": "x",
            "inference_url": "http://svc/predict",
            "supported_result_types": ["box"],
        }
        res = self.client.post(
            f"/api/projects/{self.project.id}/inference-providers/",
            data=json.dumps(body),
            content_type="application/json",
            headers=_auth(self.worker),
        )
        self.assertEqual(res.status_code, 403)

    def test_invalid_result_type_rejected(self):
        body = {
            "name": "x",
            "inference_url": "http://svc/predict",
            "supported_result_types": ["bogus"],
        }
        res = self.client.post(
            f"/api/projects/{self.project.id}/inference-providers/",
            data=json.dumps(body),
            content_type="application/json",
            headers=_auth(self.supervisor),
        )
        self.assertEqual(res.status_code, 422)

    def test_list_includes_global_and_project_excludes_other(self):
        mine = self._make_provider(project=self.project, name="mine")
        glob = self._make_provider(project=None, name="global")
        theirs = self._make_provider(project=self.other_project, name="theirs")

        res = self.client.get(
            f"/api/projects/{self.project.id}/inference-providers/",
            headers=_auth(self.worker),
        )
        self.assertEqual(res.status_code, 200)
        ids = {p["id"] for p in res.json()["items"]}
        self.assertIn(mine.id, ids)
        self.assertIn(glob.id, ids)
        self.assertNotIn(theirs.id, ids)

    def test_cannot_edit_global_via_api(self):
        glob = self._make_provider(project=None, name="global")
        res = self.client.patch(
            f"/api/projects/{self.project.id}/inference-providers/{glob.id}",
            data=json.dumps({"name": "hijack"}),
            content_type="application/json",
            headers=_auth(self.supervisor),
        )
        self.assertEqual(res.status_code, 404)
        glob.refresh_from_db()
        self.assertEqual(glob.name, "global")

    def test_update_and_delete_project_provider(self):
        prov = self._make_provider(project=self.project)
        res = self.client.patch(
            f"/api/projects/{self.project.id}/inference-providers/{prov.id}",
            data=json.dumps({"is_active": False, "auth_secret": "new"}),
            content_type="application/json",
            headers=_auth(self.supervisor),
        )
        self.assertEqual(res.status_code, 200)
        self.assertFalse(res.json()["is_active"])
        prov.refresh_from_db()
        self.assertEqual(prov.auth_secret, "new")

        res = self.client.delete(
            f"/api/projects/{self.project.id}/inference-providers/{prov.id}",
            headers=_auth(self.supervisor),
        )
        self.assertEqual(res.status_code, 204)
        self.assertFalse(InferenceServiceProvider.objects.filter(id=prov.id).exists())


# ---------------------------------------------------------------------------
# Auto-annotate endpoint (job creation; worker not run)
# ---------------------------------------------------------------------------


class AutoAnnotateStartTests(FlowBBaseTest):
    def test_start_requires_supervisor(self):
        prov = self._make_provider(project=self.project)
        self._make_image(self.project)
        res = self.client.post(
            f"/api/projects/{self.project.id}/auto-annotate/",
            data=json.dumps({"provider_id": prov.id}),
            content_type="application/json",
            headers=_auth(self.worker),
        )
        self.assertEqual(res.status_code, 403)

    def test_start_creates_job_and_items(self):
        prov = self._make_provider(project=self.project)
        img1 = self._make_image(self.project, "a.png")
        img2 = self._make_image(self.project, "b.png")
        res = self.client.post(
            f"/api/projects/{self.project.id}/auto-annotate/",
            data=json.dumps({"provider_id": prov.id}),
            content_type="application/json",
            headers=_auth(self.supervisor),
        )
        self.assertEqual(res.status_code, 201, res.content)
        body = res.json()
        self.assertEqual(body["status"], "pending")
        self.assertEqual(body["total_items"], 2)
        job = InferenceJob.objects.get(id=body["id"])
        self.assertEqual(
            set(job.items.values_list("image_id", flat=True)), {img1.id, img2.id}
        )

    def test_start_only_unannotated(self):
        prov = self._make_provider(project=self.project)
        annotated = self._make_image(self.project, "x.png")
        Annotation2D.objects.create(
            image=annotated, project=self.project, annotation_type="box", label=0
        )
        fresh = self._make_image(self.project, "y.png")
        res = self.client.post(
            f"/api/projects/{self.project.id}/auto-annotate/",
            data=json.dumps({"provider_id": prov.id, "only_unannotated": True}),
            content_type="application/json",
            headers=_auth(self.supervisor),
        )
        self.assertEqual(res.status_code, 201)
        job = InferenceJob.objects.get(id=res.json()["id"])
        self.assertEqual(
            list(job.items.values_list("image_id", flat=True)), [fresh.id]
        )

    def test_start_no_images_400(self):
        prov = self._make_provider(project=self.project)
        res = self.client.post(
            f"/api/projects/{self.project.id}/auto-annotate/",
            data=json.dumps({"provider_id": prov.id}),
            content_type="application/json",
            headers=_auth(self.supervisor),
        )
        self.assertEqual(res.status_code, 400)

    def test_start_inactive_provider_404(self):
        prov = self._make_provider(project=self.project, is_active=False)
        self._make_image(self.project)
        res = self.client.post(
            f"/api/projects/{self.project.id}/auto-annotate/",
            data=json.dumps({"provider_id": prov.id}),
            content_type="application/json",
            headers=_auth(self.supervisor),
        )
        self.assertEqual(res.status_code, 404)


# ---------------------------------------------------------------------------
# Worker (execute_inference_job, provider HTTP mocked)
# ---------------------------------------------------------------------------


class WorkerTests(FlowBBaseTest):
    def _make_job(self, provider, images):
        job = InferenceJob.objects.create(
            project=self.project, provider=provider, created_by=self.supervisor,
            total_items=len(images),
        )
        InferenceJobItem.objects.bulk_create(
            [InferenceJobItem(job=job, image=img) for img in images]
        )
        return job

    def test_worker_writes_annotations(self):
        prov = self._make_provider(project=self.project, types=("box", "polygon"))
        img = self._make_image(self.project)
        job = self._make_job(prov, [img])

        payload = InferenceResponse(
            annotations=[
                Annotation(label=0, geometry=Box2D(1, 2, 3, 4)),
                Annotation(label=1, geometry=Mask2D([[0, 0], [1, 1], [2, 0]])),
            ]
        ).to_dict()

        with patch("anno_infers.tasks.httpx.post", return_value=_fake_response(payload)):
            execute_inference_job(job.id)

        job.refresh_from_db()
        self.assertEqual(job.status, "completed")
        self.assertEqual(job.completed_items, 1)
        self.assertEqual(job.annotations_created, 2)
        self.assertEqual(Annotation2D.objects.filter(image=img).count(), 2)
        self.assertEqual(Operation.objects.filter(image=img, action="add").count(), 2)
        item = job.items.get()
        self.assertEqual(item.status, "done")
        self.assertEqual(item.attempts, 1)

    def test_worker_drops_unsupported_type(self):
        prov = self._make_provider(project=self.project, types=("box",))
        img = self._make_image(self.project)
        job = self._make_job(prov, [img])

        payload = InferenceResponse(
            annotations=[
                Annotation(label=0, geometry=Box2D(1, 2, 3, 4)),
                Annotation(label=1, geometry=Mask2D([[0, 0], [1, 1]])),  # polygon, unsupported
            ]
        ).to_dict()

        with patch("anno_infers.tasks.httpx.post", return_value=_fake_response(payload)):
            execute_inference_job(job.id)

        self.assertEqual(Annotation2D.objects.filter(image=img).count(), 1)
        self.assertEqual(
            Annotation2D.objects.get(image=img).annotation_type, "box"
        )

    def test_worker_per_item_failure_isolated(self):
        prov = self._make_provider(project=self.project, types=("box",))
        img1 = self._make_image(self.project, "a.png")
        img2 = self._make_image(self.project, "b.png")
        job = self._make_job(prov, [img1, img2])

        ok = _fake_response(
            InferenceResponse(
                annotations=[Annotation(label=0, geometry=Box2D(0, 0, 1, 1))]
            ).to_dict()
        )
        # First call raises, second succeeds.
        with patch(
            "anno_infers.tasks.httpx.post",
            side_effect=[RuntimeError("boom"), ok],
        ):
            execute_inference_job(job.id)

        job.refresh_from_db()
        self.assertEqual(job.status, "completed")
        self.assertEqual(job.completed_items, 1)
        self.assertEqual(job.failed_items, 1)
        statuses = dict(job.items.values_list("image_id", "status"))
        self.assertEqual(statuses[img1.id], "failed")
        self.assertEqual(statuses[img2.id], "done")
        self.assertIn("boom", job.items.get(image_id=img1.id).error)

    def test_worker_all_failed_marks_job_failed(self):
        prov = self._make_provider(project=self.project, types=("box",))
        img = self._make_image(self.project)
        job = self._make_job(prov, [img])
        with patch("anno_infers.tasks.httpx.post", side_effect=RuntimeError("nope")):
            execute_inference_job(job.id)
        job.refresh_from_db()
        self.assertEqual(job.status, "failed")
        self.assertEqual(job.failed_items, 1)

    def test_worker_cancel_skips_remaining(self):
        prov = self._make_provider(project=self.project, types=("box",))
        img1 = self._make_image(self.project, "a.png")
        img2 = self._make_image(self.project, "b.png")
        job = self._make_job(prov, [img1, img2])
        job.cancel_requested = True
        job.save(update_fields=["cancel_requested"])

        with patch("anno_infers.tasks.httpx.post") as mock_post:
            execute_inference_job(job.id)
            mock_post.assert_not_called()

        job.refresh_from_db()
        self.assertEqual(job.status, "cancelled")
        self.assertTrue(
            all(s == "skipped" for s in job.items.values_list("status", flat=True))
        )
        self.assertEqual(Annotation2D.objects.filter(project=self.project).count(), 0)

    def test_worker_injects_header_auth_and_metadata(self):
        prov = self._make_provider(
            project=self.project,
            types=("box",),
            auth_type="header",
            auth_param_name="X-API-Key",
            auth_secret="topsecret",
        )
        img = self._make_image(self.project)
        job = self._make_job(prov, [img])
        payload = InferenceResponse(annotations=[]).to_dict()

        with patch(
            "anno_infers.tasks.httpx.post", return_value=_fake_response(payload)
        ) as mock_post:
            execute_inference_job(job.id)

        _, kwargs = mock_post.call_args
        self.assertEqual(kwargs["headers"]["X-API-Key"], "topsecret")
        self.assertEqual(kwargs["params"], {})
        # Metadata carries label_mapping + requested_types + image bytes part.
        meta = json.loads(kwargs["data"]["metadata"])
        self.assertEqual(meta["label_mapping"], {"cat": 0, "dog": 1})
        self.assertEqual(meta["requested_types"], ["box"])
        self.assertEqual(meta["image_id"], img.id)
        self.assertIn("image", kwargs["files"])

    def test_worker_deadline_skips_and_fails(self):
        from datetime import timedelta
        from django.utils import timezone

        prov = self._make_provider(project=self.project, types=("box",))
        img = self._make_image(self.project)
        job = self._make_job(prov, [img])
        job.deadline = timezone.now() - timedelta(seconds=1)  # already past
        job.save(update_fields=["deadline"])

        with patch("anno_infers.tasks.httpx.post") as mock_post:
            execute_inference_job(job.id)
            mock_post.assert_not_called()

        job.refresh_from_db()
        self.assertEqual(job.status, "failed")
        self.assertEqual(job.error, "deadline exceeded")
        self.assertEqual(job.items.get().status, "skipped")

    def test_worker_query_auth(self):
        prov = self._make_provider(
            project=self.project,
            types=("box",),
            auth_type="query",
            auth_param_name="api_key",
            auth_secret="qsecret",
        )
        img = self._make_image(self.project)
        job = self._make_job(prov, [img])
        with patch(
            "anno_infers.tasks.httpx.post",
            return_value=_fake_response(InferenceResponse(annotations=[]).to_dict()),
        ) as mock_post:
            execute_inference_job(job.id)
        _, kwargs = mock_post.call_args
        self.assertEqual(kwargs["params"], {"api_key": "qsecret"})
        self.assertEqual(kwargs["headers"], {})


# ---------------------------------------------------------------------------
# Cancel / retry endpoints
# ---------------------------------------------------------------------------


class CancelRetryTests(FlowBBaseTest):
    def _job(self):
        prov = self._make_provider(project=self.project)
        img = self._make_image(self.project)
        job = InferenceJob.objects.create(
            project=self.project, provider=prov, created_by=self.supervisor,
            total_items=1, status="running",
        )
        item = InferenceJobItem.objects.create(job=job, image=img)
        return job, item

    def test_cancel_sets_flag(self):
        job, _ = self._job()
        res = self.client.post(
            f"/api/projects/{self.project.id}/auto-annotate/jobs/{job.id}/cancel",
            headers=_auth(self.supervisor),
        )
        self.assertEqual(res.status_code, 200)
        job.refresh_from_db()
        self.assertTrue(job.cancel_requested)
        self.assertEqual(job.status, "cancelling")

    def test_cancel_requires_supervisor(self):
        job, _ = self._job()
        res = self.client.post(
            f"/api/projects/{self.project.id}/auto-annotate/jobs/{job.id}/cancel",
            headers=_auth(self.worker),
        )
        self.assertEqual(res.status_code, 403)

    def test_retry_resets_failed_items(self):
        job, item = self._job()
        job.status = "failed"
        job.failed_items = 1
        job.save(update_fields=["status", "failed_items"])
        item.status = "failed"
        item.error = "x"
        item.save(update_fields=["status", "error"])

        res = self.client.post(
            f"/api/projects/{self.project.id}/auto-annotate/jobs/{job.id}/retry",
            headers=_auth(self.supervisor),
        )
        self.assertEqual(res.status_code, 200)
        job.refresh_from_db()
        item.refresh_from_db()
        self.assertEqual(job.status, "pending")
        self.assertEqual(job.failed_items, 0)
        self.assertEqual(item.status, "pending")

    def test_job_detail_lists_items(self):
        job, item = self._job()
        res = self.client.get(
            f"/api/projects/{self.project.id}/auto-annotate/jobs/{job.id}",
            headers=_auth(self.worker),
        )
        self.assertEqual(res.status_code, 200)
        body = res.json()
        self.assertEqual(len(body["items"]), 1)
        self.assertEqual(body["items"][0]["image_id"], item.image_id)
