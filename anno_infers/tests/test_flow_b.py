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
from anno_sdk import Annotation, Box2D, InferenceResponse, Polygon2D
from ninja_jwt.tokens import RefreshToken

from anno_infers.models import (
    InferenceResult,
    InferenceRun,
    InferenceServiceProvider,
    InferenceTask,
)
from anno_infers.tasks import execute_inference_run

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
            inference_url="http://svc.local",
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
            "inference_url": "http://svc",
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
            "inference_url": "http://svc",
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
            "inference_url": "http://svc",
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
# Auto-annotate endpoint (batch job creation; worker not run)
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

    def test_start_creates_run_and_tasks(self):
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
        # provider_snapshot is recorded and never carries the secret.
        self.assertEqual(body["provider_snapshot"]["id"], prov.id)
        self.assertNotIn("auth_secret", body["provider_snapshot"])
        run = InferenceRun.objects.get(id=body["id"])
        self.assertEqual(
            set(run.tasks.values_list("image_id", flat=True)), {img1.id, img2.id}
        )

    def test_start_includes_all_images(self):
        """All images in the project are included, annotated or not."""
        prov = self._make_provider(project=self.project)
        annotated = self._make_image(self.project, "x.png")
        Annotation2D.objects.create(
            image=annotated, project=self.project, annotation_type="box", label=0
        )
        fresh = self._make_image(self.project, "y.png")
        res = self.client.post(
            f"/api/projects/{self.project.id}/auto-annotate/",
            data=json.dumps({"provider_id": prov.id}),
            content_type="application/json",
            headers=_auth(self.supervisor),
        )
        self.assertEqual(res.status_code, 201)
        run = InferenceRun.objects.get(id=res.json()["id"])
        self.assertEqual(
            set(run.tasks.values_list("image_id", flat=True)),
            {annotated.id, fresh.id},
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
# Worker (execute_inference_run, provider HTTP mocked)
# ---------------------------------------------------------------------------


class WorkerTests(FlowBBaseTest):
    def _make_run(self, provider, images):
        run = InferenceRun.objects.create(
            project=self.project, provider=provider, created_by=self.supervisor,
            total_items=len(images),
        )
        InferenceTask.objects.bulk_create(
            [InferenceTask(run=run, image=img) for img in images]
        )
        return run

    def test_worker_writes_annotations(self):
        prov = self._make_provider(project=self.project, types=("box", "polygon"))
        img = self._make_image(self.project)
        run = self._make_run(prov, [img])

        payload = InferenceResponse(
            annotations=[
                Annotation(label=0, geometry=Box2D(1, 2, 3, 4)),
                Annotation(label=1, geometry=Polygon2D([[0, 0], [1, 1], [2, 0]])),
            ]
        ).to_dict()

        with patch("anno_infers.tasks.httpx.post", return_value=_fake_response(payload)):
            execute_inference_run(run.id)

        run.refresh_from_db()
        self.assertEqual(run.status, "completed")
        self.assertEqual(run.completed_items, 1)
        self.assertEqual(run.annotations_created, 2)
        self.assertEqual(Annotation2D.objects.filter(image=img).count(), 2)
        # Every operation is tagged with source="inference".
        add_ops = Operation.objects.filter(image=img, action="add")
        self.assertEqual(add_ops.count(), 2)
        self.assertTrue(all(op.source == "inference" for op in add_ops))
        # Each candidate is recorded as a committed InferenceResult linked to
        # the annotation it became (the reverse-lookup path).
        task = run.tasks.get()
        results = list(task.results.all())
        self.assertEqual(len(results), 2)
        self.assertTrue(all(r.status == "committed" for r in results))
        self.assertTrue(all(r.annotation_id is not None for r in results))
        for op in add_ops:
            self.assertTrue(
                InferenceResult.objects.filter(annotation_id=op.to_annotation_id).exists()
            )
        self.assertEqual(task.status, "done")
        self.assertEqual(task.attempts, 1)

    def test_worker_drops_unsupported_type(self):
        prov = self._make_provider(project=self.project, types=("box",))
        img = self._make_image(self.project)
        run = self._make_run(prov, [img])

        payload = InferenceResponse(
            annotations=[
                Annotation(label=0, geometry=Box2D(1, 2, 3, 4)),
                Annotation(label=1, geometry=Polygon2D([[0, 0], [1, 1]])),  # polygon, unsupported
            ]
        ).to_dict()

        with patch("anno_infers.tasks.httpx.post", return_value=_fake_response(payload)):
            execute_inference_run(run.id)

        self.assertEqual(Annotation2D.objects.filter(image=img).count(), 1)
        self.assertEqual(
            Annotation2D.objects.get(image=img).annotation_type, "box"
        )
        # Only the supported candidate is recorded as a result.
        self.assertEqual(InferenceResult.objects.filter(task__run=run).count(), 1)

    def test_worker_per_task_failure_isolated(self):
        prov = self._make_provider(project=self.project, types=("box",))
        img1 = self._make_image(self.project, "a.png")
        img2 = self._make_image(self.project, "b.png")
        run = self._make_run(prov, [img1, img2])

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
            execute_inference_run(run.id)

        run.refresh_from_db()
        self.assertEqual(run.status, "completed")
        self.assertEqual(run.completed_items, 1)
        self.assertEqual(run.failed_items, 1)
        statuses = dict(run.tasks.values_list("image_id", "status"))
        self.assertEqual(statuses[img1.id], "failed")
        self.assertEqual(statuses[img2.id], "done")
        self.assertIn("boom", run.tasks.get(image_id=img1.id).error)

    def test_worker_all_failed_marks_run_failed(self):
        prov = self._make_provider(project=self.project, types=("box",))
        img = self._make_image(self.project)
        run = self._make_run(prov, [img])
        with patch("anno_infers.tasks.httpx.post", side_effect=RuntimeError("nope")):
            execute_inference_run(run.id)
        run.refresh_from_db()
        self.assertEqual(run.status, "failed")
        self.assertEqual(run.failed_items, 1)

    def test_worker_cancel_skips_remaining(self):
        prov = self._make_provider(project=self.project, types=("box",))
        img1 = self._make_image(self.project, "a.png")
        img2 = self._make_image(self.project, "b.png")
        run = self._make_run(prov, [img1, img2])
        run.cancel_requested = True
        run.save(update_fields=["cancel_requested"])

        with patch("anno_infers.tasks.httpx.post") as mock_post:
            execute_inference_run(run.id)
            mock_post.assert_not_called()

        run.refresh_from_db()
        self.assertEqual(run.status, "cancelled")
        self.assertTrue(
            all(s == "skipped" for s in run.tasks.values_list("status", flat=True))
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
        run = self._make_run(prov, [img])
        payload = InferenceResponse(annotations=[]).to_dict()

        with patch(
            "anno_infers.tasks.httpx.post", return_value=_fake_response(payload)
        ) as mock_post:
            execute_inference_run(run.id)

        _, kwargs = mock_post.call_args
        self.assertEqual(kwargs["headers"]["X-API-Key"], "topsecret")
        self.assertEqual(kwargs["params"], {})
        # Metadata carries label_mapping + requested_types + image bytes part.
        meta = json.loads(kwargs["data"]["metadata"])
        self.assertEqual(meta["label_mapping"], {"cat": 0, "dog": 1})
        self.assertEqual(meta["requested_types"], ["box"])
        self.assertEqual(meta["image_id"], img.id)
        self.assertIn("image", kwargs["files"])

    def test_worker_recomputes_deadline_on_start(self):
        """Worker resets deadline based on now, ignoring an already-expired one."""
        from datetime import timedelta
        from django.utils import timezone

        prov = self._make_provider(project=self.project, types=("box",))
        img = self._make_image(self.project)
        run = self._make_run(prov, [img])
        run.deadline = timezone.now() - timedelta(seconds=3600)  # expired an hour ago
        run.save(update_fields=["deadline"])

        payload = InferenceResponse(
            annotations=[Annotation(label=0, geometry=Box2D(0, 0, 1, 1))]
        ).to_dict()

        with patch(
            "anno_infers.tasks.httpx.post", return_value=_fake_response(payload)
        ):
            execute_inference_run(run.id)

        run.refresh_from_db()
        # Deadline was recomputed, so processing succeeded.
        self.assertEqual(run.status, "completed")
        self.assertEqual(run.completed_items, 1)
        self.assertEqual(run.tasks.get().status, "done")

    def test_worker_query_auth(self):
        prov = self._make_provider(
            project=self.project,
            types=("box",),
            auth_type="query",
            auth_param_name="api_key",
            auth_secret="qsecret",
        )
        img = self._make_image(self.project)
        run = self._make_run(prov, [img])
        with patch(
            "anno_infers.tasks.httpx.post",
            return_value=_fake_response(InferenceResponse(annotations=[]).to_dict()),
        ) as mock_post:
            execute_inference_run(run.id)
        _, kwargs = mock_post.call_args
        self.assertEqual(kwargs["params"], {"api_key": "qsecret"})
        self.assertEqual(kwargs["headers"], {})


# ---------------------------------------------------------------------------
# Cancel / retry endpoints
# ---------------------------------------------------------------------------


class CancelRetryTests(FlowBBaseTest):
    def _run(self):
        prov = self._make_provider(project=self.project)
        img = self._make_image(self.project)
        run = InferenceRun.objects.create(
            project=self.project, provider=prov, created_by=self.supervisor,
            total_items=1, status="running",
        )
        task = InferenceTask.objects.create(run=run, image=img)
        return run, task

    def test_cancel_sets_flag(self):
        run, _ = self._run()
        res = self.client.post(
            f"/api/projects/{self.project.id}/auto-annotate/runs/{run.id}/cancel",
            headers=_auth(self.supervisor),
        )
        self.assertEqual(res.status_code, 200)
        run.refresh_from_db()
        self.assertTrue(run.cancel_requested)
        self.assertEqual(run.status, "cancelling")

    def test_cancel_requires_supervisor(self):
        run, _ = self._run()
        res = self.client.post(
            f"/api/projects/{self.project.id}/auto-annotate/runs/{run.id}/cancel",
            headers=_auth(self.worker),
        )
        self.assertEqual(res.status_code, 403)

    def test_retry_resets_failed_tasks(self):
        run, task = self._run()
        run.status = "failed"
        run.failed_items = 1
        run.save(update_fields=["status", "failed_items"])
        task.status = "failed"
        task.error = "x"
        task.save(update_fields=["status", "error"])

        res = self.client.post(
            f"/api/projects/{self.project.id}/auto-annotate/runs/{run.id}/retry",
            headers=_auth(self.supervisor),
        )
        self.assertEqual(res.status_code, 200)
        run.refresh_from_db()
        task.refresh_from_db()
        self.assertEqual(run.status, "pending")
        self.assertEqual(run.failed_items, 0)
        self.assertEqual(task.status, "pending")

    def test_run_detail_lists_tasks(self):
        run, task = self._run()
        res = self.client.get(
            f"/api/projects/{self.project.id}/auto-annotate/runs/{run.id}",
            headers=_auth(self.worker),
        )
        self.assertEqual(res.status_code, 200)
        body = res.json()
        self.assertEqual(len(body["tasks"]), 1)
        self.assertEqual(body["tasks"][0]["image_id"], task.image_id)


# ---------------------------------------------------------------------------
# Single-image inference (a run with one task)
# ---------------------------------------------------------------------------


class SingleImageRunTests(FlowBBaseTest):
    def test_single_image_creates_run_with_one_task(self):
        """Single-image endpoint creates an InferenceRun with one InferenceTask."""
        prov = self._make_provider(project=self.project)
        img = self._make_image(self.project)
        res = self.client.post(
            f"/api/projects/{self.project.id}/images/{img.id}/auto-annotate/",
            data=json.dumps({"provider_id": prov.id}),
            content_type="application/json",
            headers=_auth(self.worker),
        )
        self.assertEqual(res.status_code, 201, res.content)
        body = res.json()
        self.assertEqual(body["provider_id"], prov.id)
        self.assertEqual(body["total_items"], 1)
        self.assertEqual(body["status"], "pending")

        run = InferenceRun.objects.get(id=body["id"])
        self.assertEqual(run.provider_id, prov.id)
        self.assertEqual(run.created_by_id, self.worker.id)
        task = run.tasks.get()
        self.assertEqual(task.image_id, img.id)

    def test_single_image_run_detail_and_task_results(self):
        prov = self._make_provider(project=self.project)
        img = self._make_image(self.project)
        run = InferenceRun.objects.create(
            project=self.project, provider=prov, created_by=self.worker,
            total_items=1,
        )
        task = InferenceTask.objects.create(run=run, image=img)

        payload = InferenceResponse(
            annotations=[Annotation(label=0, geometry=Box2D(0, 0, 1, 1))]
        ).to_dict()

        with patch("anno_infers.tasks.httpx.post", return_value=_fake_response(payload)):
            execute_inference_run(run.id)

        task.refresh_from_db()
        self.assertEqual(task.status, "done")
        self.assertEqual(task.annotations_created, 1)

        # GET run detail via the single-image endpoint lists the one task.
        res = self.client.get(
            f"/api/projects/{self.project.id}/images/{img.id}/auto-annotate/runs/{run.id}",
            headers=_auth(self.worker),
        )
        self.assertEqual(res.status_code, 200)
        body = res.json()
        self.assertEqual(len(body["tasks"]), 1)
        self.assertEqual(body["tasks"][0]["id"], task.id)

        # GET task detail carries the candidate results.
        res = self.client.get(
            f"/api/projects/{self.project.id}/auto-annotate/runs/{run.id}/tasks/{task.id}",
            headers=_auth(self.worker),
        )
        self.assertEqual(res.status_code, 200)
        body = res.json()
        self.assertEqual(body["run_id"], run.id)
        self.assertEqual(len(body["results"]), 1)
        self.assertEqual(body["results"][0]["status"], "committed")
        self.assertEqual(body["results"][0]["result_type"], "box")
        self.assertIsNotNone(body["results"][0]["annotation_id"])

    def test_task_detail_endpoint(self):
        """GET /runs/{run_id}/tasks/{task_id} shows single-task results."""
        prov = self._make_provider(project=self.project)
        img = self._make_image(self.project)
        run = InferenceRun.objects.create(
            project=self.project, provider=prov, created_by=self.worker,
            total_items=1, status="running",
        )
        task = InferenceTask.objects.create(run=run, image=img)

        res = self.client.get(
            f"/api/projects/{self.project.id}/auto-annotate/runs/{run.id}"
            f"/tasks/{task.id}",
            headers=_auth(self.worker),
        )
        self.assertEqual(res.status_code, 200)
        body = res.json()
        self.assertEqual(body["run_id"], run.id)
        self.assertEqual(body["image_id"], img.id)
        self.assertEqual(body["results"], [])

    def test_single_image_worker_writes_annotations(self):
        """A single-image run processes the image exactly like a batch run."""
        prov = self._make_provider(project=self.project, types=("polygon",))
        img = self._make_image(self.project)
        run = InferenceRun.objects.create(
            project=self.project, provider=prov, created_by=self.worker,
            total_items=1,
        )
        task = InferenceTask.objects.create(run=run, image=img)

        payload = InferenceResponse(
            annotations=[Annotation(label=0, geometry=Polygon2D([[0, 0], [1, 1], [2, 0]]))]
        ).to_dict()

        with patch("anno_infers.tasks.httpx.post", return_value=_fake_response(payload)):
            execute_inference_run(run.id)

        task.refresh_from_db()
        self.assertEqual(task.status, "done")
        self.assertEqual(Annotation2D.objects.filter(image=img).count(), 1)
        op = Operation.objects.get(image=img)
        self.assertEqual(op.source, "inference")
        # Reverse lookup works: the result links one-to-one to the annotation.
        result = InferenceResult.objects.get(task=task)
        self.assertEqual(result.annotation, op.to_annotation)
