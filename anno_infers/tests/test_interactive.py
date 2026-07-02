"""Tests for interactive inference (SAM/SAM2/MedSAM style).

Covers the interactive provider registry, sessions, per-step provider calls
(HTTP mocked), the commit flow that creates a real Annotation2D + Operation
with source="interactive", discard, and the reverse-lookup path.
"""

import json
import tempfile
from unittest.mock import MagicMock, patch

from django.contrib.auth import get_user_model
from django.core.files.base import ContentFile
from django.test import Client, TestCase, override_settings

from anno_images.models import Annotation2D, Image2D, Operation
from anno_projects.models import Project, ProjectMembership
from anno_sdk import Annotation, InteractiveInferenceResponse, Polygon2D
from ninja_jwt.tokens import RefreshToken

from anno_infers.models import (
    InteractiveInferenceOperation,
    InteractiveInferenceServiceProvider,
    InteractiveInferenceSession,
)

User = get_user_model()

_TMP_MEDIA = tempfile.mkdtemp(prefix="anno-interactive-test-")

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


def _candidate(points, *, label=0, score=0.9) -> dict:
    return InteractiveInferenceResponse(
        annotation=Annotation(label=label, geometry=Polygon2D(points)),
        score=score,
        model_version="sam2",
    ).to_dict()


@override_settings(STORAGES=_TEST_STORAGES)
class InteractiveBaseTest(TestCase):
    def setUp(self):
        self.client = Client()
        self.supervisor = User.objects.create_user(username="sup", password="x")
        self.worker = User.objects.create_user(username="wrk", password="x")
        self.outsider = User.objects.create_user(username="out", password="x")

        self.project = Project.objects.create(
            name="P", label_mapping={"cat": 0, "dog": 1}, created_by=self.supervisor
        )
        ProjectMembership.objects.create(
            user=self.worker, project=self.project, role="worker"
        )
        self.image = self._make_image(self.project)

    def _make_image(self, project, name="a.png"):
        return Image2D.objects.create(
            project=project,
            image=ContentFile(b"\x89PNG\r\n\x1a\n-fake", name=name),
            width=64,
            height=64,
        )

    def _make_provider(self, *, project=None, prompts=("box", "positive_point"),
                       types=("polygon",), **kw):
        defaults = dict(
            name="sam",
            inference_url="http://svc.local/predict",
            supported_prompt_types=list(prompts),
            supported_result_types=list(types),
            created_by=self.supervisor,
        )
        defaults.update(kw)
        return InteractiveInferenceServiceProvider.objects.create(project=project, **defaults)

    def _start_session(self, provider, **body):
        payload = {"provider_id": provider.id, **body}
        return self.client.post(
            f"/api/projects/{self.project.id}/images/{self.image.id}/interactive-sessions/",
            data=json.dumps(payload),
            content_type="application/json",
            headers=_auth(self.worker),
        )

    def _base_url(self, session_id):
        return (
            f"/api/projects/{self.project.id}/images/{self.image.id}"
            f"/interactive-sessions/{session_id}"
        )


# ---------------------------------------------------------------------------
# Provider registry
# ---------------------------------------------------------------------------


class InteractiveProviderTests(InteractiveBaseTest):
    def test_create_provider_supervisor_hides_secret(self):
        body = {
            "name": "SAM2",
            "inference_url": "http://svc/predict",
            "supported_prompt_types": ["box", "positive_point"],
            "supported_result_types": ["polygon"],
            "auth_type": "header",
            "auth_param_name": "X-API-Key",
            "auth_secret": "s3cr3t",
        }
        res = self.client.post(
            f"/api/projects/{self.project.id}/interactive-providers/",
            data=json.dumps(body),
            content_type="application/json",
            headers=_auth(self.supervisor),
        )
        self.assertEqual(res.status_code, 201, res.content)
        data = res.json()
        self.assertNotIn("auth_secret", data)
        self.assertTrue(data["has_auth_secret"])
        self.assertEqual(data["supported_prompt_types"], ["box", "positive_point"])
        prov = InteractiveInferenceServiceProvider.objects.get(id=data["id"])
        self.assertEqual(prov.auth_secret, "s3cr3t")

    def test_create_provider_worker_forbidden(self):
        body = {
            "name": "x",
            "inference_url": "http://svc/predict",
            "supported_prompt_types": ["box"],
            "supported_result_types": ["polygon"],
        }
        res = self.client.post(
            f"/api/projects/{self.project.id}/interactive-providers/",
            data=json.dumps(body),
            content_type="application/json",
            headers=_auth(self.worker),
        )
        self.assertEqual(res.status_code, 403)

    def test_invalid_prompt_type_rejected(self):
        body = {
            "name": "x",
            "inference_url": "http://svc/predict",
            "supported_prompt_types": ["bogus"],
            "supported_result_types": ["polygon"],
        }
        res = self.client.post(
            f"/api/projects/{self.project.id}/interactive-providers/",
            data=json.dumps(body),
            content_type="application/json",
            headers=_auth(self.supervisor),
        )
        self.assertEqual(res.status_code, 422)


# ---------------------------------------------------------------------------
# Sessions: start, step, commit, discard
# ---------------------------------------------------------------------------


class InteractiveSessionTests(InteractiveBaseTest):
    def test_start_session(self):
        prov = self._make_provider(project=self.project)
        res = self._start_session(prov)
        self.assertEqual(res.status_code, 201, res.content)
        body = res.json()
        self.assertEqual(body["status"], "editing")
        self.assertEqual(body["image_id"], self.image.id)
        self.assertIsNone(body["final_annotation_id"])

    def test_start_inactive_provider_404(self):
        prov = self._make_provider(project=self.project, is_active=False)
        res = self._start_session(prov)
        self.assertEqual(res.status_code, 404)

    def test_step_records_candidate_without_annotation(self):
        prov = self._make_provider(project=self.project)
        session_id = self._start_session(prov).json()["id"]

        payload = _candidate([[0, 0], [1, 1], [2, 0]], score=0.88)
        with patch(
            "anno_infers.services.httpx.post", return_value=_fake_response(payload)
        ) as mock_post:
            res = self.client.post(
                f"{self._base_url(session_id)}/steps",
                data=json.dumps({"prompts": [{"type": "box", "x": 1, "y": 2, "width": 3, "height": 4}]}),
                content_type="application/json",
                headers=_auth(self.worker),
            )
        self.assertEqual(res.status_code, 201, res.content)
        step = res.json()
        self.assertEqual(step["step_index"], 1)
        self.assertEqual(step["result_type"], "polygon")
        self.assertEqual(step["result_data"], {"points": [[0, 0], [1, 1], [2, 0]]})
        self.assertEqual(step["result"]["score"], 0.88)
        # Prompt metadata reached the provider.
        _, kwargs = mock_post.call_args
        meta = json.loads(kwargs["data"]["metadata"])
        self.assertEqual(meta["prompts"][0]["type"], "box")
        self.assertEqual(meta["step_index"], 1)
        # Critically: no Annotation2D or Operation yet.
        self.assertEqual(Annotation2D.objects.filter(image=self.image).count(), 0)
        self.assertEqual(Operation.objects.filter(image=self.image).count(), 0)

    def test_step_unsupported_prompt_rejected(self):
        prov = self._make_provider(project=self.project, prompts=("box",))
        session_id = self._start_session(prov).json()["id"]
        with patch("anno_infers.services.httpx.post") as mock_post:
            res = self.client.post(
                f"{self._base_url(session_id)}/steps",
                data=json.dumps({"prompts": [{"type": "text", "text": "cat"}]}),
                content_type="application/json",
                headers=_auth(self.worker),
            )
            mock_post.assert_not_called()
        self.assertEqual(res.status_code, 422)

    def test_step_provider_failure_records_error(self):
        prov = self._make_provider(project=self.project)
        session_id = self._start_session(prov).json()["id"]
        with patch("anno_infers.services.httpx.post", side_effect=RuntimeError("boom")):
            res = self.client.post(
                f"{self._base_url(session_id)}/steps",
                data=json.dumps({"prompts": [{"type": "box", "x": 1, "y": 2, "width": 3, "height": 4}]}),
                content_type="application/json",
                headers=_auth(self.worker),
            )
        self.assertEqual(res.status_code, 502)
        op = InteractiveInferenceOperation.objects.get(session_id=session_id)
        self.assertIn("boom", op.error)
        # Session stays editing so the user can retry.
        session = InteractiveInferenceSession.objects.get(id=session_id)
        self.assertEqual(session.status, "editing")

    def test_commit_creates_annotation_and_operation(self):
        prov = self._make_provider(project=self.project)
        session_id = self._start_session(prov).json()["id"]

        payload = _candidate([[0, 0], [1, 1], [2, 0]], label=1)
        with patch("anno_infers.services.httpx.post", return_value=_fake_response(payload)):
            step = self.client.post(
                f"{self._base_url(session_id)}/steps",
                data=json.dumps({"prompts": [{"type": "box", "x": 1, "y": 2, "width": 3, "height": 4}]}),
                content_type="application/json",
                headers=_auth(self.worker),
            ).json()

        res = self.client.post(
            f"{self._base_url(session_id)}/commit",
            data=json.dumps({"step_id": step["id"]}),
            content_type="application/json",
            headers=_auth(self.worker),
        )
        self.assertEqual(res.status_code, 200, res.content)
        body = res.json()
        self.assertEqual(body["status"], "committed")
        self.assertIsNotNone(body["final_annotation_id"])

        annotation = Annotation2D.objects.get(id=body["final_annotation_id"])
        self.assertEqual(annotation.annotation_type, "polygon")
        self.assertEqual(annotation.label, 1)
        self.assertTrue(annotation.is_active)

        op = Operation.objects.get(to_annotation=annotation)
        self.assertEqual(op.action, "add")
        self.assertEqual(op.source, "interactive")
        self.assertEqual(op.performed_by_id, self.worker.id)

        # Reverse lookup: source + to_annotation_id -> the session.
        session = InteractiveInferenceSession.objects.get(
            final_annotation_id=op.to_annotation_id
        )
        self.assertEqual(session.id, session_id)

    def test_commit_refine_is_modify(self):
        """A session with from_annotation commits as a modify of the original."""
        original = Annotation2D.objects.create(
            image=self.image, project=self.project, annotation_type="polygon", label=0
        )
        from anno_images.models import Polygon2D as Polygon2DModel

        Polygon2DModel.objects.create(annotation=original, points=[[0, 0], [5, 5], [5, 0]])

        prov = self._make_provider(project=self.project)
        session_id = self._start_session(prov, from_annotation_id=original.id).json()["id"]

        payload = _candidate([[1, 1], [2, 2], [3, 1]], label=0)
        with patch("anno_infers.services.httpx.post", return_value=_fake_response(payload)):
            step = self.client.post(
                f"{self._base_url(session_id)}/steps",
                data=json.dumps({"prompts": [{"type": "positive_point", "x": 2, "y": 2}]}),
                content_type="application/json",
                headers=_auth(self.worker),
            ).json()

        res = self.client.post(
            f"{self._base_url(session_id)}/commit",
            data=json.dumps({"step_id": step["id"]}),
            content_type="application/json",
            headers=_auth(self.worker),
        )
        self.assertEqual(res.status_code, 200, res.content)
        new_id = res.json()["final_annotation_id"]

        op = Operation.objects.get(to_annotation_id=new_id)
        self.assertEqual(op.action, "modify")
        self.assertEqual(op.source, "interactive")
        self.assertEqual(op.from_annotation_id, original.id)
        # Original was deactivated by the immutable modify path.
        original.refresh_from_db()
        self.assertFalse(original.is_active)

    def test_discard_leaves_no_annotation(self):
        prov = self._make_provider(project=self.project)
        session_id = self._start_session(prov).json()["id"]

        payload = _candidate([[0, 0], [1, 1], [2, 0]])
        with patch("anno_infers.services.httpx.post", return_value=_fake_response(payload)):
            self.client.post(
                f"{self._base_url(session_id)}/steps",
                data=json.dumps({"prompts": [{"type": "box", "x": 1, "y": 2, "width": 3, "height": 4}]}),
                content_type="application/json",
                headers=_auth(self.worker),
            )

        res = self.client.post(
            f"{self._base_url(session_id)}/discard",
            headers=_auth(self.worker),
        )
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.json()["status"], "discarded")
        self.assertEqual(Annotation2D.objects.filter(image=self.image).count(), 0)
        self.assertEqual(Operation.objects.filter(image=self.image).count(), 0)

    def test_commit_after_discard_conflict(self):
        prov = self._make_provider(project=self.project)
        session_id = self._start_session(prov).json()["id"]
        payload = _candidate([[0, 0], [1, 1], [2, 0]])
        with patch("anno_infers.services.httpx.post", return_value=_fake_response(payload)):
            step = self.client.post(
                f"{self._base_url(session_id)}/steps",
                data=json.dumps({"prompts": [{"type": "box", "x": 1, "y": 2, "width": 3, "height": 4}]}),
                content_type="application/json",
                headers=_auth(self.worker),
            ).json()
        self.client.post(f"{self._base_url(session_id)}/discard", headers=_auth(self.worker))

        res = self.client.post(
            f"{self._base_url(session_id)}/commit",
            data=json.dumps({"step_id": step["id"]}),
            content_type="application/json",
            headers=_auth(self.worker),
        )
        self.assertEqual(res.status_code, 409)

    def test_detail_lists_steps(self):
        prov = self._make_provider(project=self.project)
        session_id = self._start_session(prov).json()["id"]
        payload = _candidate([[0, 0], [1, 1], [2, 0]])
        with patch("anno_infers.services.httpx.post", return_value=_fake_response(payload)):
            for _ in range(2):
                self.client.post(
                    f"{self._base_url(session_id)}/steps",
                    data=json.dumps({"prompts": [{"type": "box", "x": 1, "y": 2, "width": 3, "height": 4}]}),
                    content_type="application/json",
                    headers=_auth(self.worker),
                )
        res = self.client.get(self._base_url(session_id), headers=_auth(self.worker))
        self.assertEqual(res.status_code, 200)
        body = res.json()
        self.assertEqual(len(body["steps"]), 2)
        self.assertEqual([s["step_index"] for s in body["steps"]], [1, 2])
