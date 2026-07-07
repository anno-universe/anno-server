"""Tests for interactive inference (SAM/SAM2/MedSAM style), direct-call flow.

The latency-sensitive prompt loop runs browser -> service directly, so the
server only (1) opens a session via a server->service handshake that yields a
short-lived browser token, and (2) commits the user's chosen candidate as a
real ``Annotation2D`` + ``Operation`` with ``source="interactive"``. Both the
open handshake and the best-effort completion call are HTTP-mocked at
``anno_infers.services.httpx.post``.
"""

import json
import tempfile
from unittest.mock import MagicMock, patch

from django.contrib.auth import get_user_model
from django.core.files.base import ContentFile
from django.test import Client, TestCase, override_settings

from anno_images.models import Annotation2D, Image2D, Operation, Polygon2D
from anno_projects.models import Project, ProjectMembership
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


def _handshake_response(
    *, token="tok_abc", expires_at="2026-07-07T12:00:00+00:00", predict_url="https://sam.public"
) -> MagicMock:
    """A mocked httpx response for the server->service handshake.

    Also serves the best-effort ``/session/{id}/complete`` call, which ignores
    the body — a single return value covers both hops in a test.
    """
    resp = MagicMock()
    resp.raise_for_status.return_value = None
    resp.json.return_value = {
        "token": token,
        "expires_at": expires_at,
        "session_ref": "svc-ref",
        "predict_url": predict_url,
        "raw": {},
    }
    return resp


def _commit_body(points, *, label=1, annotation_type="polygon"):
    return {
        "annotation_type": annotation_type,
        "label": label,
        "polygon": {"points": points},
        "prompts": [{"type": "box", "x": 1, "y": 2, "width": 3, "height": 4}],
        "score": 0.9,
        "model_version": "sam2",
    }


@override_settings(STORAGES=_TEST_STORAGES)
class InteractiveBaseTest(TestCase):
    def setUp(self):
        self.client = Client()
        self.supervisor = User.objects.create_user(username="sup", password="x")
        self.worker = User.objects.create_user(username="wrk", password="x")
        self.worker2 = User.objects.create_user(username="wrk2", password="x")
        self.outsider = User.objects.create_user(username="out", password="x")

        self.project = Project.objects.create(
            name="P", label_mapping={"cat": 0, "dog": 1}, created_by=self.supervisor
        )
        # The creator is auto-added as a supervisor member by a signal; add the
        # workers explicitly.
        ProjectMembership.objects.create(
            user=self.worker, project=self.project, role="worker"
        )
        ProjectMembership.objects.create(
            user=self.worker2, project=self.project, role="worker"
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
            inference_url="http://svc.local",
            supported_prompt_types=list(prompts),
            supported_result_types=list(types),
            created_by=self.supervisor,
        )
        defaults.update(kw)
        return InteractiveInferenceServiceProvider.objects.create(
            project=project, **defaults
        )

    def _sessions_url(self):
        return (
            f"/api/projects/{self.project.id}/images/{self.image.id}/interactive-sessions"
        )

    def _base_url(self, session_id):
        return f"{self._sessions_url()}/{session_id}"

    def _start_session(self, provider, user=None, **body):
        """Start a session with the handshake mocked; return the response."""
        payload = {"provider_id": provider.id, **body}
        with patch(
            "anno_infers.services.httpx.post", return_value=_handshake_response()
        ) as mock_post:
            res = self.client.post(
                f"{self._sessions_url()}/",
                data=json.dumps(payload),
                content_type="application/json",
                headers=_auth(user or self.worker),
            )
        self._last_start_mock = mock_post
        return res


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

    def test_list_includes_global_provider(self):
        self._make_provider(project=None)  # global
        self._make_provider(project=self.project)  # project-scoped
        res = self.client.get(
            f"/api/projects/{self.project.id}/interactive-providers/",
            headers=_auth(self.worker),
        )
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.json()["count"], 2)


# ---------------------------------------------------------------------------
# Sessions: start, commit, discard
# ---------------------------------------------------------------------------


class InteractiveSessionTests(InteractiveBaseTest):
    def test_start_session_relays_token(self):
        prov = self._make_provider(project=self.project)
        res = self._start_session(prov)
        self.assertEqual(res.status_code, 201, res.content)
        body = res.json()
        self.assertEqual(body["status"], "editing")
        self.assertEqual(body["image_id"], self.image.id)
        self.assertIsNone(body["to_annotation_id"])
        # Short-lived token + direct-call coordinates relayed to the browser.
        self.assertEqual(body["token"], "tok_abc")
        self.assertEqual(body["token_header"], "X-Session-Token")
        self.assertEqual(body["predict_url"], "https://sam.public")
        self.assertEqual(body["supported_prompt_types"], ["box", "positive_point"])
        # Handshake carried the session + image context to the service.
        _, kwargs = self._last_start_mock.call_args
        self.assertEqual(kwargs["json"]["image_id"], self.image.id)
        self.assertEqual(kwargs["json"]["session_id"], body["id"])

    def test_start_inactive_provider_404(self):
        prov = self._make_provider(project=self.project, is_active=False)
        res = self._start_session(prov)
        self.assertEqual(res.status_code, 404)

    def test_start_handshake_failure_502_marks_failed(self):
        prov = self._make_provider(project=self.project)
        with patch(
            "anno_infers.services.httpx.post", side_effect=RuntimeError("down")
        ):
            res = self.client.post(
                f"{self._sessions_url()}/",
                data=json.dumps({"provider_id": prov.id}),
                content_type="application/json",
                headers=_auth(self.worker),
            )
        self.assertEqual(res.status_code, 502)
        session = InteractiveInferenceSession.objects.get(project=self.project)
        self.assertEqual(session.status, "failed")
        self.assertIn("down", session.error)

    def test_commit_creates_annotation_and_operation(self):
        prov = self._make_provider(project=self.project)
        session_id = self._start_session(prov).json()["id"]

        with patch(
            "anno_infers.services.httpx.post", return_value=_handshake_response()
        ) as complete_mock:
            res = self.client.post(
                f"{self._base_url(session_id)}/commit",
                data=json.dumps(_commit_body([[0, 0], [1, 1], [2, 0]], label=1)),
                content_type="application/json",
                headers=_auth(self.worker),
            )
        self.assertEqual(res.status_code, 200, res.content)
        body = res.json()
        self.assertEqual(body["status"], "committed")
        self.assertIsNotNone(body["to_annotation_id"])

        annotation = Annotation2D.objects.get(id=body["to_annotation_id"])
        self.assertEqual(annotation.annotation_type, "polygon")
        self.assertEqual(annotation.label, 1)
        self.assertTrue(annotation.is_active)
        self.assertEqual(annotation.polygon.points, [[0, 0], [1, 1], [2, 0]])

        op = Operation.objects.get(to_annotation=annotation)
        self.assertEqual(op.action, "add")
        self.assertEqual(op.source, "interactive")
        self.assertEqual(op.performed_by_id, self.worker.id)

        # The final prompts were recorded on the interactive operation.
        step = InteractiveInferenceOperation.objects.get(session_id=session_id)
        self.assertEqual(step.prompt["prompts"][0]["type"], "box")
        self.assertEqual(step.result["score"], 0.9)

        # Reverse lookup: source + to_annotation_id -> the session.
        session = InteractiveInferenceSession.objects.get(
            to_annotation_id=op.to_annotation_id
        )
        self.assertEqual(session.id, session_id)
        # Best-effort seat release was attempted.
        complete_mock.assert_called_once()

    def test_commit_refine_is_modify(self):
        original = Annotation2D.objects.create(
            image=self.image, project=self.project, annotation_type="polygon", label=0
        )
        Polygon2D.objects.create(annotation=original, points=[[0, 0], [5, 5], [5, 0]])

        prov = self._make_provider(project=self.project)
        session_id = self._start_session(
            prov, from_annotation_id=original.id
        ).json()["id"]

        with patch(
            "anno_infers.services.httpx.post", return_value=_handshake_response()
        ):
            res = self.client.post(
                f"{self._base_url(session_id)}/commit",
                data=json.dumps(_commit_body([[1, 1], [2, 2], [3, 1]], label=0)),
                content_type="application/json",
                headers=_auth(self.worker),
            )
        self.assertEqual(res.status_code, 200, res.content)
        new_id = res.json()["to_annotation_id"]

        op = Operation.objects.get(to_annotation_id=new_id)
        self.assertEqual(op.action, "modify")
        self.assertEqual(op.source, "interactive")
        self.assertEqual(op.from_annotation_id, original.id)
        original.refresh_from_db()
        self.assertFalse(original.is_active)

    def test_commit_missing_geometry_422(self):
        prov = self._make_provider(project=self.project)
        session_id = self._start_session(prov).json()["id"]
        res = self.client.post(
            f"{self._base_url(session_id)}/commit",
            data=json.dumps({"annotation_type": "polygon", "label": 1}),
            content_type="application/json",
            headers=_auth(self.worker),
        )
        self.assertEqual(res.status_code, 422)
        self.assertEqual(Annotation2D.objects.filter(image=self.image).count(), 0)

    def test_commit_by_non_owner_worker_forbidden(self):
        prov = self._make_provider(project=self.project)
        session_id = self._start_session(prov, user=self.worker).json()["id"]
        res = self.client.post(
            f"{self._base_url(session_id)}/commit",
            data=json.dumps(_commit_body([[0, 0], [1, 1], [2, 0]])),
            content_type="application/json",
            headers=_auth(self.worker2),
        )
        self.assertEqual(res.status_code, 403)

    def test_commit_by_supervisor_allowed(self):
        prov = self._make_provider(project=self.project)
        session_id = self._start_session(prov, user=self.worker).json()["id"]
        with patch(
            "anno_infers.services.httpx.post", return_value=_handshake_response()
        ):
            res = self.client.post(
                f"{self._base_url(session_id)}/commit",
                data=json.dumps(_commit_body([[0, 0], [1, 1], [2, 0]])),
                content_type="application/json",
                headers=_auth(self.supervisor),
            )
        self.assertEqual(res.status_code, 200, res.content)

    def test_discard_leaves_no_annotation(self):
        prov = self._make_provider(project=self.project)
        session_id = self._start_session(prov).json()["id"]
        with patch(
            "anno_infers.services.httpx.post", return_value=_handshake_response()
        ):
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
        with patch(
            "anno_infers.services.httpx.post", return_value=_handshake_response()
        ):
            self.client.post(
                f"{self._base_url(session_id)}/discard", headers=_auth(self.worker)
            )
            res = self.client.post(
                f"{self._base_url(session_id)}/commit",
                data=json.dumps(_commit_body([[0, 0], [1, 1], [2, 0]])),
                content_type="application/json",
                headers=_auth(self.worker),
            )
        self.assertEqual(res.status_code, 409)

    def test_detail_lists_committed_step(self):
        prov = self._make_provider(project=self.project)
        session_id = self._start_session(prov).json()["id"]
        with patch(
            "anno_infers.services.httpx.post", return_value=_handshake_response()
        ):
            self.client.post(
                f"{self._base_url(session_id)}/commit",
                data=json.dumps(_commit_body([[0, 0], [1, 1], [2, 0]])),
                content_type="application/json",
                headers=_auth(self.worker),
            )
        res = self.client.get(
            self._base_url(session_id), headers=_auth(self.worker)
        )
        self.assertEqual(res.status_code, 200)
        body = res.json()
        self.assertEqual(body["status"], "committed")
        self.assertEqual(len(body["steps"]), 1)
        self.assertEqual(body["steps"][0]["step_index"], 1)
