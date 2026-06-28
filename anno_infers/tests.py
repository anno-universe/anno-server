import json
from datetime import timedelta

from django.contrib.auth import get_user_model
from django.test import Client, TestCase
from django.utils import timezone

from anno_images.models import Annotation2D, Box2D, Image2D, Operation
from anno_projects.models import Project, ProjectAPIKey

User = get_user_model()


def _make_image(project, name="a.png"):
    # Assign the storage name directly (a string) so no real upload to S3 happens;
    # the worker list/submit paths never read the file bytes.
    return Image2D.objects.create(
        project=project,
        image=f"images/{project.id}/{name}",
        file_name=name,
        width=640,
        height=480,
    )


class ProjectInferAPITests(TestCase):
    def setUp(self):
        self.client = Client()
        self.supervisor = User.objects.create_user(username="sup", password="x")
        self.project = Project.objects.create(
            name="P",
            description="A test project",
            meta_info={"task": "segmentation"},
            label_mapping={"cat": 0, "dog": 1},
            created_by=self.supervisor,
        )
        self.other_project = Project.objects.create(
            name="Q", created_by=self.supervisor
        )
        self.image = _make_image(self.project, "a.png")
        self.other_image = _make_image(self.other_project, "b.png")
        self.key_obj, self.token = ProjectAPIKey.generate(
            project=self.project, name="gpu", created_by=self.supervisor
        )
        self.key_obj.save()

    def _hdr(self, token=None):
        return {"X-API-Key": token or self.token}

    def test_project_meta(self):
        res = self.client.get(
            "/api/infers/project/meta", headers=self._hdr()
        )
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["name"], "P")
        self.assertEqual(data["description"], "A test project")
        self.assertEqual(data["meta_info"], {"task": "segmentation"})
        self.assertEqual(data["label_mapping"], {"cat": 0, "dog": 1})
        # The other project's meta is inaccessible with this key.
        res2 = self.client.get(
            "/api/infers/project/meta",
            headers={"X-API-Key": "ak_dead.beef"},
        )
        self.assertEqual(res2.status_code, 401)

    def test_list_images_scoped_to_project(self):
        res = self.client.get("/api/infers/project/images", headers=self._hdr())
        self.assertEqual(res.status_code, 200)
        data = res.json()
        ids = [i["id"] for i in data["items"]]
        self.assertIn(self.image.id, ids)
        self.assertNotIn(self.other_image.id, ids)
        self.assertEqual(data["count"], 1)

    def test_missing_key_returns_401(self):
        res = self.client.get("/api/infers/project/images")
        self.assertEqual(res.status_code, 401)

    def test_wrong_key_returns_401(self):
        res = self.client.get(
            "/api/infers/project/images", headers=self._hdr("ak_dead.beef")
        )
        self.assertEqual(res.status_code, 401)

    def test_revoked_key_returns_401(self):
        self.key_obj.is_active = False
        self.key_obj.save(update_fields=["is_active"])
        res = self.client.get("/api/infers/project/images", headers=self._hdr())
        self.assertEqual(res.status_code, 401)

    def test_expired_key_returns_401(self):
        self.key_obj.expires_at = timezone.now() - timedelta(seconds=1)
        self.key_obj.save(update_fields=["expires_at"])
        res = self.client.get("/api/infers/project/images", headers=self._hdr())
        self.assertEqual(res.status_code, 401)

    def test_submit_creates_annotation_attributed_to_key_creator(self):
        payload = {
            "items": [
                {
                    "image_id": self.image.id,
                    "annotation_type": "box",
                    "label": 1,
                    "box": {"x": 1, "y": 2, "width": 3, "height": 4},
                    "client_ref": "r1",
                }
            ]
        }
        res = self.client.post(
            "/api/infers/project/annotations",
            data=json.dumps(payload),
            content_type="application/json",
            headers=self._hdr(),
        )
        self.assertEqual(res.status_code, 200)
        body = res.json()
        self.assertEqual(body["created"], 1)
        self.assertEqual(body["failed"], 0)
        self.assertEqual(body["results"][0]["client_ref"], "r1")

        ann_id = body["results"][0]["annotation_id"]
        ann = Annotation2D.objects.get(id=ann_id)
        self.assertEqual(ann.annotation_type, "box")
        self.assertEqual(ann.project_id, self.project.id)
        self.assertTrue(ann.is_active)
        self.assertTrue(Box2D.objects.filter(annotation=ann).exists())

        op = Operation.objects.get(to_annotation=ann)
        self.assertEqual(op.action, "add")
        self.assertEqual(op.performed_by_id, self.supervisor.id)

    def test_submit_cross_project_image_rejected_per_item(self):
        payload = {
            "items": [
                {
                    "image_id": self.other_image.id,
                    "annotation_type": "box",
                    "box": {"x": 0, "y": 0, "width": 1, "height": 1},
                    "client_ref": "bad",
                },
                {
                    "image_id": self.image.id,
                    "annotation_type": "keypoint",
                    "keypoint": {"points": [[1, 2]]},
                    "client_ref": "good",
                },
            ]
        }
        res = self.client.post(
            "/api/infers/project/annotations",
            data=json.dumps(payload),
            content_type="application/json",
            headers=self._hdr(),
        )
        self.assertEqual(res.status_code, 200)
        body = res.json()
        self.assertEqual(body["created"], 1)
        self.assertEqual(body["failed"], 1)
        by_ref = {r["client_ref"]: r for r in body["results"]}
        self.assertEqual(by_ref["bad"]["status"], "error")
        self.assertEqual(by_ref["good"]["status"], "created")
        self.assertFalse(
            Annotation2D.objects.filter(image=self.other_image).exists()
        )

    def test_submit_mismatched_subtype_is_per_item_error(self):
        payload = {
            "items": [
                {
                    "image_id": self.image.id,
                    "annotation_type": "box",
                    "polygon": {"points": [[1, 2], [3, 4]]},
                    "client_ref": "m",
                }
            ]
        }
        res = self.client.post(
            "/api/infers/project/annotations",
            data=json.dumps(payload),
            content_type="application/json",
            headers=self._hdr(),
        )
        self.assertEqual(res.status_code, 200)
        body = res.json()
        self.assertEqual(body["failed"], 1)
        self.assertEqual(Annotation2D.objects.filter(image=self.image).count(), 0)

    def test_has_active_annotations_filter(self):
        res = self.client.get(
            "/api/infers/project/images?has_active_annotations=false",
            headers=self._hdr(),
        )
        self.assertEqual([i["id"] for i in res.json()["items"]], [self.image.id])
        res = self.client.get(
            "/api/infers/project/images?has_active_annotations=true",
            headers=self._hdr(),
        )
        self.assertEqual(res.json()["items"], [])
