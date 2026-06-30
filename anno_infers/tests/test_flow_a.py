import json
from datetime import timedelta

from django.contrib.auth import get_user_model
from django.test import Client, TestCase
from django.utils import timezone

from anno_images.models import Annotation2D, Box2D, Image2D, Keypoint2D, Operation
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

    # -- per-image annotation submission ---------------------------------

    def test_submit_per_image_creates_annotations(self):
        """POST /images/{id}/annotations with 2 annotations creates both."""
        payload = {
            "annotations": [
                {
                    "annotation_type": "box",
                    "label": 0,
                    "box": {"x": 1, "y": 2, "width": 3, "height": 4},
                    "client_ref": "a1",
                },
                {
                    "annotation_type": "polygon",
                    "label": 1,
                    "polygon": {"points": [[0, 0], [10, 0], [10, 10]]},
                    "client_ref": "a2",
                },
            ]
        }
        res = self.client.post(
            f"/api/infers/project/images/{self.image.id}/annotations",
            data=json.dumps(payload),
            content_type="application/json",
            headers=self._hdr(),
        )
        self.assertEqual(res.status_code, 200)
        body = res.json()
        self.assertEqual(body["created"], 2)
        self.assertEqual(body["failed"], 0)
        self.assertEqual(len(body["results"]), 2)
        for r in body["results"]:
            self.assertEqual(r["image_id"], self.image.id)
            self.assertEqual(r["status"], "created")
            self.assertIsNotNone(r["annotation_id"])
        # Check DB rows.
        anns = Annotation2D.objects.filter(image=self.image)
        self.assertEqual(anns.count(), 2)
        self.assertEqual(anns.filter(annotation_type="box").count(), 1)
        self.assertEqual(anns.filter(annotation_type="polygon").count(), 1)
        # Each should have an Operation.
        self.assertEqual(Operation.objects.filter(to_annotation__in=anns).count(), 2)

    def test_submit_per_image_image_not_found(self):
        """POST /images/{id}/annotations with a nonexistent image returns 404."""
        payload = {
            "annotations": [
                {
                    "annotation_type": "box",
                    "box": {"x": 0, "y": 0, "width": 1, "height": 1},
                }
            ]
        }
        res = self.client.post(
            "/api/infers/project/images/99999/annotations",
            data=json.dumps(payload),
            content_type="application/json",
            headers=self._hdr(),
        )
        self.assertEqual(res.status_code, 404)

    def test_submit_per_image_cross_project_image(self):
        """An image from another project returns 404 (not revealed to exist)."""
        payload = {
            "annotations": [
                {
                    "annotation_type": "box",
                    "box": {"x": 0, "y": 0, "width": 1, "height": 1},
                }
            ]
        }
        res = self.client.post(
            f"/api/infers/project/images/{self.other_image.id}/annotations",
            data=json.dumps(payload),
            content_type="application/json",
            headers=self._hdr(),
        )
        self.assertEqual(res.status_code, 404)
        self.assertFalse(
            Annotation2D.objects.filter(image=self.other_image).exists()
        )

    def test_submit_per_image_mismatched_subtype_is_per_item_error(self):
        """annotation_type "box" with polygon data fails that item, not the request."""
        payload = {
            "annotations": [
                {
                    "annotation_type": "box",
                    "polygon": {"points": [[1, 2], [3, 4]]},
                    "client_ref": "bad",
                },
                {
                    "annotation_type": "keypoint",
                    "keypoint": {"points": [[5, 6]]},
                    "client_ref": "good",
                },
            ]
        }
        res = self.client.post(
            f"/api/infers/project/images/{self.image.id}/annotations",
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
        # Only the good annotation exists in the DB.
        self.assertEqual(Annotation2D.objects.filter(image=self.image).count(), 1)

    def test_submit_per_image_client_ref_roundtrip(self):
        """client_ref is echoed back in each result item."""
        payload = {
            "annotations": [
                {"annotation_type": "box", "box": {"x": 0, "y": 0, "width": 1, "height": 1}, "client_ref": "ref-xyz"},
            ]
        }
        res = self.client.post(
            f"/api/infers/project/images/{self.image.id}/annotations",
            data=json.dumps(payload),
            content_type="application/json",
            headers=self._hdr(),
        )
        self.assertEqual(res.status_code, 200)
        body = res.json()
        self.assertEqual(body["results"][0]["client_ref"], "ref-xyz")

    def test_submit_per_image_auth_required(self):
        """No API key returns 401."""
        payload = {
            "annotations": [
                {"annotation_type": "box", "box": {"x": 0, "y": 0, "width": 1, "height": 1}},
            ]
        }
        res = self.client.post(
            f"/api/infers/project/images/{self.image.id}/annotations",
            data=json.dumps(payload),
            content_type="application/json",
        )
        self.assertEqual(res.status_code, 401)

    # -- per-image annotation modify --------------------------------------

    def _create_annotation(self, annotation_type="box", label=0, **geometry):
        """Helper that creates an annotation via the POST endpoint."""
        payload = {
            "annotations": [
                {
                    "annotation_type": annotation_type,
                    "label": label,
                    annotation_type: geometry,
                }
            ]
        }
        res = self.client.post(
            f"/api/infers/project/images/{self.image.id}/annotations",
            data=json.dumps(payload),
            content_type="application/json",
            headers=self._hdr(),
        )
        return res.json()["results"][0]["annotation_id"]

    def test_modify_per_image_updates_geometry(self):
        """PATCH updates geometry: new annotation created, old deactivated."""
        ann_id = self._create_annotation(
            "box", label=0, x=1, y=2, width=3, height=4
        )

        payload = {
            "annotation_type": "box",
            "label": 0,
            "box": {"x": 10, "y": 20, "width": 30, "height": 40},
        }
        res = self.client.patch(
            f"/api/infers/project/images/{self.image.id}/annotations/{ann_id}",
            data=json.dumps(payload),
            content_type="application/json",
            headers=self._hdr(),
        )
        self.assertEqual(res.status_code, 200)
        body = res.json()
        self.assertEqual(body["annotation_type"], "box")
        self.assertEqual(body["data"], {"x": 10, "y": 20, "width": 30, "height": 40, "rotation": 0.0})
        self.assertTrue(body["is_active"])

        # Old annotation is now inactive.
        self.assertFalse(Annotation2D.objects.get(id=ann_id).is_active)
        # New annotation exists.
        new_ann = Annotation2D.objects.get(id=body["id"])
        self.assertTrue(new_ann.is_active)
        # Operation(modify) links old → new.
        op = Operation.objects.get(from_annotation_id=ann_id, to_annotation=new_ann)
        self.assertEqual(op.action, "modify")
        self.assertEqual(op.performed_by_id, self.supervisor.id)

    def test_modify_per_image_updates_label(self):
        """PATCH with a new label updates it; omitted label falls back to old."""
        ann_id = self._create_annotation(
            "keypoint", label=0, points=[[1, 2]]
        )

        payload = {
            "annotation_type": "keypoint",
            "label": 1,
            "keypoint": {"points": [[3, 4]]},
        }
        res = self.client.patch(
            f"/api/infers/project/images/{self.image.id}/annotations/{ann_id}",
            data=json.dumps(payload),
            content_type="application/json",
            headers=self._hdr(),
        )
        self.assertEqual(res.status_code, 200)
        body = res.json()
        self.assertEqual(body["label"], 1)
        self.assertEqual(body["data"], {"points": [[3.0, 4.0]]})

    def test_modify_per_image_annotation_not_found(self):
        """PATCH to a nonexistent annotation returns 404."""
        payload = {
            "annotation_type": "box",
            "box": {"x": 0, "y": 0, "width": 1, "height": 1},
        }
        res = self.client.patch(
            f"/api/infers/project/images/{self.image.id}/annotations/99999",
            data=json.dumps(payload),
            content_type="application/json",
            headers=self._hdr(),
        )
        self.assertEqual(res.status_code, 404)

    def test_modify_per_image_cross_project(self):
        """Annotation from another project returns 404."""
        # Create an annotation in the other project via the other project's key.
        other_key, other_token = ProjectAPIKey.generate(
            project=self.other_project, name="gpu2", created_by=self.supervisor
        )
        other_key.save()
        payload = {
            "annotations": [
                {
                    "annotation_type": "box",
                    "box": {"x": 0, "y": 0, "width": 1, "height": 1},
                }
            ]
        }
        res = self.client.post(
            f"/api/infers/project/images/{self.other_image.id}/annotations",
            data=json.dumps(payload),
            content_type="application/json",
            headers={"X-API-Key": other_token},
        )
        other_ann_id = res.json()["results"][0]["annotation_id"]

        # Now try to modify it with our project's key — should 404.
        payload2 = {
            "annotation_type": "box",
            "box": {"x": 5, "y": 5, "width": 2, "height": 2},
        }
        res2 = self.client.patch(
            f"/api/infers/project/images/{self.other_image.id}/annotations/{other_ann_id}",
            data=json.dumps(payload2),
            content_type="application/json",
            headers=self._hdr(),
        )
        self.assertEqual(res2.status_code, 404)

    def test_modify_per_image_auth_required(self):
        """No API key returns 401."""
        ann_id = self._create_annotation(
            "box", label=0, x=1, y=2, width=3, height=4
        )
        payload = {
            "annotation_type": "box",
            "box": {"x": 5, "y": 5, "width": 2, "height": 2},
        }
        res = self.client.patch(
            f"/api/infers/project/images/{self.image.id}/annotations/{ann_id}",
            data=json.dumps(payload),
            content_type="application/json",
        )
        self.assertEqual(res.status_code, 401)

    def test_modify_per_image_already_inactive(self):
        """Modifying an inactive annotation returns 404."""
        ann_id = self._create_annotation(
            "box", label=0, x=1, y=2, width=3, height=4
        )
        # Deactivate it.
        ann = Annotation2D.objects.get(id=ann_id)
        ann.is_active = False
        ann.save(update_fields=["is_active"])

        payload = {
            "annotation_type": "box",
            "box": {"x": 5, "y": 5, "width": 2, "height": 2},
        }
        res = self.client.patch(
            f"/api/infers/project/images/{self.image.id}/annotations/{ann_id}",
            data=json.dumps(payload),
            content_type="application/json",
            headers=self._hdr(),
        )
        self.assertEqual(res.status_code, 404)

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
