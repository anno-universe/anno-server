"""Tests for anno_exports: format builders, task execution, and API endpoints."""

import io
import json
import math
import os
import tempfile
import zipfile
from datetime import timedelta

from django.contrib.auth import get_user_model
from django.core.files.base import ContentFile
from django.test import Client, TestCase, override_settings
from django.utils import timezone
from ninja_jwt.tokens import RefreshToken

from anno_images.models import Annotation2D, Box2D, Image2D, Keypoint2D, Polygon2D
from anno_projects.models import Project, ProjectMembership

from anno_exports.management.commands.run_scheduler import _cleanup_expired_exports
from anno_exports.models import ExportTask, ExportTaskResult
from anno_exports.schemas import ExportTaskDetailOutput, ExportTaskOutput
from anno_exports.services import (
    _box_to_corners,
    _extent_from_corners,
    _invert_label_mapping,
    build_coco,
    build_yolo,
    load_export_data,
)
from anno_exports.tasks import execute_export

User = get_user_model()

_TMP_MEDIA = tempfile.mkdtemp(prefix="anno-exports-test-")

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


def _make_png_bytes():
    return b"\x89PNG\r\n\x1a\n" + b"\x00" * 64


# ---------------------------------------------------------------------------
# _invert_label_mapping tests
# ---------------------------------------------------------------------------


class InvertLabelMappingTests(TestCase):
    def test_flat_simple(self):
        mapping = {"cat": 0, "dog": 1}
        result = _invert_label_mapping(mapping)
        self.assertEqual(result, {0: "cat", 1: "dog"})

    def test_flat_complex(self):
        mapping = {
            "cat": {"id": 0, "color": "#FF0000"},
            "dog": {"id": 1, "color": "#00FF00"},
        }
        result = _invert_label_mapping(mapping)
        self.assertEqual(result, {0: "cat", 1: "dog"})

    def test_nested_labels_key(self):
        mapping = {
            "labels": {
                "c1": {"id": 1, "color": "#AAA"},
                "c2": {"id": 2, "color": "#BBB"},
            },
            "version": 2,
        }
        result = _invert_label_mapping(mapping)
        self.assertEqual(result, {1: "c1", 2: "c2"})

    def test_nested_labels_flat(self):
        mapping = {
            "labels": {"cat": 0, "dog": 1},
            "version": 1,
        }
        result = _invert_label_mapping(mapping)
        self.assertEqual(result, {0: "cat", 1: "dog"})

    def test_empty(self):
        self.assertEqual(_invert_label_mapping({}), {})


# ---------------------------------------------------------------------------
# box_to_corners / extent tests
# ---------------------------------------------------------------------------


class BoxRotationTests(TestCase):
    def test_axis_aligned_box(self):
        box = Box2D(x=10, y=20, width=100, height=50, rotation=0)
        corners = _box_to_corners(box)
        self.assertEqual(corners, [(10, 20), (110, 20), (110, 70), (10, 70)])

    def test_rotated_box_90_degrees(self):
        box = Box2D(x=10, y=20, width=100, height=50, rotation=90)
        corners = _box_to_corners(box)
        # Center = (60, 45). After 90 deg CCW rotation:
        # Expected: TL(-50,-25) rotated →...
        for cx, cy in corners:
            self.assertAlmostEqual(
                (cx - 60) ** 2 + (cy - 45) ** 2, 50**2 + 25**2, delta=1
            )

    def test_rotated_box_near_zero_threshold(self):
        box = Box2D(x=10, y=20, width=100, height=50, rotation=1e-7)
        corners = _box_to_corners(box)
        self.assertEqual(len(corners), 4)
        self.assertAlmostEqual(corners[0][0], 10, delta=1e-3)
        self.assertAlmostEqual(corners[0][1], 20, delta=1e-3)

    def test_extent_from_corners(self):
        corners = [(1, 5), (10, 2), (12, 8), (3, 10)]
        bbox = _extent_from_corners(corners)
        self.assertEqual(bbox, [1, 2, 11, 8])

    def test_corners_order_tl_tr_br_bl(self):
        box = Box2D(x=0, y=0, width=100, height=50, rotation=0)
        corners = _box_to_corners(box)
        # TL should have smallest x, y; TR largest x, smallest y
        self.assertLess(corners[0][1], corners[2][1])  # TL above BR
        self.assertLess(corners[1][1], corners[2][1])  # TR above BR


# ---------------------------------------------------------------------------
# COCO builder
# ---------------------------------------------------------------------------


@override_settings(STORAGES=_TEST_STORAGES)
class COCOBuilderTest(TestCase):
    def setUp(self):
        self.project = Project.objects.create(
            name="Test",
            label_mapping={"cat": 0, "dog": 1},
            created_by=User.objects.create_user(username="a", password="x"),
        )
        self.image = Image2D.objects.create(
            project=self.project,
            image=ContentFile(_make_png_bytes(), name="a.png"),
            width=640,
            height=480,
        )

    def _polygon_ann(self):
        ann = Annotation2D.objects.create(
            image=self.image, project=self.project, annotation_type="polygon", label=0
        )
        Polygon2D.objects.create(
            annotation=ann,
            points=[[100, 100], [200, 100], [200, 200], [100, 200]],
        )
        return ann

    def _box_ann(self, rotation=0):
        ann = Annotation2D.objects.create(
            image=self.image, project=self.project, annotation_type="box", label=1
        )
        Box2D.objects.create(
            annotation=ann,
            x=50, y=60, width=80, height=100, rotation=rotation,
        )
        return ann

    def _keypoint_ann(self):
        ann = Annotation2D.objects.create(
            image=self.image, project=self.project, annotation_type="keypoint", label=0
        )
        Keypoint2D.objects.create(
            annotation=ann,
            points=[[320, 240], [330, 250]],
        )
        return ann

    def test_coco_structure(self):
        self._polygon_ann()
        self._box_ann()
        images, annotations = load_export_data(self.project.id)
        coco = build_coco(images, annotations, self.project.label_mapping)

        self.assertIn("images", coco)
        self.assertIn("annotations", coco)
        self.assertIn("categories", coco)
        self.assertEqual(len(coco["images"]), 1)
        self.assertEqual(len(coco["annotations"]), 2)
        self.assertEqual(len(coco["categories"]), 2)

    def test_coco_polygon(self):
        self._polygon_ann()
        images, annotations = load_export_data(self.project.id)
        coco = build_coco(images, annotations, self.project.label_mapping)

        ann = coco["annotations"][0]
        self.assertEqual(ann["category_id"], 0)
        self.assertIn("segmentation", ann)
        self.assertGreater(ann["area"], 0)

    def test_coco_axis_aligned_box(self):
        self._box_ann(rotation=0)
        images, annotations = load_export_data(self.project.id)
        coco = build_coco(images, annotations, self.project.label_mapping)

        ann = coco["annotations"][0]
        self.assertEqual(ann["category_id"], 1)
        self.assertEqual(ann["bbox"], [50, 60, 80, 100])
        self.assertEqual(ann["area"], 8000)
        self.assertNotIn("segmentation", ann)

    def test_coco_rotated_box(self):
        self._box_ann(rotation=45)
        images, annotations = load_export_data(self.project.id)
        coco = build_coco(images, annotations, self.project.label_mapping)

        ann = coco["annotations"][0]
        self.assertEqual(ann["area"], 8000)
        self.assertIn("segmentation", ann)

    def test_coco_keypoint(self):
        self._keypoint_ann()
        images, annotations = load_export_data(self.project.id)
        coco = build_coco(images, annotations, self.project.label_mapping)

        ann = coco["annotations"][0]
        self.assertIn("keypoints", ann)
        self.assertEqual(ann["num_keypoints"], 2)
        self.assertEqual(ann["keypoints"], [320, 240, 2, 330, 250, 2])


# ---------------------------------------------------------------------------
# YOLO builder
# ---------------------------------------------------------------------------


@override_settings(STORAGES=_TEST_STORAGES)
class YOLOBuilderTest(TestCase):
    def setUp(self):
        self.project = Project.objects.create(
            name="Test",
            label_mapping={"cat": 0, "dog": 1},
            created_by=User.objects.create_user(username="a", password="x"),
        )
        self.image = Image2D.objects.create(
            project=self.project,
            image=ContentFile(_make_png_bytes(), name="a.png"),
            width=640,
            height=480,
        )

    def _polygon_ann(self):
        ann = Annotation2D.objects.create(
            image=self.image, project=self.project, annotation_type="polygon", label=0
        )
        Polygon2D.objects.create(
            annotation=ann, points=[[100, 100], [200, 100], [200, 200], [100, 200]],
        )
        return ann

    def _box_ann(self, rotation=0):
        ann = Annotation2D.objects.create(
            image=self.image, project=self.project, annotation_type="box", label=1
        )
        Box2D.objects.create(
            annotation=ann, x=50, y=60, width=80, height=100, rotation=rotation,
        )
        return ann

    def _keypoint_ann(self):
        ann = Annotation2D.objects.create(
            image=self.image, project=self.project, annotation_type="keypoint", label=0
        )
        Keypoint2D.objects.create(
            annotation=ann, points=[[320, 240]],
        )
        return ann

    def test_yolo_structure(self):
        self._polygon_ann()
        self._box_ann()
        self._keypoint_ann()
        images, annotations = load_export_data(self.project.id)
        yolo = build_yolo(images, annotations, self.project.label_mapping)

        labels = yolo["labels"][self.image.id]
        self.assertEqual(len(labels), 2)  # keypoint skipped
        self.assertIn("classes_txt", yolo)
        self.assertIn("cat\n", yolo["classes_txt"])
        self.assertIn("dog\n", yolo["classes_txt"])

    def test_yolo_axis_aligned_box(self):
        self._box_ann(rotation=0)
        images, annotations = load_export_data(self.project.id)
        yolo = build_yolo(images, annotations, self.project.label_mapping)

        line = yolo["labels"][self.image.id][0]
        parts = line.split()
        self.assertEqual(parts[0], "1")
        self.assertAlmostEqual(float(parts[1]), 90 / 640, delta=0.01)
        self.assertAlmostEqual(float(parts[2]), 110 / 480, delta=0.01)
        self.assertAlmostEqual(float(parts[3]), 80 / 640, delta=0.01)
        self.assertAlmostEqual(float(parts[4]), 100 / 480, delta=0.01)

    def test_yolo_rotated_box(self):
        self._box_ann(rotation=45)
        images, annotations = load_export_data(self.project.id)
        yolo = build_yolo(images, annotations, self.project.label_mapping)

        line = yolo["labels"][self.image.id][0]
        parts = line.split()
        self.assertEqual(parts[0], "1")
        self.assertEqual(len(parts), 9)  # class + 8 values (4 corners)

    def test_yolo_skips_keypoint(self):
        self._keypoint_ann()
        images, annotations = load_export_data(self.project.id)
        yolo = build_yolo(images, annotations, self.project.label_mapping)
        self.assertEqual(len(yolo["labels"][self.image.id]), 0)


# ---------------------------------------------------------------------------
# ExportTask execution
# ---------------------------------------------------------------------------


@override_settings(STORAGES=_TEST_STORAGES)
class ExportTaskExecutionTest(TestCase):
    def setUp(self):
        self.user = User.objects.create_user(username="a", password="x")
        self.project = Project.objects.create(
            name="Test Project",
            label_mapping={"cat": 0},
            created_by=self.user,
        )
        self.image = Image2D.objects.create(
            project=self.project,
            image=ContentFile(_make_png_bytes(), name="a.png"),
            width=64,
            height=64,
        )

    def _make_annotation(self):
        ann = Annotation2D.objects.create(
            image=self.image, project=self.project, annotation_type="box", label=0
        )
        Box2D.objects.create(annotation=ann, x=0, y=0, width=32, height=32, rotation=0)
        return ann

    def test_execute_export_coco(self):
        self._make_annotation()
        task = ExportTask.objects.create(
            project=self.project,
            created_by=self.user,
            format="coco",
            expires_at=timezone.now() + timedelta(hours=1),
        )
        execute_export(task.id)
        task.refresh_from_db()
        self.assertEqual(task.status, ExportTask.STATUS_COMPLETED)
        result = ExportTaskResult.objects.get(task=task)
        self.assertTrue(result.file_available)
        self.assertGreater(result.file_size, 0)

        with zipfile.ZipFile(result.export_file.path, "r") as zf:
            names = zf.namelist()
            self.assertIn("annotations.json", names)

    def test_execute_export_yolo(self):
        self._make_annotation()
        task = ExportTask.objects.create(
            project=self.project,
            created_by=self.user,
            format="yolo",
            expires_at=timezone.now() + timedelta(hours=1),
        )
        execute_export(task.id)
        task.refresh_from_db()
        self.assertEqual(task.status, ExportTask.STATUS_COMPLETED)

        result = ExportTaskResult.objects.get(task=task)
        with zipfile.ZipFile(result.export_file.path, "r") as zf:
            names = zf.namelist()
            self.assertIn("classes.txt", names)
            self.assertTrue(any(n.startswith("labels/") for n in names))

    def test_execute_export_with_images(self):
        self._make_annotation()
        task = ExportTask.objects.create(
            project=self.project,
            created_by=self.user,
            format="coco",
            include_images=True,
            expires_at=timezone.now() + timedelta(hours=1),
        )
        execute_export(task.id)
        task.refresh_from_db()
        self.assertEqual(task.status, ExportTask.STATUS_COMPLETED)

        result = ExportTaskResult.objects.get(task=task)
        with zipfile.ZipFile(result.export_file.path, "r") as zf:
            names = zf.namelist()
            self.assertTrue(any(n.startswith("images/") for n in names))

    def test_execute_export_no_annotations(self):
        task = ExportTask.objects.create(
            project=self.project,
            created_by=self.user,
            format="coco",
            expires_at=timezone.now() + timedelta(hours=1),
        )
        execute_export(task.id)
        task.refresh_from_db()
        self.assertEqual(task.status, ExportTask.STATUS_FAILED)
        self.assertIn("No images", task.error)


# ---------------------------------------------------------------------------
# API endpoints
# ---------------------------------------------------------------------------


@override_settings(STORAGES=_TEST_STORAGES)
class ExportAPITest(TestCase):
    def setUp(self):
        self.client = Client()
        self.supervisor = User.objects.create_user(username="sup", password="x")
        self.worker = User.objects.create_user(username="wrk", password="x")
        self.outsider = User.objects.create_user(username="out", password="x")

        self.project = Project.objects.create(
            name="P",
            label_mapping={"a": 0},
            created_by=self.supervisor,
        )
        ProjectMembership.objects.create(
            user=self.worker, project=self.project, role="worker"
        )

    def test_create_export_supervisor(self):
        res = self.client.post(
            f"/api/projects/{self.project.id}/exports/",
            data=json.dumps({"format": "coco"}),
            content_type="application/json",
            headers=_auth(self.supervisor),
        )
        self.assertEqual(res.status_code, 201)
        data = res.json()
        self.assertEqual(data["format"], "coco")
        self.assertEqual(data["status"], "pending")

    def test_create_export_worker_forbidden(self):
        res = self.client.post(
            f"/api/projects/{self.project.id}/exports/",
            data=json.dumps({"format": "coco"}),
            content_type="application/json",
            headers=_auth(self.worker),
        )
        self.assertEqual(res.status_code, 403)

    def test_create_export_outsider_forbidden(self):
        res = self.client.post(
            f"/api/projects/{self.project.id}/exports/",
            data=json.dumps({"format": "coco"}),
            content_type="application/json",
            headers=_auth(self.outsider),
        )
        self.assertEqual(res.status_code, 403)

    def test_create_export_invalid_format(self):
        res = self.client.post(
            f"/api/projects/{self.project.id}/exports/",
            data=json.dumps({"format": "invalid"}),
            content_type="application/json",
            headers=_auth(self.supervisor),
        )
        self.assertEqual(res.status_code, 422)

    def test_create_export_custom_expires(self):
        future = (timezone.now() + timedelta(hours=48)).isoformat()
        res = self.client.post(
            f"/api/projects/{self.project.id}/exports/",
            data=json.dumps({"format": "coco", "expires_at": future}),
            content_type="application/json",
            headers=_auth(self.supervisor),
        )
        self.assertEqual(res.status_code, 201)

    def test_create_export_never_expire(self):
        res = self.client.post(
            f"/api/projects/{self.project.id}/exports/",
            data=json.dumps({"format": "coco", "expires_at": None}),
            content_type="application/json",
            headers=_auth(self.supervisor),
        )
        self.assertEqual(res.status_code, 201)
        self.assertIsNone(res.json()["expires_at"])

    def test_list_exports_member(self):
        ExportTask.objects.create(
            project=self.project, created_by=self.supervisor, format="coco",
        )
        res = self.client.get(
            f"/api/projects/{self.project.id}/exports/",
            headers=_auth(self.worker),
        )
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["count"], 1)

    def test_list_exports_outsider_forbidden(self):
        res = self.client.get(
            f"/api/projects/{self.project.id}/exports/",
            headers=_auth(self.outsider),
        )
        self.assertEqual(res.status_code, 403)

    def test_download_not_completed(self):
        task = ExportTask.objects.create(
            project=self.project, created_by=self.supervisor, format="coco",
        )
        res = self.client.get(
            f"/api/projects/{self.project.id}/exports/{task.id}/download",
            headers=_auth(self.supervisor),
        )
        self.assertEqual(res.status_code, 404)

    def test_download_worker_forbidden(self):
        task = ExportTask.objects.create(
            project=self.project, created_by=self.supervisor, format="coco",
        )
        res = self.client.get(
            f"/api/projects/{self.project.id}/exports/{task.id}/download",
            headers=_auth(self.worker),
        )
        self.assertEqual(res.status_code, 403)

    def test_delete_file_manual(self):
        task = ExportTask.objects.create(
            project=self.project,
            created_by=self.supervisor,
            format="coco",
            status=ExportTask.STATUS_COMPLETED,
        )
        ExportTaskResult.objects.create(task=task)
        # Put a fake file
        tmp = tempfile.mktemp(suffix=".zip", dir=_TMP_MEDIA)
        with open(tmp, "wb") as f:
            f.write(b"fake zip")
        from django.core.files import File
        result = task.result
        with open(tmp, "rb") as f:
            result.export_file.save("test.zip", File(f), save=True)

        res = self.client.delete(
            f"/api/projects/{self.project.id}/exports/{task.id}/file",
            headers=_auth(self.supervisor),
        )
        self.assertEqual(res.status_code, 204)
        result.refresh_from_db()
        self.assertFalse(result.file_available)
        self.assertIsNotNone(result.file_deleted_at)

    def test_detail(self):
        task = ExportTask.objects.create(
            project=self.project, created_by=self.supervisor, format="coco",
        )
        res = self.client.get(
            f"/api/projects/{self.project.id}/exports/{task.id}",
            headers=_auth(self.supervisor),
        )
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.json()["id"], task.id)
        self.assertIsNone(res.json()["result"])


# ---------------------------------------------------------------------------
# Expired export cleanup
# ---------------------------------------------------------------------------


@override_settings(STORAGES=_TEST_STORAGES)
class ExpiredExportCleanupTest(TestCase):
    def setUp(self):
        self.user = User.objects.create_user(username="a", password="x")
        self.project = Project.objects.create(
            name="Test", label_mapping={"a": 0}, created_by=self.user,
        )

    def test_cleanup_marks_as_expired(self):
        task = ExportTask.objects.create(
            project=self.project,
            created_by=self.user,
            format="coco",
            status=ExportTask.STATUS_COMPLETED,
            expires_at=timezone.now() - timedelta(hours=1),
        )
        ExportTaskResult.objects.create(task=task)

        _cleanup_expired_exports()

        task.refresh_from_db()
        self.assertEqual(task.status, ExportTask.STATUS_EXPIRED)

    def test_cleanup_moves_file_and_marks_expired(self):
        task = ExportTask.objects.create(
            project=self.project,
            created_by=self.user,
            format="coco",
            status=ExportTask.STATUS_COMPLETED,
            expires_at=timezone.now() - timedelta(hours=1),
        )
        result = ExportTaskResult.objects.create(task=task)
        result.export_file.save("test.zip", ContentFile(b"data"), save=True)

        _cleanup_expired_exports()

        task.refresh_from_db()
        result.refresh_from_db()
        self.assertEqual(task.status, ExportTask.STATUS_EXPIRED)
        self.assertFalse(bool(result.export_file))
        self.assertIsNotNone(result.file_deleted_at)

    def test_non_expired_ignored(self):
        task = ExportTask.objects.create(
            project=self.project,
            created_by=self.user,
            format="coco",
            status=ExportTask.STATUS_COMPLETED,
            expires_at=timezone.now() + timedelta(hours=1),
        )
        ExportTaskResult.objects.create(task=task)

        _cleanup_expired_exports()

        task.refresh_from_db()
        self.assertEqual(task.status, ExportTask.STATUS_COMPLETED)

    def test_expired_without_result_marked_expired(self):
        task = ExportTask.objects.create(
            project=self.project,
            created_by=self.user,
            format="coco",
            status=ExportTask.STATUS_COMPLETED,
            expires_at=timezone.now() - timedelta(hours=1),
        )

        _cleanup_expired_exports()

        task.refresh_from_db()
        self.assertEqual(task.status, ExportTask.STATUS_EXPIRED)

    def test_no_expiry_not_cleaned(self):
        task = ExportTask.objects.create(
            project=self.project,
            created_by=self.user,
            format="coco",
            status=ExportTask.STATUS_COMPLETED,
            expires_at=None,
        )
        ExportTaskResult.objects.create(task=task)

        _cleanup_expired_exports()

        task.refresh_from_db()
        self.assertEqual(task.status, ExportTask.STATUS_COMPLETED)
