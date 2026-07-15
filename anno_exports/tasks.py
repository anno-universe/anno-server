from __future__ import annotations

import json
import logging
import os
import shutil
import tempfile
import zipfile
from pathlib import Path

from django.core.files import File
from django.utils import timezone
from django_tasks import task

from .models import ExportTask, ExportTaskResult
from .services import build_coco, build_yolo, load_export_data

logger = logging.getLogger(__name__)


@task()
def run_export(task_id: int) -> None:
    execute_export(task_id)


def execute_export(task_id: int) -> None:
    logger.info("Worker picked up export task %d", task_id)

    task = (
        ExportTask.objects.select_related("project")
        .filter(pk=task_id)
        .first()
    )
    if task is None or task.status not in (
        ExportTask.STATUS_PENDING,
        ExportTask.STATUS_RUNNING,
    ):
        logger.info(
            "Export task %d skipped: task=%s status=%s",
            task_id,
            "found" if task else "not_found",
            task.status if task else "n/a",
        )
        return

    logger.info(
        "Export task %d starting: project=%s format=%s include_images=%s",
        task_id,
        task.project.name,
        task.format,
        task.include_images,
    )

    task.status = ExportTask.STATUS_RUNNING
    task.started_at = timezone.now()
    task.error = ""
    task.save(update_fields=["status", "started_at", "error"])

    try:
        images, annotations = load_export_data(task.project_id)
    except Exception as exc:
        logger.error("Export task %d: data loading failed: %s", task_id, exc, exc_info=True)
        task.status = ExportTask.STATUS_FAILED
        task.error = str(exc)
        task.finished_at = timezone.now()
        task.save(update_fields=["status", "error", "finished_at"])
        return

    if len(images) == 0:
        task.status = ExportTask.STATUS_FAILED
        task.error = "No images with active annotations found."
        task.finished_at = timezone.now()
        task.save(update_fields=["status", "error", "finished_at"])
        return

    label_mapping = task.project.label_mapping or {}
    include_images = task.include_images

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)

            if task.format == ExportTask.FORMAT_COCO:
                coco = build_coco(images, annotations, label_mapping)
                coco_path = tmp / "annotations.json"
                coco_path.write_text(json.dumps(coco, ensure_ascii=False, indent=2), encoding="utf-8")
            else:
                yolo = build_yolo(images, annotations, label_mapping)
                labels_dir = tmp / "labels"
                labels_dir.mkdir()
                for img in images:
                    lines = yolo["labels"].get(img.id, [])
                    if not lines:
                        continue
                    stem = Path(img.file_name or f"image_{img.id}").stem
                    label_file = labels_dir / f"{stem}.txt"
                    label_file.write_text("\n".join(lines) + "\n", encoding="utf-8")
                if yolo["classes_txt"]:
                    (tmp / "classes.txt").write_text(yolo["classes_txt"], encoding="utf-8")

            if include_images:
                images_dir = tmp / "images"
                images_dir.mkdir()
                for img in images:
                    src = img.image.path
                    if not os.path.isfile(src):
                        continue
                    stem = Path(img.file_name or f"image_{img.id}").stem
                    ext = os.path.splitext(img.file_name or src)[1] or ".png"
                    dst = images_dir / f"{stem}{ext}"
                    shutil.copy2(src, dst)

            project_name = task.project.name.replace(" ", "_")
            ts = timezone.now().strftime("%Y%m%dT%H%M%S")
            zip_name = f"{project_name}_{task.format}_{ts}.zip"
            zip_path = tmp / zip_name

            with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
                for f in sorted(tmp.rglob("*")):
                    if f == zip_path or f.is_dir():
                        continue
                    arcname = str(f.relative_to(tmp))
                    zf.write(f, arcname)

            with open(zip_path, "rb") as fh:
                result = ExportTaskResult(task=task)
                result.export_file.save(zip_name, File(fh), save=False)
                result.file_size = os.path.getsize(zip_path)
                result.save()

        task.status = ExportTask.STATUS_COMPLETED
        logger.info("Export task %d completed: %d images", task_id, len(images))

    except Exception as exc:
        logger.error("Export task %d failed: %s", task_id, exc, exc_info=True)
        task.status = ExportTask.STATUS_FAILED
        task.error = str(exc)

    task.finished_at = timezone.now()
    task.save(update_fields=["status", "error", "finished_at"])
