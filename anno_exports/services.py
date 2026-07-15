import json
import math
import os
from pathlib import Path


def _invert_label_mapping(label_mapping: dict) -> dict[int, str]:
    result: dict[int, str] = {}

    labels = label_mapping.get("labels") if isinstance(label_mapping.get("labels"), dict) else None
    if labels is not None:
        source = labels
    else:
        source = label_mapping

    for name, value in source.items():
        if isinstance(value, dict):
            class_id = value.get("id") if "id" in value else value.get("class_id")
            if isinstance(class_id, int):
                result[class_id] = name
        elif isinstance(value, int):
            result[value] = name

    return result


def _rotate_point(px: float, py: float, cx: float, cy: float, cos_a: float, sin_a: float):
    dx, dy = px - cx, py - cy
    return (cx + dx * cos_a - dy * sin_a, cy + dx * sin_a + dy * cos_a)


def _box_to_corners(box):
    """Box2D -> 4 corners, rotation around CENTER (matching anno-web frontend)."""
    cx = box.x + box.width / 2
    cy = box.y + box.height / 2
    hw = box.width / 2
    hh = box.height / 2
    rot = box.rotation or 0
    if abs(rot) < 1e-6:
        return [
            (box.x, box.y),
            (box.x + box.width, box.y),
            (box.x + box.width, box.y + box.height),
            (box.x, box.y + box.height),
        ]
    rad = math.radians(rot)
    c, s = math.cos(rad), math.sin(rad)
    local = [(-hw, -hh), (hw, -hh), (hw, hh), (-hw, hh)]
    return [_rotate_point(cx + dx, cy + dy, cx, cy, c, s) for dx, dy in local]


def _extent_from_corners(corners: list) -> list:
    xs = [c[0] for c in corners]
    ys = [c[1] for c in corners]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    return [min_x, min_y, max_x - min_x, max_y - min_y]


# ---------------------------------------------------------------------------
# COCO builder
# ---------------------------------------------------------------------------


def build_coco(images, annotations, label_mapping: dict) -> dict:
    label_names = _invert_label_mapping(label_mapping)

    coco_images = []
    coco_annotations = []
    coco_categories = []

    annotation_id = 1
    for image in images:
        coco_images.append({
            "id": image.id,
            "file_name": image.file_name or os.path.basename(image.image.name or "") or f"image_{image.id}",
            "width": image.width,
            "height": image.height,
        })

    for cat_id in sorted(label_names.keys()):
        coco_categories.append({"id": cat_id, "name": label_names[cat_id], "supercategory": ""})

    for annotation in annotations:
        bbox = None
        seg = None
        area = 0.0
        keypoints = None
        num_keypoints = None

        if annotation.annotation_type == "polygon" and hasattr(annotation, "polygon"):
            pts = annotation.polygon.points
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            bbox = [min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys)]
            seg = [[c for p in pts for c in p]]
            n = len(xs)
            area = 0.5 * abs(sum(
                xs[i] * ys[(i + 1) % n] - xs[(i + 1) % n] * ys[i]
                for i in range(n)
            ))

        elif annotation.annotation_type == "box" and hasattr(annotation, "box"):
            box = annotation.box
            area = box.width * box.height
            if abs(box.rotation or 0) < 1e-6:
                bbox = [box.x, box.y, box.width, box.height]
            else:
                corners = _box_to_corners(box)
                seg = [[c for corner in corners for c in corner]]
                bbox = _extent_from_corners(corners)

        elif annotation.annotation_type == "keypoint" and hasattr(annotation, "keypoint"):
            pts = annotation.keypoint.points
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            bbox = [min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys)]
            keypoints = []
            num_keypoints = 0
            for kp in pts:
                keypoints.extend([kp[0], kp[1], 2])
                num_keypoints += 1
            area = bbox[2] * bbox[3]

        coco_ann = {
            "id": annotation_id,
            "image_id": annotation.image_id,
            "category_id": annotation.label,
            "bbox": bbox or [0, 0, 0, 0],
            "area": area,
            "iscrowd": 0,
        }
        if seg:
            coco_ann["segmentation"] = seg
        if keypoints:
            coco_ann["keypoints"] = keypoints
            coco_ann["num_keypoints"] = num_keypoints

        coco_annotations.append(coco_ann)
        annotation_id += 1

    return {
        "images": coco_images,
        "annotations": coco_annotations,
        "categories": coco_categories,
    }


# ---------------------------------------------------------------------------
# YOLO builder
# ---------------------------------------------------------------------------


def build_yolo(images, annotations, label_mapping: dict) -> dict:
    label_names = _invert_label_mapping(label_mapping)

    image_labels: dict[int, list[str]] = {}
    for image in images:
        image_labels[image.id] = []

    for annotation in annotations:
        img_id = annotation.image_id
        if img_id not in image_labels:
            continue

        label = annotation.label
        if label is None:
            continue

        image = next((img for img in images if img.id == img_id), None)
        if image is None or not image.width or not image.height:
            continue
        W, H = image.width, image.height

        if annotation.annotation_type == "polygon" and hasattr(annotation, "polygon"):
            pts = annotation.polygon.points
            parts = [str(label)]
            for p in pts:
                parts.append(f"{p[0] / W:.6f}")
                parts.append(f"{p[1] / H:.6f}")
            image_labels[img_id].append(" ".join(parts))

        elif annotation.annotation_type == "box" and hasattr(annotation, "box"):
            box = annotation.box
            if abs(box.rotation or 0) < 1e-6:
                cx = (box.x + box.width / 2) / W
                cy = (box.y + box.height / 2) / H
                nw = box.width / W
                nh = box.height / H
                image_labels[img_id].append(f"{label} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
            else:
                corners = _box_to_corners(box)
                parts = [str(label)]
                for c in corners:
                    parts.append(f"{c[0] / W:.6f}")
                    parts.append(f"{c[1] / H:.6f}")
                image_labels[img_id].append(" ".join(parts))

        elif annotation.annotation_type == "keypoint":
            continue

    class_lines = "".join(f"{label_names[cat_id]}\n" for cat_id in sorted(label_names.keys()))

    return {"labels": image_labels, "classes_txt": class_lines}


# ---------------------------------------------------------------------------
# Data loader
# ---------------------------------------------------------------------------


def load_export_data(project_id: int):
    from anno_images.models import Image2D, Annotation2D

    images = list(
        Image2D.objects.filter(
            project_id=project_id,
            annotations__is_active=True,
        )
        .distinct()
        .order_by("id")
    )
    if not images:
        return [], []

    image_ids = [img.id for img in images]
    annotations = list(
        Annotation2D.objects.filter(
            image_id__in=image_ids,
            is_active=True,
        )
        .select_related("polygon", "box", "keypoint")
        .order_by("image_id", "id")
    )
    return images, annotations
