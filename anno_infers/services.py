from anno_images.models import Annotation2D, Box2D, Keypoint2D, Polygon2D, Operation


def _create_subtype(annotation, *, polygon=None, box=None, keypoint=None):
    """Create the subtype row for an annotation, mirroring
    ``Annotation2DController._create_subtype`` in anno_images."""
    if annotation.annotation_type == "polygon" and polygon is not None:
        Polygon2D.objects.create(annotation=annotation, points=polygon.points)
    elif annotation.annotation_type == "box" and box is not None:
        Box2D.objects.create(
            annotation=annotation,
            x=box.x,
            y=box.y,
            width=box.width,
            height=box.height,
            rotation=box.rotation,
        )
    elif annotation.annotation_type == "keypoint" and keypoint is not None:
        Keypoint2D.objects.create(annotation=annotation, points=keypoint.points)
    else:
        raise ValueError(
            "Missing or mismatched subtype data for "
            f"annotation_type='{annotation.annotation_type}'"
        )


def create_ai_annotation(
    *,
    image,
    project,
    annotation_type,
    label=None,
    polygon=None,
    box=None,
    keypoint=None,
    performed_by,
):
    """Create one AI-contributed annotation, identical to the human write-path.

    Mirrors ``Annotation2DController.create``: creates the ``Annotation2D``, its
    subtype row, and an ``Operation(action="add")`` audit record. AI annotations
    are deliberately indistinguishable from human ones. The caller is responsible
    for the surrounding transaction / savepoint.
    """
    annotation = Annotation2D.objects.create(
        image=image,
        project=project,
        annotation_type=annotation_type,
        label=label,
    )
    _create_subtype(annotation, polygon=polygon, box=box, keypoint=keypoint)
    Operation.objects.create(
        image=image,
        to_annotation=annotation,
        action="add",
        performed_by=performed_by,
    )
    return annotation
