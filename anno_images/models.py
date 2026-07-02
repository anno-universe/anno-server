from django.conf import settings
from django.db import models

from anno_projects.models import Project


def _image_upload_to(instance, filename):
    return f"images/{instance.project_id}/{filename}"


class Image2D(models.Model):
    project = models.ForeignKey(
        Project,
        on_delete=models.CASCADE,
        related_name="images",
    )
    image = models.ImageField(upload_to=_image_upload_to)
    file_name = models.CharField(max_length=255, blank=True, default="")
    width = models.PositiveIntegerField(null=True, blank=True)
    height = models.PositiveIntegerField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        db_table = "anno_image_2d"
        ordering = ["-created_at"]
        verbose_name = "image 2D"
        verbose_name_plural = "images 2D"

    def __str__(self):
        return self.file_name or f"Image #{self.id}"


class Annotation2D(models.Model):
    ANNOTATION_TYPE_CHOICES = [
        ("polygon", "Polygon"),
        ("box", "Box"),
        ("keypoint", "Keypoint"),
    ]

    image = models.ForeignKey(
        Image2D,
        on_delete=models.CASCADE,
        related_name="annotations",
    )
    project = models.ForeignKey(
        Project,
        on_delete=models.CASCADE,
        related_name="annotations",
        help_text="Denormalized FK for fast project-scoped queries.",
    )
    annotation_type = models.CharField(max_length=20, choices=ANNOTATION_TYPE_CHOICES)
    label = models.IntegerField(
        null=True,
        blank=True,
        help_text="Numeric classification label (references Project.label_mapping).",
    )
    is_active = models.BooleanField(
        default=True,
        help_text="Whether this is the current version of the annotation.",
    )
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        db_table = "anno_annotation_2d"
        ordering = ["-created_at"]
        verbose_name = "annotation 2D"
        verbose_name_plural = "annotations 2D"
        indexes = [
            models.Index(fields=["image", "is_active"]),
            models.Index(fields=["image", "annotation_type", "is_active"]),
            models.Index(fields=["project", "is_active"]),
        ]

    def __str__(self):
        type_label = dict(self.ANNOTATION_TYPE_CHOICES).get(self.annotation_type, self.annotation_type)
        return f"{type_label} annotation #{self.id}"


class Polygon2D(models.Model):
    annotation = models.OneToOneField(
        Annotation2D,
        on_delete=models.CASCADE,
        primary_key=True,
        related_name="polygon",
    )
    points = models.JSONField(
        help_text="List of polygon vertices as [[x1,y1], [x2,y2], ...].",
    )

    class Meta:
        db_table = "anno_polygon_2d"
        verbose_name = "polygon 2D"
        verbose_name_plural = "polygons 2D"

    def __str__(self):
        return f"Polygon for annotation #{self.annotation_id}"


class Box2D(models.Model):
    annotation = models.OneToOneField(
        Annotation2D,
        on_delete=models.CASCADE,
        primary_key=True,
        related_name="box",
    )
    x = models.FloatField()
    y = models.FloatField()
    width = models.FloatField()
    height = models.FloatField()
    rotation = models.FloatField(
        null=True,
        blank=True,
        default=0.0,
        help_text="Rotation angle in degrees clockwise.",
    )

    class Meta:
        db_table = "anno_box_2d"
        verbose_name = "box 2D"
        verbose_name_plural = "boxes 2D"

    def __str__(self):
        return f"Box for annotation #{self.annotation_id}"


class Keypoint2D(models.Model):
    annotation = models.OneToOneField(
        Annotation2D,
        on_delete=models.CASCADE,
        primary_key=True,
        related_name="keypoint",
    )
    points = models.JSONField(
        help_text="List of keypoint coordinates as [[x1,y1], [x2,y2], ...].",
    )

    class Meta:
        db_table = "anno_keypoint_2d"
        verbose_name = "keypoint 2D"
        verbose_name_plural = "keypoints 2D"

    def __str__(self):
        return f"Keypoint #{self.annotation_id}"


class Operation(models.Model):
    ACTION_CHOICES = [
        ("add", "Add"),
        ("modify", "Modify"),
        ("delete", "Delete"),
    ]

    # Where the operation originated. ``action`` stays orthogonal (what changed);
    # ``source`` records who/what produced it. AI operations are reverse-traceable
    # from (source, to_annotation_id): inference -> InferenceResult.annotation,
    # interactive -> InteractiveInferenceSession.final_annotation.
    SOURCE_HUMAN = "human"
    SOURCE_INFERENCE = "inference"
    SOURCE_INTERACTIVE = "interactive"
    SOURCE_CHOICES = [
        (SOURCE_HUMAN, "Human"),
        (SOURCE_INFERENCE, "Inference"),
        (SOURCE_INTERACTIVE, "Interactive"),
    ]

    image = models.ForeignKey(
        Image2D,
        on_delete=models.CASCADE,
        related_name="operations",
    )
    from_annotation = models.ForeignKey(
        Annotation2D,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="operations_as_from",
        help_text="The annotation before the change (null for 'add' operations).",
    )
    to_annotation = models.ForeignKey(
        Annotation2D,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="operations_as_to",
        help_text="The annotation after the change (null for 'delete' operations).",
    )
    action = models.CharField(max_length=10, choices=ACTION_CHOICES)
    source = models.CharField(
        max_length=12,
        choices=SOURCE_CHOICES,
        default=SOURCE_HUMAN,
        db_index=True,
        help_text="Origin of the operation: human, inference or interactive.",
    )
    performed_by = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.PROTECT,
        related_name="operations",
    )
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = "anno_operation"
        ordering = ["-created_at"]
        verbose_name = "operation"
        verbose_name_plural = "operations"
        indexes = [
            models.Index(fields=["image", "created_at"]),
            models.Index(fields=["from_annotation"]),
            models.Index(fields=["to_annotation"]),
            models.Index(fields=["source", "to_annotation"]),
        ]

    def __str__(self):
        return f"Operation #{self.id}: {self.action} on image #{self.image_id}"
