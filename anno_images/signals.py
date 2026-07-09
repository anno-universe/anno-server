from django.db.models.signals import post_delete, pre_save
from django.dispatch import receiver

from anno_projects.models import Project

from .models import Annotation2D, Image2D


@receiver(pre_save, sender=Image2D)
def populate_image_metadata(sender, instance, **kwargs):
    if instance.image and not instance.file_name:
        instance.file_name = instance.image.name.rsplit("/", 1)[-1]


@receiver(pre_save, sender=Annotation2D)
def enforce_denormalized_project(sender, instance, **kwargs):
    if instance.image_id and not instance.project_id:
        instance.project = instance.image.project


@receiver(post_delete, sender=Project)
def cleanup_project_files(sender, instance, **kwargs):
    for image in instance.images.all():
        image.image.delete(save=False)
