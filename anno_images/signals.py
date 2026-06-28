import boto3
from botocore.client import Config
from django.conf import settings
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
def cleanup_project_s3_files(sender, instance, **kwargs):
    s3 = boto3.client(
        "s3",
        endpoint_url=settings.AWS_S3_ENDPOINT_URL,
        aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
        aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
        config=Config(signature_version="s3v4"),
        region_name=settings.AWS_S3_REGION_NAME,
    )
    prefix = f"images/{instance.pk}/"
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(
        Bucket=settings.AWS_STORAGE_BUCKET_NAME, Prefix=prefix
    ):
        objects = page.get("Contents", [])
        if objects:
            s3.delete_objects(
                Bucket=settings.AWS_STORAGE_BUCKET_NAME,
                Delete={"Objects": [{"Key": obj["Key"]} for obj in objects]},
            )
