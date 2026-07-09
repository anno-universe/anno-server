import logging
import sys

import boto3
from django.apps import AppConfig
from django.conf import settings

logger = logging.getLogger(__name__)


class AnnoImagesConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "anno_images"

    def ready(self):
        import anno_images.signals  # noqa: F401
        self._ensure_bucket_exists()

    def _ensure_bucket_exists(self):
        if "migrate" in sys.argv or "makemigrations" in sys.argv:
            return

        bucket = settings.AWS_STORAGE_BUCKET_NAME
        try:
            s3 = boto3.client(
                "s3",
                endpoint_url=settings.AWS_S3_ENDPOINT_URL,
                aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
                aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
                region_name=settings.AWS_S3_REGION_NAME,
            )
            try:
                s3.head_bucket(Bucket=bucket)
            except s3.exceptions.ClientError as exc:
                if exc.response["Error"]["Code"] == "404":
                    s3.create_bucket(Bucket=bucket)
                    logger.info("Created S3 bucket %r at %s", bucket, settings.AWS_S3_ENDPOINT_URL)
                else:
                    raise
        except Exception:
            logger.warning(
                "Could not verify or create S3 bucket %r at %s. "
                "Uploads will fail until the bucket exists.",
                bucket,
                settings.AWS_S3_ENDPOINT_URL,
                exc_info=True,
            )
