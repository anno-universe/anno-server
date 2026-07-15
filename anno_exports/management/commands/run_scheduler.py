import logging

from django.conf import settings
from django.core.management.base import BaseCommand
from django.db.models import Q
from django.utils import timezone

logger = logging.getLogger(__name__)


def _cleanup_expired_exports():
    from anno_exports.models import ExportTask

    expired = ExportTask.objects.filter(
        ~Q(expires_at=None),
        expires_at__lt=timezone.now(),
        status=ExportTask.STATUS_COMPLETED,
    ).select_related("result")

    cleaned = 0
    for task in expired:
        result = getattr(task, "result", None)
        if result is None or not result.export_file:
            continue
        try:
            result.export_file.delete(save=False)
            result.export_file = None
            result.file_deleted_at = timezone.now()
            result.save(update_fields=["export_file", "file_deleted_at"])
            cleaned += 1
        except Exception:
            logger.warning(
                "Failed to clean up export file for task %d", task.id, exc_info=True
            )

    if cleaned:
        logger.info("Cleaned up %d expired export file(s).", cleaned)


class Command(BaseCommand):
    help = "Run the background scheduler (periodic cleanup of expired exports, etc.)."

    def handle(self, **options):
        from django_apscheduler.jobstores import DjangoJobStore
        from apscheduler.schedulers.blocking import BlockingScheduler

        interval = getattr(settings, "EXPORT_CLEANUP_INTERVAL_MINUTES", 10)

        scheduler = BlockingScheduler()
        scheduler.add_jobstore(DjangoJobStore(), "default")

        scheduler.add_job(
            _cleanup_expired_exports,
            "interval",
            minutes=interval,
            id="cleanup_expired_exports",
            replace_existing=True,
            max_instances=1,
        )

        self.stdout.write(
            self.style.SUCCESS(
                f"Export cleanup scheduler started (interval={interval} min)."
            )
        )
        logger.info(
            "Export cleanup scheduler started (interval=%d min).", interval
        )

        try:
            scheduler.start()
        except (KeyboardInterrupt, SystemExit):
            logger.info("Export cleanup scheduler stopped.")
