from django.db.models.signals import post_save
from django.dispatch import receiver

from .models import Project, ProjectMembership


@receiver(post_save, sender=Project)
def auto_add_creator_as_supervisor(sender, instance, created, **kwargs):
    if created:
        ProjectMembership.objects.create(
            user=instance.created_by,
            project=instance,
            role="supervisor",
        )


# The last-supervisor rule now lives in ProjectMembership.delete() — a pre_delete
# signal no longer fires because membership removal is a soft delete (a save).
