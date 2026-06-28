from django.core.exceptions import ValidationError
from django.db.models.signals import post_save, pre_delete
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


@receiver(pre_delete, sender=ProjectMembership)
def prevent_last_supervisor_removal(sender, instance, **kwargs):
    if instance.role == "supervisor":
        remaining = ProjectMembership.objects.filter(
            project=instance.project, role="supervisor"
        ).exclude(pk=instance.pk).exists()
        if not remaining:
            raise ValidationError(
                "Cannot remove the last supervisor from a project."
            )
