from django.apps import apps
from django.contrib.auth.models import Group, Permission
from django.db.models.signals import post_migrate
from django.dispatch import receiver

GROUPS_PERMISSIONS = {
    "admin": {
        "is_admin": True,
    },
    "user": {
        "is_admin": False,
        "permissions": [],
    },
}


@receiver(post_migrate)
def create_auth_groups(sender, **kwargs):
    if sender.name != "anno_users":
        return

    for group_name, config in GROUPS_PERMISSIONS.items():
        group, created = Group.objects.get_or_create(name=group_name)

        if config.get("is_admin"):
            group.permissions.set(Permission.objects.all())
        elif config["permissions"]:
            for codename in config["permissions"]:
                try:
                    perm = Permission.objects.get(codename=codename)
                    group.permissions.add(perm)
                except Permission.DoesNotExist:
                    pass

    Group.objects.filter(name__in=["supervisor", "worker"]).delete()
