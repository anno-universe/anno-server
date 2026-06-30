import json

from django.contrib.auth.models import Group
from django.contrib.auth import get_user_model
from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import Client, TestCase
from ninja_jwt.tokens import RefreshToken

from anno_images.models import Image2D
from anno_projects.models import Project, ProjectMembership

from .models import ProjectTag, ImageTag

User = get_user_model()


def _jwt_headers(user):
    access = RefreshToken.for_user(user).access_token
    return {"Authorization": f"Bearer {access}"}


# Minimal 1x1 white PNG for Image2D creation.
_ONE_PX_PNG = (
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
    b"\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\x0cIDATx\x9cc\xf8\x0f"
    b"\x00\x00\x01\x01\x00\x05\x18\xd8N\x00\x00\x00\x00IEND\xaeB`\x82"
)


class ImageTagRemovalTests(TestCase):
    def setUp(self):
        self.client = Client()

        # Users
        self.worker_a = User.objects.create_user(username="worker_a", password="x")
        self.worker_b = User.objects.create_user(username="worker_b", password="x")
        self.supervisor = User.objects.create_user(username="sup", password="x")
        self.other_supervisor = User.objects.create_user(
            username="sup2", password="x"
        )
        self.outsider = User.objects.create_user(username="out", password="x")

        # System admin
        self.admin = User.objects.create_user(username="admin", password="x")
        admin_group, _ = Group.objects.get_or_create(name="admin")
        self.admin.groups.add(admin_group)

        # Project — supervisor is creator
        self.project = Project.objects.create(
            name="Test Project", created_by=self.supervisor
        )

        # Memberships — note: the creator (self.supervisor) is auto-added
        # as a supervisor by the post_save signal on Project, so we only
        # need to create memberships for the other users.
        ProjectMembership.objects.create(
            user=self.worker_a, project=self.project, role="worker",
            added_by=self.supervisor,
        )
        ProjectMembership.objects.create(
            user=self.worker_b, project=self.project, role="worker",
            added_by=self.supervisor,
        )
        ProjectMembership.objects.create(
            user=self.other_supervisor, project=self.project, role="supervisor",
            added_by=self.supervisor,
        )

        # Image
        self.image = Image2D.objects.create(
            project=self.project,
            image=SimpleUploadedFile("test.png", _ONE_PX_PNG, content_type="image/png"),
            file_name="test.png",
        )

        # Project tags (definitions)
        self.tag_a = ProjectTag.objects.create(
            project=self.project,
            name="worker_finish",
            display_name="Worker Finish",
            created_by=self.supervisor,
        )
        self.tag_b = ProjectTag.objects.create(
            project=self.project,
            name="worker_question",
            display_name="Worker Question",
            created_by=self.supervisor,
        )
        self.tag_s = ProjectTag.objects.create(
            project=self.project,
            name="supervisor_review",
            display_name="Supervisor Review",
            created_by=self.supervisor,
        )

        # ImageTags applied by each user
        self.tag_by_worker_a = ImageTag.objects.create(
            image=self.image, tag=self.tag_a, applied_by=self.worker_a,
        )
        self.tag_by_worker_b = ImageTag.objects.create(
            image=self.image, tag=self.tag_b, applied_by=self.worker_b,
        )
        self.tag_by_supervisor = ImageTag.objects.create(
            image=self.image, tag=self.tag_s, applied_by=self.supervisor,
        )

    # ------------------------------------------------------------------
    # URL helper
    # ------------------------------------------------------------------

    def _url(self, tag_id):
        return (
            f"/api/projects/{self.project.id}"
            f"/images/{self.image.id}"
            f"/tags/{tag_id}"
        )

    # ------------------------------------------------------------------
    # Worker tests
    # ------------------------------------------------------------------

    def test_worker_removes_own_tag(self):
        res = self.client.delete(
            self._url(self.tag_a.id), headers=_jwt_headers(self.worker_a)
        )
        self.assertEqual(res.status_code, 204)
        self.assertFalse(
            ImageTag.objects.filter(
                image=self.image, tag=self.tag_a
            ).exists()
        )

    def test_worker_cannot_remove_other_worker_tag(self):
        res = self.client.delete(
            self._url(self.tag_b.id), headers=_jwt_headers(self.worker_a)
        )
        self.assertEqual(res.status_code, 403)
        self.assertTrue(
            ImageTag.objects.filter(
                image=self.image, tag=self.tag_b
            ).exists()
        )

    def test_worker_cannot_remove_supervisor_tag(self):
        res = self.client.delete(
            self._url(self.tag_s.id), headers=_jwt_headers(self.worker_a)
        )
        self.assertEqual(res.status_code, 403)
        self.assertTrue(
            ImageTag.objects.filter(
                image=self.image, tag=self.tag_s
            ).exists()
        )

    # ------------------------------------------------------------------
    # Supervisor tests
    # ------------------------------------------------------------------

    def test_supervisor_removes_worker_tag(self):
        res = self.client.delete(
            self._url(self.tag_a.id), headers=_jwt_headers(self.supervisor)
        )
        self.assertEqual(res.status_code, 204)
        self.assertFalse(
            ImageTag.objects.filter(
                image=self.image, tag=self.tag_a
            ).exists()
        )

    def test_supervisor_removes_own_tag(self):
        res = self.client.delete(
            self._url(self.tag_s.id), headers=_jwt_headers(self.supervisor)
        )
        self.assertEqual(res.status_code, 204)
        self.assertFalse(
            ImageTag.objects.filter(
                image=self.image, tag=self.tag_s
            ).exists()
        )

    def test_supervisor_removes_other_supervisor_tag(self):
        """Supervisor B removes a tag applied by Supervisor A."""
        # Create a tag applied by other_supervisor
        other_tag = ProjectTag.objects.create(
            project=self.project,
            name="sup2_review",
            display_name="Sup2 Review",
            created_by=self.supervisor,
        )
        ImageTag.objects.create(
            image=self.image, tag=other_tag, applied_by=self.other_supervisor,
        )
        res = self.client.delete(
            self._url(other_tag.id), headers=_jwt_headers(self.supervisor)
        )
        self.assertEqual(res.status_code, 204)
        self.assertFalse(
            ImageTag.objects.filter(
                image=self.image, tag=other_tag
            ).exists()
        )

    # ------------------------------------------------------------------
    # Admin tests
    # ------------------------------------------------------------------

    def test_admin_removes_any_tag(self):
        """System admin can remove any tag regardless of membership."""
        res = self.client.delete(
            self._url(self.tag_a.id), headers=_jwt_headers(self.admin)
        )
        self.assertEqual(res.status_code, 204)
        self.assertFalse(
            ImageTag.objects.filter(
                image=self.image, tag=self.tag_a
            ).exists()
        )

    # ------------------------------------------------------------------
    # Non-member test
    # ------------------------------------------------------------------

    def test_non_member_cannot_remove(self):
        res = self.client.delete(
            self._url(self.tag_a.id), headers=_jwt_headers(self.outsider)
        )
        self.assertIn(res.status_code, (401, 403))
        self.assertTrue(
            ImageTag.objects.filter(
                image=self.image, tag=self.tag_a
            ).exists()
        )

    # ------------------------------------------------------------------
    # Edge cases
    # ------------------------------------------------------------------

    def test_worker_removes_nonexistent_tag_returns_404(self):
        res = self.client.delete(
            self._url(99999), headers=_jwt_headers(self.worker_a)
        )
        self.assertEqual(res.status_code, 404)

    def test_worker_removes_tag_from_nonexistent_image_returns_404(self):
        url = (
            f"/api/projects/{self.project.id}"
            f"/images/99999"
            f"/tags/{self.tag_a.id}"
        )
        res = self.client.delete(url, headers=_jwt_headers(self.worker_a))
        self.assertEqual(res.status_code, 404)
