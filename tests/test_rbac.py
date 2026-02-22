"""
Tests for RBAC (Role-Based Access Control)
اختبارات نظام التحكم بالصلاحيات
"""

import pytest
import os


class TestRBACModule:
    """Test the RBAC module itself"""

    def test_role_permissions_mapping(self):
        """Each role should have defined permissions"""
        from api.rbac import Role, ROLE_PERMISSIONS

        for role in Role:
            assert role in ROLE_PERMISSIONS, f"Role {role} missing from ROLE_PERMISSIONS"
            assert len(ROLE_PERMISSIONS[role]) > 0, f"Role {role} has no permissions"

    def test_viewer_has_read_only(self):
        """Viewer role should only have read permissions"""
        from api.rbac import Role, ROLE_PERMISSIONS

        viewer_perms = ROLE_PERMISSIONS[Role.VIEWER]
        for perm in viewer_perms:
            assert "write" not in perm.value, f"Viewer shouldn't have write: {perm}"
            assert "create" not in perm.value, f"Viewer shouldn't have create: {perm}"
            assert "edit" not in perm.value, f"Viewer shouldn't have edit: {perm}"
            assert "delete" not in perm.value, f"Viewer shouldn't have delete: {perm}"
            assert "manage" not in perm.value, f"Viewer shouldn't have manage: {perm}"

    def test_admin_has_all_permissions(self):
        """Admin role should have all permissions"""
        from api.rbac import Role, Permission, ROLE_PERMISSIONS

        admin_perms = ROLE_PERMISSIONS[Role.ADMIN]
        for perm in Permission:
            assert perm in admin_perms, f"Admin missing permission: {perm}"

    def test_president_has_all_permissions(self):
        """President role should have all permissions"""
        from api.rbac import Role, Permission, ROLE_PERMISSIONS

        president_perms = ROLE_PERMISSIONS[Role.PRESIDENT]
        for perm in Permission:
            assert perm in president_perms, f"President missing permission: {perm}"

    def test_user_has_permission(self):
        """user_has_permission should correctly check permissions"""
        from api.rbac import user_has_permission, Permission

        admin_user = {"role": "admin"}
        viewer_user = {"role": "viewer"}

        assert user_has_permission(admin_user, Permission.SYSTEM_USERS_MANAGE)
        assert not user_has_permission(viewer_user, Permission.SYSTEM_USERS_MANAGE)
        assert user_has_permission(viewer_user, Permission.ERP_INVOICES_READ)

    def test_extra_permissions(self):
        """Users with extra permissions should have those checked"""
        from api.rbac import user_has_permission, Permission

        user_with_extra = {
            "role": "viewer",
            "permissions": ["system:users:manage"],
        }
        # Viewer normally doesn't have this, but extra permissions grant it
        assert user_has_permission(user_with_extra, Permission.SYSTEM_USERS_MANAGE)

    def test_get_all_roles(self):
        """get_all_roles should return structured data"""
        from api.rbac import get_all_roles

        roles = get_all_roles()
        assert len(roles) > 0
        for role_info in roles:
            assert "role" in role_info
            assert "permissions" in role_info
            assert "permission_count" in role_info

    def test_get_all_permissions(self):
        """get_all_permissions should return grouped permissions"""
        from api.rbac import get_all_permissions

        perms = get_all_permissions()
        assert len(perms) > 0
        for group in perms:
            assert "group" in group
            assert "permissions" in group


class TestAdminEndpoints:
    """Test admin/user management endpoints"""

    def test_list_roles(self, client):
        """Should list all available roles"""
        response = client.get("/api/v1/admin/roles")
        assert response.status_code == 200

        data = response.json()
        assert "roles" in data
        assert "total" in data
        assert data["total"] > 0

    def test_list_permissions(self, client):
        """Should list all available permissions"""
        response = client.get("/api/v1/admin/permissions")
        assert response.status_code == 200

        data = response.json()
        assert "permissions" in data
        assert "total" in data
        assert data["total"] > 0

    def test_list_users(self, client):
        """Should list registered users"""
        response = client.get("/api/v1/admin/users")
        assert response.status_code == 200

        data = response.json()
        assert "users" in data
        assert any(u["username"] == "president" for u in data["users"])

    def test_get_current_user_info(self, client):
        """Should return current user's info and permissions"""
        response = client.get("/api/v1/admin/me")
        assert response.status_code == 200

        data = response.json()
        assert "username" in data
        assert "role" in data
        assert "role_permissions" in data

    def test_create_user_with_valid_data(self, client):
        """Should create a new user"""
        response = client.post(
            "/api/v1/admin/users",
            json={
                "username": "test_user_rbac",
                "password": "test123",
                "full_name": "Test User",
                "role": "viewer",
            },
        )
        assert response.status_code == 200

        data = response.json()
        assert data["username"] == "test_user_rbac"
        assert data["role"] == "viewer"

    def test_create_duplicate_user_fails(self, client):
        """Should fail when creating a user that already exists"""
        response = client.post(
            "/api/v1/admin/users",
            json={
                "username": "president",
                "password": "test",
                "role": "viewer",
            },
        )
        assert response.status_code == 400

    def test_create_user_with_invalid_role(self, client):
        """Should fail with invalid role"""
        response = client.post(
            "/api/v1/admin/users",
            json={
                "username": "test_invalid_role",
                "password": "test",
                "role": "supreme_leader",  # invalid
            },
        )
        assert response.status_code == 400

    def test_delete_president_fails(self, client):
        """Cannot delete the president account"""
        response = client.delete("/api/v1/admin/users/president")
        assert response.status_code == 403
