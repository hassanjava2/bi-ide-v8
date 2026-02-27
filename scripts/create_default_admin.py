#!/usr/bin/env python3
"""Create default admin user if not exists

Usage:
    python scripts/create_default_admin.py
    python scripts/create_default_admin.py --username president --email admin@bi-ide.com
"""

import asyncio
import argparse
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.database import db_manager
from core.user_service import UserService, RoleService, DEFAULT_ROLES
from core.config import settings


async def init_roles():
    """Initialize default roles if not exist"""
    async with db_manager.get_session() as db:
        role_service = RoleService(db)
        
        print("🔧 Checking default roles...")
        
        created_count = 0
        for role_name, role_data in DEFAULT_ROLES.items():
            from sqlalchemy import select
            from core.user_models import RoleDB
            
            result = await db.execute(
                select(RoleDB).where(RoleDB.name == role_name)
            )
            existing = result.scalar_one_or_none()
            
            if existing:
                continue
            
            await role_service.create_role(
                name=role_name,
                description=role_data["description"],
                permissions=role_data["permissions"]
            )
            created_count += 1
        
        if created_count > 0:
            print(f"  ✓ Created {created_count} roles")
        else:
            print("  ✓ All roles already exist")


async def create_admin(username: str = "president", email: str = "admin@bi-ide.com"):
    """Create default admin user if not exists"""
    await db_manager.initialize()
    
    # Initialize roles first
    await init_roles()
    
    async with db_manager.get_session() as session:
        user_service = UserService(session)
        
        # Check if admin exists
        admin = await user_service.get_user_by_username(username)
        if not admin:
            print(f"\n👤 Creating default admin user '{username}'...")
            admin = await user_service.create_user(
                username=username,
                email=email,
                password=settings.ADMIN_PASSWORD,
                first_name="System",
                last_name="Administrator",
                is_active=True,
                is_superuser=True,
                role_names=["admin"]
            )
            print(f"  ✓ Admin user created: {admin.username} (ID: {admin.id})")
            print(f"    Email: {admin.email}")
            print(f"    Superuser: {admin.is_superuser}")
        else:
            print(f"\n👤 Admin user already exists: {admin.username} (ID: {admin.id})")
            
            # Ensure admin has admin role
            if not admin.has_role("admin"):
                await user_service.assign_role_to_user(admin.id, "admin")
                print("  ✓ Admin role assigned")
    
    await db_manager.close()


def main():
    parser = argparse.ArgumentParser(
        description="Create default admin user",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--username",
        default="president",
        help="Admin username (default: president)"
    )
    parser.add_argument(
        "--email",
        default="admin@bi-ide.com",
        help="Admin email (default: admin@bi-ide.com)"
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("🚀 BI-IDE Default Admin Creation")
    print("=" * 60)
    
    asyncio.run(create_admin(args.username, args.email))
    
    print("\n" + "=" * 60)
    print("✅ Done!")
    print("=" * 60)


if __name__ == "__main__":
    main()
