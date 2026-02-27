#!/usr/bin/env python3
"""
Initialize Default Roles and Admin User
تهيئة الأدوار الافتراضية والمستخدم المدير

Usage:
    python scripts/init_roles.py
    python scripts/init_roles.py --admin-username admin --admin-email admin@example.com --admin-password secret123
"""

import argparse
import asyncio
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqlalchemy import select
from core.database import db_manager
from core.user_service import UserService, RoleService, DEFAULT_ROLES
from core.user_models import UserDB, RoleDB


async def init_roles():
    """Initialize default roles"""
    async with db_manager.get_session() as db:
        role_service = RoleService(db)
        
        print("🔧 Initializing default roles...")
        
        created_count = 0
        for role_name, role_data in DEFAULT_ROLES.items():
            # Check if role already exists
            result = await db.execute(
                select(RoleDB).where(RoleDB.name == role_name)
            )
            existing = result.scalar_one_or_none()
            
            if existing:
                print(f"  ✓ Role '{role_name}' already exists")
                continue
            
            # Create role
            role = await role_service.create_role(
                name=role_name,
                description=role_data["description"],
                permissions=role_data["permissions"]
            )
            print(f"  ✓ Created role '{role_name}' with {len(role_data['permissions'])} permissions")
            created_count += 1
        
        print(f"\n✅ Roles initialization complete. Created {created_count} new roles.")
        return True


async def init_admin_user(username: str, email: str, password: str, first_name: str = "Admin", last_name: str = "User"):
    """Initialize admin user if not exists"""
    async with db_manager.get_session() as db:
        user_service = UserService(db)
        
        print(f"\n👤 Checking admin user '{username}'...")
        
        # Check if user exists by username
        existing_user = await user_service.get_user_by_username(username)
        if existing_user:
            print(f"  ✓ Admin user '{username}' already exists (ID: {existing_user.id})")
            return existing_user
        
        # Check if user exists by email
        existing_email = await user_service.get_user_by_email(email)
        if existing_email:
            print(f"  ✓ Admin email '{email}' already exists (ID: {existing_email.id})")
            return existing_email
        
        # Create admin user
        user = await user_service.create_user(
            username=username,
            email=email,
            password=password,
            first_name=first_name,
            last_name=last_name,
            is_active=True,
            is_superuser=True,
            role_names=["admin"]
        )
        
        print(f"  ✓ Created admin user '{username}' (ID: {user.id})")
        print(f"    Email: {email}")
        print(f"    Superuser: Yes")
        print(f"    Roles: admin")
        
        return user


async def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Initialize default roles and admin user for BI-IDE",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Initialize with defaults (from .env or environment variables)
  python scripts/init_roles.py
  
  # Initialize with custom admin credentials
  python scripts/init_roles.py --admin-username myadmin --admin-email admin@company.com --admin-password MySecurePass123
  
  # Initialize roles only (skip admin user creation)
  python scripts/init_roles.py --skip-admin
  
  # Initialize admin only (skip roles creation)
  python scripts/init_roles.py --skip-roles
        """
    )
    
    parser.add_argument(
        "--admin-username",
        default=os.getenv("ADMIN_USERNAME", "admin"),
        help="Admin username (default: admin or ADMIN_USERNAME env var)"
    )
    parser.add_argument(
        "--admin-email",
        default=os.getenv("ADMIN_EMAIL", "admin@bi-ide.local"),
        help="Admin email (default: admin@bi-ide.local or ADMIN_EMAIL env var)"
    )
    parser.add_argument(
        "--admin-password",
        default=os.getenv("ADMIN_PASSWORD", "admin123"),
        help="Admin password (default: from ADMIN_PASSWORD env var or 'admin123')"
    )
    parser.add_argument(
        "--admin-first-name",
        default="System",
        help="Admin first name (default: System)"
    )
    parser.add_argument(
        "--admin-last-name",
        default="Administrator",
        help="Admin last name (default: Administrator)"
    )
    parser.add_argument(
        "--skip-admin",
        action="store_true",
        help="Skip admin user creation"
    )
    parser.add_argument(
        "--skip-roles",
        action="store_true",
        help="Skip roles initialization"
    )
    parser.add_argument(
        "--list-roles",
        action="store_true",
        help="List all default roles and exit"
    )
    
    args = parser.parse_args()
    
    # List roles only
    if args.list_roles:
        print("📋 Default Roles:\n")
        for role_name, role_data in DEFAULT_ROLES.items():
            print(f"  • {role_name}")
            print(f"    Description: {role_data['description']}")
            print(f"    Permissions: {len(role_data['permissions'])}")
            for perm in role_data['permissions']:
                print(f"      - {perm}")
            print()
        return
    
    print("=" * 60)
    print("🚀 BI-IDE User Management Initialization")
    print("=" * 60)
    
    # Initialize database connection
    print("\n📡 Connecting to database...")
    await db_manager.initialize()
    print(f"   Database: {db_manager.database_url}")
    
    success = True
    
    try:
        # Initialize roles
        if not args.skip_roles:
            success = await init_roles() and success
        else:
            print("\n⏭️  Skipping roles initialization")
        
        # Initialize admin user
        if not args.skip_admin:
            if not args.admin_password or args.admin_password == "admin123":
                print("\n⚠️  Warning: Using default admin password. Please change it in production!")
            
            await init_admin_user(
                username=args.admin_username,
                email=args.admin_email,
                password=args.admin_password,
                first_name=args.admin_first_name,
                last_name=args.admin_last_name
            )
        else:
            print("\n⏭️  Skipping admin user creation")
        
        print("\n" + "=" * 60)
        print("✅ Initialization complete!")
        print("=" * 60)
        
        if not args.skip_admin:
            print(f"\n🔑 Admin Credentials:")
            print(f"   Username: {args.admin_username}")
            print(f"   Email: {args.admin_email}")
            print(f"   Password: {'*' * len(args.admin_password)}")
        
    except Exception as e:
        print(f"\n❌ Error during initialization: {e}")
        success = False
    finally:
        await db_manager.close()
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    asyncio.run(main())
