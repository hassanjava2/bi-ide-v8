"""
BI IDE v8 - App Factory
نقطة الدخول الجديدة — تُنشئ تطبيق FastAPI مع كل الـ routers
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import datetime
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from api.middleware import ErrorHandlingMiddleware
from api.rate_limit import RateLimitMiddleware


def create_app() -> FastAPI:
    """Application factory — creates and configures the FastAPI app."""

    # ── Lifecycle events ──

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        print("=" * 60)
        print("🚀 Starting BI IDE v8 — Unified API")
        print("=" * 60)

        # Core modules
        try:
            from core.database import db_manager
            from core.cache import cache_manager

            await db_manager.initialize()
            print("✅ Database initialized")

            await cache_manager.initialize()
            print("✅ Cache initialized")
        except Exception as e:
            print(f"⚠️ Core modules init: {e}")

        # AI Hierarchy
        hierarchy = None
        AI_CORE_HOST = os.getenv("AI_CORE_HOST", None)
        if AI_CORE_HOST:
            print(f"🔗 Remote AI Mode: {AI_CORE_HOST}")
        else:
            try:
                from hierarchy import ai_hierarchy
                hierarchy = ai_hierarchy
                if hierarchy:
                    await hierarchy.initialize()
                    print("🧠 AI Hierarchy initialized (15 layers)")
            except Exception as e:
                print(f"⚠️ AI Hierarchy: {e}")

        # IDE Service
        try:
            from ide.ide_service import get_ide_service
            from api.routes.ide import set_ide_service
            ide_service = get_ide_service(hierarchy)
            set_ide_service(ide_service)
            print("💻 IDE Service ready")
        except Exception as e:
            print(f"⚠️ IDE Service: {e}")

        # ERP Service (in-memory fallback)
        try:
            from erp.erp_service import get_erp_service
            from api.routes.erp import set_erp_service
            erp_service = get_erp_service(hierarchy)
            set_erp_service(erp_service)
            print("🏢 ERP Service (in-memory) ready")
        except Exception as e:
            print(f"⚠️ ERP Service: {e}")

        # ERP Database Service (PostgreSQL/SQLite)
        try:
            from erp.erp_db_service import get_erp_db_service
            from api.routes.erp import set_erp_db_service
            erp_db = get_erp_db_service(hierarchy)
            await erp_db.initialize()
            set_erp_db_service(erp_db)
            print("🏢 ERP Database Service ready (DB-backed)")
        except Exception as e:
            print(f"⚠️ ERP DB Service: {e} — using in-memory fallback")

        # Specialized Network
        try:
            from hierarchy.specialized_ai_network import get_specialized_network_service
            from api.routes.network import set_network_service
            network_service = get_specialized_network_service()
            set_network_service(network_service)
            print("🧬 Specialized AI Network ready")
        except Exception as e:
            print(f"⚠️ Network Service: {e}")

        # Council
        try:
            from api.routes.council import init_council
            init_council()
        except Exception as e:
            print(f"⚠️ Council init: {e}")

        # Ideas
        try:
            from api.routes.ideas import init_ideas
            init_ideas()
        except Exception as e:
            print(f"⚠️ Ideas init: {e}")

        # Checkpoint sync
        try:
            from api.routes.checkpoints import start_checkpoint_sync
            await start_checkpoint_sync()
        except Exception as e:
            print(f"⚠️ Checkpoint sync: {e}")

        print("=" * 60)
        print("✅ All services initialized")
        print("=" * 60)

        yield

        print("\n🔄 Shutting down...")

        try:
            from api.routes.checkpoints import stop_checkpoint_sync
            await stop_checkpoint_sync()
        except Exception:
            pass

        try:
            from core.database import db_manager
            await db_manager.close()
            print("✅ Database connection closed")
        except Exception:
            pass

        print("✅ Shutdown complete")

    app = FastAPI(
        title="BI IDE v8 - Unified AI-Powered Platform",
        description="""
        **BI IDE v8** - منصة متكاملة للتطوير وإدارة الموارد المؤسسية مدعومة بالذكاء الاصطناعي

        ## Features
        * **IDE**: بيئة تطوير متكاملة مع AI Copilot
        * **ERP**: نظام إدارة موارد مؤسسية
        * **AI Hierarchy**: نظام هرمي ذكي (10+ طبقات)
        * **Smart Council**: 16 حكيم AI للقرارات الاستراتيجية

        ## Authentication
        Use `/api/v1/auth/login` to obtain a Bearer token.

        ## Rate Limits
        * General API: 120 req/min
        * Auth endpoints: 10 req/min
        * AI/Council: 30 req/min
        """,
        version="8.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        lifespan=lifespan,
    )

    # ── Middleware (order matters: last added = first executed) ──
    app.add_middleware(ErrorHandlingMiddleware)
    app.add_middleware(RateLimitMiddleware)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=os.getenv("CORS_ORIGINS", "http://localhost:5173,http://localhost:3000").split(","),
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Import & register routers ──
    from api.routes import router as health_router
    from api.routes.auth_routes import router as auth_router
    from api.routes.council import router as council_router
    from api.routes.ide import router as ide_router
    from api.routes.erp import router as erp_router
    from api.routes.system import router as system_router
    from api.routes.network import router as network_router
    from api.routes.checkpoints import router as checkpoint_router
    from api.routes.ideas import router as ideas_router
    from api.routes.admin import router as admin_router

    app.include_router(health_router)
    app.include_router(auth_router)
    app.include_router(council_router)
    app.include_router(ide_router)
    app.include_router(erp_router)
    app.include_router(system_router)
    app.include_router(network_router)
    app.include_router(checkpoint_router)
    app.include_router(ideas_router)
    app.include_router(admin_router)

    # Include orchestrator router (existing separate module)
    try:
        from orchestrator_api import router as orchestrator_router
        app.include_router(orchestrator_router)
    except ImportError:
        print("⚠️ orchestrator_api not available")

    # ── Static files (SPA) ──
    if os.path.exists("ui/dist"):
        from fastapi.responses import FileResponse

        if os.path.exists("ui/dist/assets"):
            app.mount("/assets", StaticFiles(directory="ui/dist/assets"), name="assets")

        @app.get("/")
        async def serve_index():
            return FileResponse("ui/dist/index.html")

        @app.get("/{path:path}")
        async def serve_spa(path: str):
            if path.startswith("api/") or path in ("docs", "redoc", "openapi.json", "health", "ready", "metrics"):
                from fastapi import HTTPException
                raise HTTPException(404, "Not Found")
            return FileResponse("ui/dist/index.html")

    return app


# Create the app instance
app = create_app()


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", "8000"))
    print("=" * 50)
    print("🚀 BI IDE v8 - Unified API Starting")
    print(f"📍 URL: http://localhost:{port}")
    print("🔧 Services: IDE + ERP + AI (15 layers)")
    print("=" * 50)

    uvicorn.run(app, host="0.0.0.0", port=port)
