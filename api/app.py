"""
BI IDE v8 - App Factory
نقطة الدخول الجديدة — تُنشئ تطبيق FastAPI مع كل الـ routers
"""

import sys
import os
import asyncio
import threading

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import datetime
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from api.middleware import ErrorHandlingMiddleware
from api.rate_limit_redis import RedisRateLimitMiddleware


def create_app() -> FastAPI:
    """Application factory — creates and configures the FastAPI app."""

    # ── Lifecycle events ──

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if os.getenv("PYTEST_RUNNING") == "1":
            # Keep tests deterministic: avoid background startup tasks that can race
            # with test-controlled database initialization and cause SQLite locks.
            yield
            return

        def _spawn_step(label: str, step_factory, timeout_sec: float, *, run_in_thread: bool = False):
            """Schedule a startup step without blocking request handling.

            If run_in_thread=True, runs the coroutine in a dedicated daemon thread
            with its own event loop. This avoids blocking the main asyncio loop if
            the step contains accidental blocking calls (time.sleep, requests, etc).
            """

            if run_in_thread:
                def _thread_runner():
                    try:
                        asyncio.run(asyncio.wait_for(step_factory(), timeout=timeout_sec))
                        print(f"✅ {label}")
                    except asyncio.TimeoutError:
                        print(f"⚠️ {label}: timed out after {timeout_sec}s")
                    except Exception as e:
                        print(f"⚠️ {label}: {e}")

                t = threading.Thread(target=_thread_runner, daemon=True)
                t.start()
                return t

            async def _runner():
                await _run_step(label, step_factory(), timeout_sec)

            return asyncio.create_task(_runner())

        def _spawn_sync_step(label: str, func, timeout_sec: float):
            async def _coro():
                await asyncio.to_thread(func)

            return _spawn_step(label, lambda: _coro(), timeout_sec)

        async def _run_step(label: str, coro, timeout_sec: float):
            try:
                await asyncio.wait_for(coro, timeout=timeout_sec)
                print(f"✅ {label}")
                return True
            except asyncio.TimeoutError:
                print(f"⚠️ {label}: timed out after {timeout_sec}s")
                return False
            except Exception as e:
                print(f"⚠️ {label}: {e}")
                return False

        print("=" * 60)
        print("🚀 Starting BI IDE v8 — Unified API")
        print("=" * 60)

        # Core modules
        try:
            from core.database import db_manager
            from core.cache import cache_manager

            # Do not block server startup on DB/Redis. Initialize in background.
            _spawn_step(
                "Database initialized",
                lambda: db_manager.initialize(),
                float(os.getenv("STARTUP_DB_TIMEOUT", "60")),
            )

            _spawn_step(
                "Cache initialized",
                lambda: cache_manager.initialize(),
                float(os.getenv("STARTUP_CACHE_TIMEOUT", "20")),
            )
        except Exception as e:
            print(f"⚠️ Core modules init: {e}")

        # Initialize default admin user
        try:
            from scripts.create_default_admin import create_admin
            _spawn_step(
                "Default admin initialized",
                lambda: create_admin(),
                float(os.getenv("STARTUP_ADMIN_TIMEOUT", "30")),
            )
        except Exception as e:
            print(f"⚠️ Admin init: {e}")

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
                    # IMPORTANT: hierarchy initialization may contain blocking calls; keep it off main loop.
                    _spawn_step(
                        "AI Hierarchy initialized (15 layers)",
                        lambda: hierarchy.initialize(),
                        float(os.getenv("STARTUP_HIERARCHY_TIMEOUT", "180")),
                        run_in_thread=True,
                    )
            except Exception as e:
                print(f"⚠️ AI Hierarchy import: {e}")

        # IDE Service
        try:
            def _init_ide():
                from ide.ide_service import get_ide_service
                from api.routes.ide import set_ide_service
                ide_service = get_ide_service(hierarchy)
                set_ide_service(ide_service)

            _spawn_sync_step(
                "IDE Service ready",
                _init_ide,
                float(os.getenv("STARTUP_IDE_TIMEOUT", "60")),
            )
        except Exception as e:
            print(f"⚠️ IDE Service schedule: {e}")

        # ERP Database Service (PostgreSQL/SQLite) - Full Integration
        try:
            from erp.erp_database_service import get_erp_db_service
            from api.routes.erp import set_erp_db_service

            async def _init_erp_db():
                erp_db = get_erp_db_service(hierarchy)
                await erp_db.initialize()
                set_erp_db_service(erp_db)

            _spawn_step(
                "ERP Database Service ready (PostgreSQL/SQLite)",
                lambda: _init_erp_db(),
                float(os.getenv("STARTUP_ERP_DB_TIMEOUT", "60")),
            )
        except Exception as e:
            print(f"⚠️ ERP DB Service schedule: {e}")

        # Specialized Network
        try:
            def _init_network():
                from hierarchy.specialized_ai_network import get_specialized_network_service
                from api.routes.network import set_network_service
                network_service = get_specialized_network_service()
                set_network_service(network_service)

            _spawn_sync_step(
                "Specialized AI Network ready",
                _init_network,
                float(os.getenv("STARTUP_NETWORK_TIMEOUT", "60")),
            )
        except Exception as e:
            print(f"⚠️ Network Service schedule: {e}")

        # Council
        try:
            def _init_council():
                from api.routes.council import init_council
                init_council()

            _spawn_sync_step(
                "Council initialized",
                _init_council,
                float(os.getenv("STARTUP_COUNCIL_TIMEOUT", "60")),
            )
        except Exception as e:
            print(f"⚠️ Council schedule: {e}")

        # Ideas
        try:
            def _init_ideas():
                from api.routes.ideas import init_ideas
                init_ideas()

            _spawn_sync_step(
                "Ideas initialized",
                _init_ideas,
                float(os.getenv("STARTUP_IDEAS_TIMEOUT", "60")),
            )
        except Exception as e:
            print(f"⚠️ Ideas schedule: {e}")

        # Checkpoint sync
        try:
            from api.routes.checkpoints import start_checkpoint_sync
            # Can involve slow network I/O; run in background so API becomes responsive immediately.
            _spawn_step(
                "Checkpoint sync started",
                lambda: start_checkpoint_sync(),
                float(os.getenv("STARTUP_CHECKPOINT_SYNC_TIMEOUT", "120")),
                run_in_thread=True,
            )
        except Exception as e:
            print(f"⚠️ Checkpoint sync: {e}")

        # Sync Manager (auto-sync to RTX 5090 + daily backup)
        try:
            from core.sync_manager import sync_manager
            _spawn_step(
                "Sync Manager started",
                lambda: sync_manager.start(),
                float(os.getenv("STARTUP_SYNC_TIMEOUT", "10")),
                run_in_thread=True,
            )
        except Exception as e:
            print(f"⚠️ Sync Manager: {e}")

        # Training Coordinator (auto-generates training tasks)
        # NOTE: The coordinator runs an INFINITE loop, so it must NOT use
        # _spawn_step (which has a timeout). Instead, run as a persistent
        # daemon thread.
        try:
            from core.training_coordinator import training_coordinator
            from orchestrator_api import state as orchestrator_state

            def _coordinator_thread():
                asyncio.run(training_coordinator.start(orchestrator_state))

            coord_t = threading.Thread(target=_coordinator_thread, daemon=True, name="training-coordinator")
            coord_t.start()
            print("✅ Training Coordinator started (persistent daemon)")
        except Exception as e:
            print(f"⚠️ Training Coordinator: {e}")

        print("=" * 60)
        print("✅ Startup tasks scheduled (API is ready to accept requests)")
        print("=" * 60)

        yield

        print("\n🔄 Shutting down...")

        try:
            from api.routes.checkpoints import stop_checkpoint_sync
            await stop_checkpoint_sync()
        except Exception:
            pass

        try:
            from core.sync_manager import sync_manager
            sync_manager.stop()
        except Exception:
            pass

        try:
            from core.training_coordinator import training_coordinator
            training_coordinator.stop()
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
    # Use Redis-backed rate limiter for multi-instance support
    redis_url = os.getenv("REDIS_URL")
    if redis_url:
        app.add_middleware(RedisRateLimitMiddleware)
    else:
        from api.rate_limit import RateLimitMiddleware
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
    from api.routes.users import router as users_router
    from api.routes.council import router as council_router
    from api.routes.ide import router as ide_router
    from api.routes.erp import router as erp_router
    from api.routes.system import router as system_router
    from api.routes.network import router as network_router
    from api.routes.checkpoints import router as checkpoint_router
    from api.routes.ideas import router as ideas_router
    from api.routes.admin import router as admin_router
    from api.routes.rtx4090 import router as rtx4090_router
    from api.routes.community import router as community_router
    from api.routes.downloads import router as downloads_router
    from api.routes.training_data import router as training_data_router
    try:
        from mobile.api.mobile_routes import router as mobile_router
    except ImportError:
        mobile_router = None

    app.include_router(health_router)
    app.include_router(auth_router)
    app.include_router(users_router)
    app.include_router(council_router)
    app.include_router(ide_router)
    app.include_router(erp_router)
    app.include_router(system_router)
    app.include_router(network_router)
    app.include_router(checkpoint_router)
    app.include_router(ideas_router)
    app.include_router(admin_router)
    app.include_router(rtx4090_router)
    app.include_router(training_data_router)
    app.include_router(community_router, prefix="/api/v1")
    app.include_router(downloads_router)
    if mobile_router:
        app.include_router(mobile_router, prefix="/api/v1")

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

        # Define excluded paths for SPA catch-all
        # This prevents the catch-all from interfering with API and system routes
        EXCLUDED_PATHS = {
            "api", "docs", "redoc", "openapi.json",
            "health", "ready", "metrics", "static",
            "admin", "auth", "test", "debug", "api/v1"
        }
        
        @app.get("/{path:path}")
        async def serve_spa(path: str):
            """
            Serve SPA for all non-API routes.
            
            SECURITY NOTE: EXCLUDED_PATHS must be updated if new API routes
            are added that don't start with 'api/'.
            """
            # Check if path starts with any excluded prefix
            if any(path.startswith(p) for p in EXCLUDED_PATHS):
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
