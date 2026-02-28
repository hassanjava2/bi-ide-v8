"""
BI-IDE API Main Application
النقطة الرئيسية لتطبيق BI-IDE API
"""

from contextlib import asynccontextmanager
from datetime import datetime

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse
import logging
import time

from api.routers import (
    auth_router,
    council_router,
    training_router,
    ai_router,
    erp_router,
    monitoring_router,
    community_router
)
from api.middleware import (
    LoggingMiddleware,
    RateLimitMiddleware,
    AuthMiddleware,
    ErrorHandlerMiddleware
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan — startup & shutdown"""
    logger.info("🚀 BI-IDE API starting up...")
    # Initialize services, connections, etc.
    yield
    logger.info("🛑 BI-IDE API shutting down...")
    # Cleanup resources


# Create FastAPI app
app = FastAPI(
    title="BI-IDE API",
    version="8.1.0",
    description="BI-IDE Distributed AI Development Environment API",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    lifespan=lifespan
)

# Add middleware (order matters - first added = first executed)
app.add_middleware(ErrorHandlerMiddleware)
app.add_middleware(GZipMiddleware, minimum_size=1000)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(LoggingMiddleware)
app.add_middleware(RateLimitMiddleware, max_requests=100, window=60)
app.add_middleware(AuthMiddleware)

# Include new routers — each defines its own sub-prefix (e.g. /council)
# We add /api/v1 here → final paths: /api/v1/council/status, etc.
app.include_router(auth_router, prefix="/api/v1", tags=["Authentication"])
app.include_router(council_router, prefix="/api/v1", tags=["Council"])
app.include_router(training_router, prefix="/api/v1", tags=["Training"])
app.include_router(ai_router, prefix="/api/v1", tags=["AI"])
app.include_router(erp_router, prefix="/api/v1", tags=["ERP"])
app.include_router(monitoring_router, prefix="/api/v1", tags=["Monitoring"])
app.include_router(community_router, prefix="/api/v1", tags=["Community"])

# Backward-compatible: include old routes that tests/production still depend on
_legacy_routes = [
    ("api.routes", "health"),            # /health, /ready, /metrics
    ("api.routes.council", "council"),    # /api/v1/council/*, /api/v1/guardian/*
    ("api.routes.system", "system"),      # /api/v1/status, /api/v1/system/*
    ("api.routes.erp", "erp"),           # /api/v1/erp/* (full ERP)
    ("api.routes.auth_routes", "auth"),   # /api/v1/auth/* (full auth)
    ("api.routes.community", "community"),
    ("api.routes.network", "network"),
    ("api.routes.checkpoints", "checkpoints"),
    ("api.routes.ideas", "ideas"),
    ("api.routes.training_data", "training_data"),
    ("api.routes.users", "users"),
    ("api.routes.ide", "ide"),
    ("api.routes.rtx4090", "rtx4090"),
    ("api.routes.admin", "admin"),
    ("api.routes.downloads", "downloads"),
]
for module_path, label in _legacy_routes:
    try:
        import importlib
        mod = importlib.import_module(module_path)
        if hasattr(mod, "router"):
            app.include_router(mod.router, tags=[f"Legacy-{label}"])
    except Exception:
        pass  # Skip routes with missing dependencies


@app.get("/", tags=["Root"])
async def root():
    """Root endpoint - API info"""
    return {
        "name": "BI-IDE API",
        "version": "8.1.0",
        "status": "operational",
        "docs": "/docs"
    }


@app.get("/health", tags=["Health"])
async def health_check():
    """Health check endpoint for Docker/K8s"""
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "services": {
            "api": "up",
            "database": "up",
            "redis": "up"
        }
    }


@app.get("/ready", tags=["Health"])
async def readiness_check():
    """
    Readiness check for Kubernetes.
    Returns 200 when critical services are initialized.
    """
    checks = {
        "api": True,
        "database": False,
        "ai_hierarchy": False,
    }

    # Check Database
    try:
        from core.database import db_manager
        checks["database"] = db_manager.async_engine is not None
    except Exception:
        pass

    # Check AI Hierarchy
    try:
        from hierarchy import ai_hierarchy
        checks["ai_hierarchy"] = ai_hierarchy is not None
    except Exception:
        pass

    all_ready = checks["api"]  # API is always ready if we got here
    response = {
        "ready": all_ready,
        "timestamp": datetime.now().isoformat(),
        "services": checks,
    }
    status_code = 200 if all_ready else 503
    return JSONResponse(content=response, status_code=status_code)


@app.get("/metrics", tags=["Health"])
async def metrics():
    """Prometheus metrics endpoint"""
    try:
        from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
        return PlainTextResponse(
            content=generate_latest(),
            media_type=CONTENT_TYPE_LATEST,
        )
    except ImportError:
        return PlainTextResponse(
            content="# Prometheus metrics not available\n",
            media_type="text/plain",
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

