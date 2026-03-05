"""BI-IDE API Routers"""
from .auth import router as auth_router
from .council import router as council_router
from .training import router as training_router
from .hierarchy import router as hierarchy_router
from .ai import router as ai_router
from .erp import router as erp_router
from .monitoring import router as monitoring_router
from .community import router as community_router
from .admin import router as admin_router
from .rtx5090 import router as rtx5090_router
from .network import router as network_router
from .brain import router as brain_router

__all__ = [
    'auth_router',
    'council_router',
    'training_router',
    'hierarchy_router',
    'ai_router',
    'erp_router',
    'monitoring_router',
    'community_router',
    'admin_router',
    'rtx5090_router',
    'network_router',
    'brain_router',
]

