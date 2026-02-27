# Mobile API Package
"""
Mobile-optimized API endpoints

Optimized for mobile applications with:
- Lighter payloads
- Offline support indicators
- Mobile-specific endpoints
"""

from .mobile_routes import router as mobile_router

__all__ = ['mobile_router']
