"""
Error Handling Middleware - معالجة الأخطاء الموحدة
"""

import time
import traceback
from datetime import datetime

from fastapi import Request, HTTPException
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware


class ErrorHandlingMiddleware(BaseHTTPMiddleware):
    """Unified error handling and request logging middleware"""

    async def dispatch(self, request: Request, call_next):
        start_time = time.perf_counter()

        try:
            response = await call_next(request)

            # Log slow requests
            duration_ms = (time.perf_counter() - start_time) * 1000
            if duration_ms > 5000:
                print(f"⚠️ Slow request: {request.method} {request.url.path} took {duration_ms:.0f}ms")

            return response

        except HTTPException:
            raise  # Let FastAPI handle HTTP exceptions

        except Exception as e:
            duration_ms = (time.perf_counter() - start_time) * 1000
            error_id = f"ERR-{int(time.time())}"

            print(f"❌ [{error_id}] Unhandled error on {request.method} {request.url.path}: {e}")
            traceback.print_exc()

            return JSONResponse(
                status_code=500,
                content={
                    "error": "Internal Server Error",
                    "error_id": error_id,
                    "message": str(e),
                    "path": str(request.url.path),
                    "timestamp": datetime.now().isoformat(),
                    "duration_ms": round(duration_ms, 2),
                },
            )
