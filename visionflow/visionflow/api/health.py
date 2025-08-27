"""
Health check endpoints for VisionFlow services
Simple endpoints without ML dependencies
"""

from fastapi import FastAPI
from fastapi.responses import JSONResponse
import os
from datetime import datetime

def create_health_app() -> FastAPI:
    """Create a minimal FastAPI app with just health endpoints"""
    app = FastAPI(
        title="VisionFlow Health Service",
        description="Lightweight health check service",
        version="1.0.0"
    )
    
    @app.get("/health")
    async def health_check():
        """Basic health check endpoint"""
        return JSONResponse(
            status_code=200,
            content={
                "status": "healthy",
                "service": os.environ.get("SERVICE_NAME", "visionflow"),
                "timestamp": datetime.utcnow().isoformat(),
                "version": "1.0.0"
            }
        )
    
    @app.get("/health/ready")
    async def readiness_check():
        """Readiness check - can be extended with dependency checks"""
        # In future, check database connectivity, redis, etc.
        return JSONResponse(
            status_code=200,
            content={
                "status": "ready",
                "service": os.environ.get("SERVICE_NAME", "visionflow"),
                "timestamp": datetime.utcnow().isoformat()
            }
        )
    
    @app.get("/health/live")
    async def liveness_check():
        """Liveness check - basic service alive check"""
        return JSONResponse(
            status_code=200,
            content={
                "status": "alive",
                "service": os.environ.get("SERVICE_NAME", "visionflow"),
                "timestamp": datetime.utcnow().isoformat()
            }
        )
    
    return app

# Create the app instance
app = create_health_app()