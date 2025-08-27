"""
Health service for generation service
Minimal service that can start without ML dependencies
"""

from fastapi import FastAPI
from fastapi.responses import JSONResponse
import os
from datetime import datetime

def create_generation_health_app() -> FastAPI:
    """Create a minimal FastAPI app for generation service health"""
    app = FastAPI(
        title="VisionFlow Generation Health Service", 
        description="Health check service for video generation",
        version="1.0.0"
    )
    
    @app.get("/health")
    async def health_check():
        """Generation service health check"""
        return JSONResponse(
            status_code=200,
            content={
                "status": "healthy",
                "service": "generation-service",
                "timestamp": datetime.utcnow().isoformat(),
                "version": "1.0.0",
                "capabilities": ["health-check"]  # ML capabilities loaded separately
            }
        )
    
    @app.get("/health/ready")
    async def readiness_check():
        """Check if generation service is ready"""
        # TODO: Add ML model loading check
        return JSONResponse(
            status_code=200,
            content={
                "status": "ready",
                "service": "generation-service", 
                "timestamp": datetime.utcnow().isoformat(),
                "ml_models_loaded": False  # Will be true when ML is loaded
            }
        )
    
    @app.get("/health/live")
    async def liveness_check():
        """Generation service liveness check"""
        return JSONResponse(
            status_code=200,
            content={
                "status": "alive",
                "service": "generation-service",
                "timestamp": datetime.utcnow().isoformat()
            }
        )
    
    @app.get("/status")
    async def service_status():
        """Detailed service status"""
        return JSONResponse(
            status_code=200,
            content={
                "service": "generation-service",
                "status": "operational",
                "mode": "health-only",  # Will change to "full" when ML is loaded
                "timestamp": datetime.utcnow().isoformat(),
                "environment": os.environ.get("ENVIRONMENT", "local"),
                "device": os.environ.get("MODEL_DEVICE", "auto")
            }
        )
    
    return app

# Create the app instance
app = create_generation_health_app()