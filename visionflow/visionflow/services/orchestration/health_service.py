"""
Health service for orchestration service
Minimal service that can start without ML dependencies
"""

from fastapi import FastAPI
from fastapi.responses import JSONResponse
import os
from datetime import datetime

def create_orchestration_health_app() -> FastAPI:
    """Create a minimal FastAPI app for orchestration service health"""
    app = FastAPI(
        title="VisionFlow Orchestration Health Service",
        description="Health check service for task orchestration", 
        version="1.0.0"
    )
    
    @app.get("/health")
    async def health_check():
        """Orchestration service health check"""
        return JSONResponse(
            status_code=200,
            content={
                "status": "healthy",
                "service": "orchestration-service",
                "timestamp": datetime.utcnow().isoformat(),
                "version": "1.0.0",
                "capabilities": ["health-check", "task-routing"]
            }
        )
    
    @app.get("/health/ready") 
    async def readiness_check():
        """Check if orchestration service is ready"""
        return JSONResponse(
            status_code=200,
            content={
                "status": "ready",
                "service": "orchestration-service",
                "timestamp": datetime.utcnow().isoformat(),
                "langchain_loaded": False  # Will be true when LangChain is loaded
            }
        )
    
    @app.get("/health/live")
    async def liveness_check():
        """Orchestration service liveness check"""
        return JSONResponse(
            status_code=200, 
            content={
                "status": "alive",
                "service": "orchestration-service",
                "timestamp": datetime.utcnow().isoformat()
            }
        )
    
    @app.get("/status")
    async def service_status():
        """Detailed orchestration service status"""
        return JSONResponse(
            status_code=200,
            content={
                "service": "orchestration-service",
                "status": "operational", 
                "mode": "health-only",  # Will change to "full" when LangChain is loaded
                "timestamp": datetime.utcnow().isoformat(),
                "environment": os.environ.get("ENVIRONMENT", "local"),
                "vertex_ai_project": os.environ.get("VERTEX_AI_PROJECT", "not-configured")
            }
        )
    
    return app

# Create the app instance  
app = create_orchestration_health_app()