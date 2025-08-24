"""
Standalone video generation service
Can run independently or as part of the main application
"""

import asyncio
import time
import uuid
from datetime import datetime
from typing import Dict, Any, Optional

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

from .wan_model_service import enhanced_generation_service
from ...shared.config import get_settings
from ...shared.models import VideoGenerationRequest, GenerationResult
from ...shared.monitoring import get_logger, setup_logging

# Setup logging
setup_logging()
logger = get_logger("generation_service")

# Get settings
settings = get_settings()

# Create FastAPI app
app = FastAPI(
    title="VisionFlow Generation Service",
    description="GPU-accelerated video generation with WAN 2.1",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Service startup time
startup_time = time.time()


class GenerationServiceRequest(BaseModel):
    """Request model for generation service"""
    prompt: str = Field(..., description="Optimized prompt for generation")
    original_prompt: str = Field(..., description="Original user prompt")
    generation_params: Dict[str, Any] = Field(..., description="Generation parameters")
    routing_decision: Dict[str, Any] = Field(..., description="Routing decision")
    job_id: str = Field(..., description="Job identifier")


class GenerationServiceResponse(BaseModel):
    """Response model for generation service"""
    job_id: str
    status: str
    video_path: Optional[str] = None
    generation_time: float
    quality_metrics: Dict[str, Any]
    error: Optional[str] = None


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    timestamp: datetime
    uptime: float
    gpu_available: bool
    model_loaded: bool
    memory_usage: Dict[str, Any]


@app.on_event("startup")
async def startup_event():
    """Application startup event."""
    logger.info("Starting VisionFlow Generation Service")
    
    # Initialize memory manager and check GPU
    memory_info = enhanced_generation_service.memory_manager.check_memory()
    logger.info("GPU/Memory status", **memory_info)
    
    # Optionally preload model
    try:
        # This will load the model into cache
        logger.info("Preloading WAN 2.1 model...")
        await enhanced_generation_service.model_loader.load_pipeline(
            settings.model.wan_model_path,
            "wan2-1-fast"
        )
        logger.info("Model preloaded successfully")
    except Exception as e:
        logger.warning(f"Could not preload model: {e}")
    
    logger.info("VisionFlow Generation Service started successfully")


@app.on_event("shutdown")
async def shutdown_event():
    """Application shutdown event."""
    logger.info("Shutting down VisionFlow Generation Service")
    
    # Cleanup GPU memory
    enhanced_generation_service.memory_manager.cleanup_memory()


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint with GPU status."""
    
    # Check memory status
    memory_info = enhanced_generation_service.memory_manager.check_memory()
    
    # Check if model is loaded
    model_loaded = len(enhanced_generation_service.model_loader._pipelines) > 0
    
    # Check GPU availability
    gpu_available = "cuda" in enhanced_generation_service.memory_manager.device
    
    status = "healthy"
    if gpu_available and memory_info.get("gpu_utilization", 0) > 0.95:
        status = "degraded"
    
    return HealthResponse(
        status=status,
        timestamp=datetime.utcnow(),
        uptime=time.time() - startup_time,
        gpu_available=gpu_available,
        model_loaded=model_loaded,
        memory_usage=memory_info
    )


@app.post("/generate", response_model=GenerationServiceResponse)
async def generate_video(request: GenerationServiceRequest):
    """Generate video using WAN 2.1 model."""
    
    try:
        logger.info(f"Starting video generation for job {request.job_id}")
        
        # Create VideoGenerationRequest from parameters
        video_request = VideoGenerationRequest(**request.generation_params)
        
        # Create mock prompt optimization (this would normally come from prompt service)
        from ...shared.models import PromptOptimization
        prompt_optimization = PromptOptimization(
            original_prompt=request.original_prompt,
            optimized_prompt=request.prompt,
            optimization_strategy="external",
            quality_score=0.8,
            modifications=[]
        )
        
        # Create mock routing decision (this would normally come from router service)
        from ...shared.models import RoutingDecision
        routing_decision = RoutingDecision(**request.routing_decision)
        
        # Generate video
        result = await enhanced_generation_service.generate_video(
            video_request,
            prompt_optimization,
            routing_decision
        )
        
        logger.info(f"Video generation completed for job {request.job_id}")
        
        return GenerationServiceResponse(
            job_id=request.job_id,
            status="completed",
            video_path=result.video_path,
            generation_time=result.generation_time,
            quality_metrics=result.quality_metrics
        )
        
    except Exception as e:
        logger.error(f"Video generation failed for job {request.job_id}: {e}")
        
        return GenerationServiceResponse(
            job_id=request.job_id,
            status="failed",
            video_path=None,
            generation_time=0.0,
            quality_metrics={},
            error=str(e)
        )


@app.get("/models/status")
async def get_model_status():
    """Get status of loaded models."""
    
    loader = enhanced_generation_service.model_loader
    memory_info = enhanced_generation_service.memory_manager.check_memory()
    
    return {
        "loaded_models": list(loader._pipelines.keys()),
        "model_count": len(loader._pipelines),
        "max_cached_models": loader.max_cached_models,
        "memory_usage": memory_info,
        "device": enhanced_generation_service.memory_manager.device
    }


@app.post("/models/cleanup")
async def cleanup_models():
    """Force cleanup of model cache and GPU memory."""
    
    try:
        # Clear model cache
        enhanced_generation_service.model_loader._pipelines.clear()
        enhanced_generation_service.model_loader._last_access.clear()
        
        # Cleanup GPU memory
        enhanced_generation_service.memory_manager.cleanup_memory()
        
        memory_info = enhanced_generation_service.memory_manager.check_memory()
        
        logger.info("Model cache and GPU memory cleaned up")
        
        return {
            "status": "success",
            "message": "Models and memory cleaned up",
            "memory_usage": memory_info
        }
        
    except Exception as e:
        logger.error(f"Cleanup failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Cleanup failed: {e}"
        )


@app.get("/metrics")
async def get_metrics():
    """Get service metrics."""
    
    memory_info = enhanced_generation_service.memory_manager.check_memory()
    
    return {
        "service": "visionflow-generation",
        "uptime": time.time() - startup_time,
        "cache_size": len(enhanced_generation_service._generation_cache),
        "loaded_models": len(enhanced_generation_service.model_loader._pipelines),
        "memory_usage": memory_info,
        "device": enhanced_generation_service.memory_manager.device
    }


if __name__ == "__main__":
    uvicorn.run(
        "visionflow.services.generation.standalone_service:app",
        host="0.0.0.0",
        port=8002,
        reload=False,
        log_level="info"
    )
