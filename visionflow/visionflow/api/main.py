"""
VisionFlow API Main Application
Production FastAPI application for video evaluation services.
"""

import os
import asyncio
from contextlib import asynccontextmanager
from typing import Dict, Any

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from prometheus_fastapi_instrumentator import Instrumentator

from ..shared.monitoring import (
    get_logger, 
    track_video_generation,
    update_job_queue_length,
    update_job_progress,
    update_success_rate,
    VIDEO_GENERATION_JOBS_ACTIVE
)
from ..shared.models import VideoGenerationRequest, VideoGenerationResult
from ..services.evaluation.video_evaluation_orchestrator import VideoEvaluationOrchestrator, SamplingStrategy
from ..services.generation.video_metadata_tracker import metadata_tracker
from ..services.generation.wan_video_service import WanVideoGenerationService

logger = get_logger(__name__)

# Global evaluation orchestrator
orchestrator: VideoEvaluationOrchestrator = None
# Global video generation service
video_generation_service: WanVideoGenerationService = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management"""
    global orchestrator, video_generation_service
    
    logger.info("🚀 Starting VisionFlow API...")
    
    # Initialize evaluation orchestrator
    orchestrator = VideoEvaluationOrchestrator(
        sampling_strategy=SamplingStrategy.ADAPTIVE,
        max_frames_per_video=int(os.getenv('MAX_FRAMES_PER_VIDEO', '20'))
    )
    
    # Initialize video generation service
    video_generation_service = WanVideoGenerationService()
    
    # Initialize metrics with default values to ensure they appear in /metrics
    # Initialize for all quality levels
    for quality in ["low", "medium", "high", "ultra"]:
        VIDEO_GENERATION_JOBS_ACTIVE.labels(status="processing", quality=quality).set(0)
        VIDEO_GENERATION_JOBS_ACTIVE.labels(status="completed", quality=quality).set(0)
        VIDEO_GENERATION_JOBS_ACTIVE.labels(status="failed", quality=quality).set(0)
        update_success_rate(quality, 1.0)
    
    update_job_queue_length(0)
    
    logger.info("✅ VisionFlow API ready with metrics initialized")
    
    yield
    
    logger.info("🔐 Shutting down VisionFlow API...")

# Create FastAPI application
app = FastAPI(
    title="VisionFlow Video Evaluation & Generation API",
    description="Production API for automated video quality assessment and WAN 2.1 video generation",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add Prometheus metrics
if os.getenv('ENABLE_PROMETHEUS', 'false').lower() == 'true':
    instrumentator = Instrumentator()
    instrumentator.instrument(app).expose(app)

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "VisionFlow Video Evaluation & Generation API",
        "version": "1.0.0",
        "status": "running",
        "capabilities": {
            "video_evaluation": "available",
            "video_generation": "available",
            "metadata_tracking": "available"
        },
        "endpoints": {
            "evaluation": "/evaluate/video",
            "generation": "/generate/video",
            "metadata": "/metadata/video",
            "analytics": "/analytics/summary",
            "health": "/health"
        }
    }

@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint"""
    from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
    from fastapi.responses import Response
    # Generate metrics from default registry (includes our custom metrics)
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": str(asyncio.get_event_loop().time()),
        "services": {
            "evaluation": "ready" if orchestrator else "not_ready",
            "video_generation": "ready" if video_generation_service else "not_ready",
            "metadata": "ready"
        }
    }

@app.post("/evaluate/video")
async def evaluate_video(
    video_path: str,
    prompt: str,
    background_tasks: BackgroundTasks
):
    """
    Evaluate a video with a text prompt
    
    Args:
        video_path: Path to the video file
        prompt: Text prompt used to generate the video
    
    Returns:
        Evaluation results with scores and confidence levels
    """
    if not orchestrator:
        raise HTTPException(status_code=503, detail="Evaluation service not ready")
    
    try:
        # Run evaluation
        result = await orchestrator.evaluate_video(video_path, prompt)
        
        # Return structured results
        return {
            "evaluation_id": result.evaluation_id,
            "overall_score": result.overall_score,
            "overall_confidence": result.overall_confidence,
            "confidence_level": result.confidence_level.value,
            "decision": result.decision,
            "requires_review": result.requires_human_review,
            "dimension_scores": [
                {
                    "dimension": dim.dimension.value,
                    "score": dim.score,
                    "confidence": dim.confidence
                }
                for dim in result.dimension_scores
            ],
            "processing_time": result.processing_time,
            "frames_evaluated": result.frames_evaluated
        }
        
    except Exception as e:
        logger.error(f"❌ Video evaluation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/metadata/video")
async def get_video_metadata(video_path: str):
    """Get metadata for a video file"""
    try:
        metadata = await metadata_tracker.get_video_metadata(video_path)
        
        if not metadata:
            raise HTTPException(status_code=404, detail="Video metadata not found")
        
        return {
            "generation_id": metadata.generation_id,
            "filename": metadata.filename,
            "prompt": metadata.prompt,
            "quality": metadata.quality,
            "duration": metadata.duration,
            "created_at": metadata.created_at.isoformat(),
            "model_name": metadata.model_name,
            "evaluation_score": metadata.overall_score,
            "confidence_level": metadata.confidence_level
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Failed to get metadata: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/analytics/summary")
async def get_analytics_summary():
    """Get analytics summary"""
    try:
        # Get video discovery info
        discovered = await metadata_tracker.discover_existing_videos()
        
        with_metadata = len([v for v in discovered if v['has_metadata']])
        total_videos = len(discovered)
        
        return {
            "total_videos": total_videos,
            "videos_with_metadata": with_metadata,
            "metadata_coverage": (with_metadata / total_videos * 100) if total_videos > 0 else 0,
            "storage_backends": len(metadata_tracker.storage_backends),
            "system_status": "operational"
        }
        
    except Exception as e:
        logger.error(f"❌ Failed to get analytics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Video Generation Endpoints
@app.post("/generate/video")
async def generate_video(request: VideoGenerationRequest):
    """
    Generate video using WAN 2.1 models
    
    Args:
        request: Video generation request with prompt and parameters
    
    Returns:
        Video generation result with video path and metadata
    """
    if not video_generation_service:
        raise HTTPException(status_code=503, detail="Video generation service not ready")
    
    try:
        logger.info(f"🎬 Starting video generation: '{request.prompt[:50]}...'")
        
        # Update metrics - job started
        VIDEO_GENERATION_JOBS_ACTIVE.labels(status="processing", quality=request.quality.value).inc()
        update_job_queue_length(1)  # Simple demo - in production, track actual queue
        
        # Generate video using the WAN service
        result = await video_generation_service.generate_video(request)
        
        if result.get("status") == "failed":
            # Update metrics - job failed
            VIDEO_GENERATION_JOBS_ACTIVE.labels(status="processing", quality=request.quality.value).dec()
            VIDEO_GENERATION_JOBS_ACTIVE.labels(status="failed", quality=request.quality.value).inc()
            update_success_rate(request.quality.value, 0.0)
            raise HTTPException(status_code=500, detail=result.get("error", "Video generation failed"))
        
        # Update metrics - job succeeded
        VIDEO_GENERATION_JOBS_ACTIVE.labels(status="processing", quality=request.quality.value).dec()
        VIDEO_GENERATION_JOBS_ACTIVE.labels(status="completed", quality=request.quality.value).inc()
        update_job_queue_length(0)
        update_success_rate(request.quality.value, 1.0)
        
        # Return structured result
        return VideoGenerationResult(
            video_path=result.get("video_path", ""),
            metadata={
                "prompt": request.prompt,
                "duration": request.duration,
                "quality": request.quality.value,
                "fps": request.fps,
                "resolution": request.resolution,
                "seed": request.seed,
                "guidance_scale": request.guidance_scale,
                "num_inference_steps": request.num_inference_steps,
                "generation_time": result.get("generation_time", 0),
                "memory_usage": result.get("memory_usage", {})
            },
            quality_metrics=result.get("quality_metrics", {}),
            generation_time=result.get("generation_time", 0),
            model_version=result.get("model_version", "wan2.1")
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Video generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/generate/video/status/{generation_id}")
async def get_generation_status(generation_id: str):
    """Get status of a video generation job"""
    try:
        # This would typically query a job queue or database
        # For now, return a placeholder response
        return {
            "generation_id": generation_id,
            "status": "completed",  # This should be dynamic
            "progress": 100,
            "estimated_completion": None,
            "created_at": "2024-01-01T00:00:00Z"  # This should be dynamic
        }
    except Exception as e:
        logger.error(f"❌ Failed to get generation status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/generate/video/history")
async def get_generation_history(limit: int = 10, offset: int = 0):
    """Get history of video generation jobs"""
    try:
        # This would typically query a database
        # For now, return a placeholder response
        return {
            "generations": [],
            "total_count": 0,
            "limit": limit,
            "offset": offset
        }
    except Exception as e:
        logger.error(f"❌ Failed to get generation history: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/generate/video/{generation_id}")
async def cancel_generation(generation_id: str):
    """Cancel a video generation job"""
    try:
        # This would typically cancel a running job
        # For now, return a placeholder response
        return {
            "generation_id": generation_id,
            "status": "cancelled",
            "message": "Generation job cancelled successfully"
        }
    except Exception as e:
        logger.error(f"❌ Failed to cancel generation: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)