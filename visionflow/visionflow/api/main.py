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

from ..shared.monitoring import get_logger
from ..services.evaluation.video_evaluation_orchestrator import VideoEvaluationOrchestrator, SamplingStrategy
from ..services.generation.video_metadata_tracker import metadata_tracker

logger = get_logger(__name__)

# Global evaluation orchestrator
orchestrator: VideoEvaluationOrchestrator = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management"""
    global orchestrator
    
    logger.info("🚀 Starting VisionFlow API...")
    
    # Initialize evaluation orchestrator
    orchestrator = VideoEvaluationOrchestrator(
        sampling_strategy=SamplingStrategy.ADAPTIVE,
        max_frames_per_video=int(os.getenv('MAX_FRAMES_PER_VIDEO', '20'))
    )
    
    logger.info("✅ VisionFlow API ready")
    
    yield
    
    logger.info("🔐 Shutting down VisionFlow API...")

# Create FastAPI application
app = FastAPI(
    title="VisionFlow Video Evaluation API",
    description="Production API for automated video quality assessment",
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
        "message": "VisionFlow Video Evaluation API",
        "version": "1.0.0",
        "status": "running"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": str(asyncio.get_event_loop().time()),
        "services": {
            "evaluation": "ready" if orchestrator else "not_ready",
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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)