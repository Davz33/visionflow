"""Main FastAPI application for VisionFlow."""

import asyncio
import time
import uuid
from datetime import datetime
from typing import Any, Dict, Optional

from fastapi import Depends, FastAPI, HTTPException, Request, status
from pydantic import Field
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from prometheus_fastapi_instrumentator import Instrumentator

from ..shared.config import get_settings
from ..shared.database import DatabaseManager, get_db
from ..shared.models import (
    ErrorResponse,
    HealthCheck,
    JobResponse,
    JobStatusResponse,
    MetricsResponse,
    VideoGenerationRequest,
)
from ..shared.monitoring import (
    get_logger,
    set_service_info,
    setup_logging,
    start_metrics_server,
)
from .orchestrator import orchestrator
# from ..services.evaluation import get_evaluation_orchestrator
# from ..services.mlflow_integration import get_model_tracker

# Setup logging
setup_logging()
logger = get_logger("api_main")

# Get settings
settings = get_settings()

# Create FastAPI app
app = FastAPI(
    title="VisionFlow API",
    description="Enterprise video generation platform with WAN 2.1",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add Prometheus instrumentation
instrumentator = Instrumentator()
instrumentator.instrument(app).expose(app)

# Database manager
db_manager = DatabaseManager()

# Service startup time
startup_time = time.time()


@app.on_event("startup")
async def startup_event():
    """Application startup event."""
    logger.info("Starting VisionFlow API server")
    
    # Set service info for metrics
    set_service_info("visionflow-api", "0.1.0", "Video generation API")
    
    # Start metrics server
    start_metrics_server()
    
    # Create database tables
    try:
        from ..shared.database import create_tables
        create_tables()
        logger.info("Database tables created/verified")
    except Exception as e:
        logger.error("Failed to create database tables", error=str(e))
    
    logger.info("VisionFlow API server started successfully")


@app.on_event("shutdown")
async def shutdown_event():
    """Application shutdown event."""
    logger.info("Shutting down VisionFlow API server")


@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all HTTP requests."""
    start_time = time.time()
    
    # Generate request ID
    request_id = str(uuid.uuid4())
    
    logger.info(
        "Request started",
        request_id=request_id,
        method=request.method,
        url=str(request.url),
        client_ip=request.client.host if request.client else None,
    )
    
    response = await call_next(request)
    
    process_time = time.time() - start_time
    
    logger.info(
        "Request completed",
        request_id=request_id,
        method=request.method,
        url=str(request.url),
        status_code=response.status_code,
        process_time=process_time,
    )
    
    return response


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler."""
    logger.error(
        "Unhandled exception",
        error=str(exc),
        path=request.url.path,
        method=request.method,
    )
    
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=ErrorResponse(
            error="InternalServerError",
            message="An internal server error occurred",
            timestamp=datetime.utcnow(),
        ).dict(),
    )


@app.get("/health", response_model=HealthCheck)
async def health_check():
    """Health check endpoint."""
    import psutil
    
    # Check database connection
    try:
        job = db_manager.get_job(uuid.uuid4())  # This will return None but tests connection
        database_connected = True
    except Exception:
        database_connected = False
    
    # Check Redis connection
    try:
        import redis
        r = redis.Redis(host=settings.redis.host, port=settings.redis.port, db=settings.redis.db)
        r.ping()
        redis_connected = True
    except Exception:
        redis_connected = False
    
    # Check storage connection
    try:
        from minio import Minio
        client = Minio(
            settings.storage.endpoint,
            access_key=settings.storage.access_key,
            secret_key=settings.storage.secret_key,
            secure=settings.storage.secure,
        )
        client.list_buckets()
        storage_connected = True
    except Exception:
        storage_connected = False
    
    # Model loading status (simplified check)
    model_loaded = True  # Assume loaded for now
    
    # Check MLFlow health
    mlflow_connected = None
    try:
        # tracker = get_model_tracker()
        health = tracker.health_check()
        mlflow_connected = health.get("mlflow_initialized", False)
    except Exception:
        mlflow_connected = False
    
    # System metrics
    cpu_usage = psutil.cpu_percent(interval=1)
    memory = psutil.virtual_memory()
    disk = psutil.disk_usage('/')
    
    return HealthCheck(
        status="healthy" if all([database_connected, redis_connected, storage_connected]) else "degraded",
        timestamp=datetime.utcnow(),
        version="0.1.0",
        uptime=time.time() - startup_time,
        database_connected=database_connected,
        redis_connected=redis_connected,
        storage_connected=storage_connected,
        model_loaded=model_loaded,
        mlflow_connected=mlflow_connected,
        cpu_usage=cpu_usage,
        memory_usage=memory.percent,
        disk_usage=disk.percent,
    )


@app.get("/metrics", response_model=MetricsResponse)
async def get_metrics():
    """Get service metrics."""
    # This is a simplified implementation
    # In production, you'd aggregate from monitoring systems
    
    return MetricsResponse(
        total_requests=1000,  # Placeholder
        successful_requests=950,
        failed_requests=50,
        average_response_time=2.5,
        total_videos_generated=100,
        average_generation_time=45.0,
        cache_hit_rate=0.75,
        active_jobs=3,
        queue_length=2,
        gpu_utilization=65.0,
    )


@app.post("/api/v1/generate", response_model=JobResponse)
async def generate_video(request: VideoGenerationRequest):
    """Generate a video from text prompt."""
    try:
        # Create job in database
        job = db_manager.create_job(
            prompt=request.prompt,
            duration=request.duration,
            quality=request.quality,
            fps=request.fps,
            resolution=request.resolution,
            seed=request.seed,
            guidance_scale=request.guidance_scale,
            num_inference_steps=request.num_inference_steps,
        )
        
        # Start async processing
        asyncio.create_task(orchestrator.process_request(str(job.id), request))
        
        logger.info(
            "Video generation job created",
            job_id=str(job.id),
            prompt=request.prompt[:100] + "..." if len(request.prompt) > 100 else request.prompt,
        )
        
        return JobResponse(
            job_id=str(job.id),
            status=job.status,
            created_at=job.created_at,
            progress=0.0,
        )
        
    except Exception as e:
        logger.error("Failed to create video generation job", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create job: {e}",
        )


@app.get("/api/v1/jobs/{job_id}/status", response_model=JobStatusResponse)
async def get_job_status(job_id: str):
    """Get job status and progress."""
    try:
        job_uuid = uuid.UUID(job_id)
        job = db_manager.get_job(job_uuid)
        
        if not job:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Job not found",
            )
        
        # Calculate progress based on status
        progress_map = {
            "pending": 0.0,
            "processing": 0.5,
            "completed": 1.0,
            "failed": 0.0,
            "cancelled": 0.0,
        }
        
        return JobStatusResponse(
            job_id=job_id,
            status=job.status,
            progress=progress_map.get(job.status.value, 0.0),
            created_at=job.created_at,
            updated_at=job.updated_at,
            completed_at=job.completed_at,
            intent_analysis=job.intent_analysis,
            routing_decision=job.routing_decision,
            prompt_optimization=job.prompt_optimization,
            generation_result=job.generation_result,
            postprocessing_result=job.postprocessing_result,
            error_message=job.error_message,
            error_code=job.error_code,
        )
        
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid job ID format",
        )
    except Exception as e:
        logger.error("Failed to get job status", job_id=job_id, error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get job status: {e}",
        )


@app.get("/api/v1/jobs/{job_id}/download")
async def download_video(job_id: str):
    """Download generated video."""
    try:
        job_uuid = uuid.UUID(job_id)
        job = db_manager.get_job(job_uuid)
        
        if not job:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Job not found",
            )
        
        if job.status.value != "completed":
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Job not completed yet",
            )
        
        # Get video path from generation result
        if not job.generation_result or "video_path" not in job.generation_result:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Video file not found",
            )
        
        video_path = job.generation_result["video_path"]
        
        # In production, you'd serve this through a CDN or object storage
        # For now, return the path
        return {"download_url": f"/api/v1/files/{video_path}"}
        
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid job ID format",
        )
    except Exception as e:
        logger.error("Failed to get download link", job_id=job_id, error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get download link: {e}",
        )


@app.post("/api/v1/evaluate/{job_id}")
async def evaluate_video(job_id: str, include_benchmarks: bool = True, evaluation_level: str = "standard"):
    """Evaluate a completed video generation job."""
    try:
        job_uuid = uuid.UUID(job_id)
        job = db_manager.get_job(job_uuid)
        
        if not job:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Job not found",
            )
        
        if job.status.value != "completed":
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Job not completed yet",
            )
        
        if not job.generation_result:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No generation result found for job",
            )
        
        # Get evaluation orchestrator
        evaluation_orchestrator = get_evaluation_orchestrator()
        
        # Import evaluation types
        # from ..services.evaluation.evaluation_orchestrator import EvaluationRequest, EvaluationLevel
        
        # Map string to enum
        level_mapping = {
            "basic": EvaluationLevel.BASIC,
            "standard": EvaluationLevel.STANDARD,
            "comprehensive": EvaluationLevel.COMPREHENSIVE,
            "expert": EvaluationLevel.EXPERT
        }
        
        eval_level = level_mapping.get(evaluation_level.lower(), EvaluationLevel.STANDARD)
        
        # Create evaluation request
        eval_request = EvaluationRequest(
            job_id=job_id,
            video_path=job.generation_result.get("video_path", "/tmp/generated_video.mp4"),
            original_prompt=job.prompt,
            evaluation_level=eval_level,
            include_benchmarks=include_benchmarks,
            generation_metadata={
                "generation_time": job.generation_result.get("generation_time", 0),
                "estimated_cost": job.generation_result.get("estimated_cost", 0),
                "model_used": job.generation_result.get("model_used", "unknown")
            }
        )
        
        # Run evaluation
        eval_response = await evaluation_orchestrator.evaluate_video_comprehensive(eval_request)
        
        return {
            "job_id": job_id,
            "evaluation_id": eval_response.evaluation_id,
            "overall_score": eval_response.evaluation_result.quality_dimensions.overall_quality_score,
            "quality_grade": eval_response.evaluation_result.quality_dimensions.quality_grade.value,
            "dimension_scores": eval_response.evaluation_result.quality_dimensions.get_dimension_scores(),
            "executive_summary": eval_response.executive_summary,
            "key_insights": eval_response.key_insights,
            "improvement_recommendations": eval_response.improvement_recommendations,
            "benchmark_comparisons": {
                name: {
                    "overall_score": comp.overall_score,
                    "percentile_rank": comp.percentile_rank,
                    "performance_level": comp.performance_level,
                    "strengths": comp.strengths,
                    "improvement_areas": comp.improvement_areas
                }
                for name, comp in (eval_response.benchmark_comparisons or {}).items()
            },
            "processing_time": eval_response.processing_time,
            "created_at": eval_response.created_at
        }
        
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid job ID format",
        )
    except Exception as e:
        logger.error("Failed to evaluate video", job_id=job_id, error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to evaluate video: {e}",
        )


# @app.post("/api/v1/feedback/{evaluation_id}")
@app.get("/api/v1/evaluation/benchmarks")
async def get_benchmark_summary():
    """Get summary of available benchmarks."""
    try:
        # from ..services.evaluation.benchmarks import get_benchmark_service
        
        benchmark_service = get_benchmark_service()
        summary = benchmark_service.get_benchmark_summary()
        
        return summary
        
    except Exception as e:
        logger.error("Failed to get benchmark summary", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get benchmark summary: {e}",
        )


@app.get("/api/v1/evaluation/performance-report")
async def get_performance_report():
    """Get comprehensive system performance report."""
    try:
        evaluation_orchestrator = get_evaluation_orchestrator()
        
        report = await evaluation_orchestrator.get_system_performance_report()
        
        return report
        
    except Exception as e:
        logger.error("Failed to get performance report", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get performance report: {e}",
        )


@app.get("/api/v1/evaluation/status")
async def get_evaluation_system_status():
    """Get evaluation system status and configuration."""
    try:
        evaluation_orchestrator = get_evaluation_orchestrator()
        
        status_info = evaluation_orchestrator.get_orchestrator_status()
        
        return status_info
        
    except Exception as e:
        logger.error("Failed to get evaluation system status", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get evaluation system status: {e}",
        )


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "visionflow.api.main:app",
        host=settings.api.host,
        port=settings.api.port,
        reload=settings.api.reload,
        log_level=settings.api.log_level,
        workers=settings.api.workers,
    )
