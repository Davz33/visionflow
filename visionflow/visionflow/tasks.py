"""Celery tasks for VisionFlow video generation."""

import asyncio
import logging
from datetime import datetime
from typing import Any, Dict
from uuid import UUID
import httpx

from celery import Task
from .worker import celery_app
from .shared.config import get_settings
from .shared.database import get_session_factory, VideoGenerationJob
from .shared.models import JobStatus, VideoGenerationRequest, VideoQuality

logger = logging.getLogger(__name__)


class CallbackTask(Task):
    """Base task class with callback support."""
    
    def on_success(self, retval: Any, task_id: str, args: Any, kwargs: Any) -> None:
        """Called when task succeeds."""
        logger.info(f"Task {task_id} completed successfully")
    
    def on_failure(self, exc: Exception, task_id: str, args: Any, kwargs: Any, einfo: Any) -> None:
        """Called when task fails."""
        logger.error(f"Task {task_id} failed: {exc}")


@celery_app.task(bind=True, base=CallbackTask, name="generate_video")
def generate_video(self: Task, job_id: str, prompt: str, **kwargs: Any) -> Dict[str, Any]:
    """
    Generate video using WAN 2.1 model.
    
    Args:
        job_id: UUID of the video generation job
        prompt: Text prompt for video generation
        **kwargs: Additional generation parameters
        
    Returns:
        Dictionary with generation results
    """
    try:
        logger.info(f"Starting video generation for job {job_id}")
        
        # Update job status to processing
        SessionLocal = get_session_factory()
        with SessionLocal() as db:
            job = db.query(VideoGenerationJob).filter(
                VideoGenerationJob.id == UUID(job_id)
            ).first()
            
            if not job:
                raise ValueError(f"Job {job_id} not found")
            
            job.status = JobStatus.PROCESSING
            db.commit()
        
        # Prepare generation request for HTTP call
        generation_payload = {
            "prompt": prompt,
            "original_prompt": prompt,
            "generation_params": {
                "duration": kwargs.get("duration", 5),
                "fps": kwargs.get("fps", 24),
                "resolution": kwargs.get("resolution", "512x512"),
                "seed": kwargs.get("seed"),
                "guidance_scale": kwargs.get("guidance_scale", 7.5),
                "num_inference_steps": kwargs.get("num_inference_steps", 50),
                "quality": kwargs.get("quality", "medium")
            },
            "routing_decision": {"service": "wan2.1", "reason": "default"},
            "job_id": job_id
        }
        
        # Call generation service via HTTP (lightweight!)
        logger.info(f"🔄 Delegating job {job_id} to real WAN generation service at http://host.docker.internal:8002")
        with httpx.Client(timeout=600.0) as client:  # 10 minute timeout for real generation
            response = client.post(
                "http://host.docker.internal:8002/generate",
                json=generation_payload
            )
            response.raise_for_status()
            result = response.json()
        
        # Update job with results
        with SessionLocal() as db:
            job = db.query(VideoGenerationJob).filter(
                VideoGenerationJob.id == UUID(job_id)
            ).first()
            
            if job:
                job.status = JobStatus.COMPLETED
                job.generation_result = result
                job.completed_at = datetime.utcnow()
                db.commit()
        
        logger.info(f"Video generation completed for job {job_id}")
        return result
        
    except Exception as exc:
        logger.error(f"Video generation failed for job {job_id}: {exc}")
        
        # Update job status to failed
        try:
            with SessionLocal() as db:
                job = db.query(VideoGenerationJob).filter(
                    VideoGenerationJob.id == UUID(job_id)
                ).first()
                
                if job:
                    job.status = JobStatus.FAILED
                    job.error_message = str(exc)
                    job.error_code = type(exc).__name__
                    db.commit()
        except Exception as db_exc:
            logger.error(f"Failed to update job status: {db_exc}")
        
        # Re-raise the exception for Celery to handle
        raise exc


@celery_app.task(bind=True, base=CallbackTask, name="process_video_workflow")
def process_video_workflow(
    self: Task, 
    job_id: str, 
    prompt: str, 
    **kwargs: Any
) -> Dict[str, Any]:
    """
    Process complete video generation workflow including orchestration.
    
    Args:
        job_id: UUID of the video generation job
        prompt: Text prompt for video generation
        **kwargs: Additional workflow parameters
        
    Returns:
        Dictionary with workflow results
    """
    try:
        logger.info(f"Starting video workflow for job {job_id}")
        
        # This would integrate with the orchestration service
        # For now, delegate to the basic video generation task
        return generate_video.apply_async(args=[job_id, prompt], kwargs=kwargs).get()
        
    except Exception as exc:
        logger.error(f"Video workflow failed for job {job_id}: {exc}")
        raise exc
