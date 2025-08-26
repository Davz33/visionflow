"""Celery tasks for VisionFlow video generation."""

import asyncio
import logging
from datetime import datetime
from typing import Any, Dict
from uuid import UUID
import httpx
import json

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


@celery_app.task(bind=True, base=CallbackTask, name="check_fine_tuning_triggers")
def check_fine_tuning_triggers_task(self: Task) -> Dict[str, Any]:
    """
    Periodic task to check if fine-tuning triggers should be activated.
    Monitors system performance and alerts when model retraining is needed.
    
    Returns:
        Dictionary with trigger results and recommendations
    """
    try:
        logger.info("🔍 Starting fine-tuning trigger check")
        
        # Import here to avoid circular imports
        from .services.evaluation.confidence_manager import ConfidenceManager
        
        # Create confidence manager and check triggers
        confidence_manager = ConfidenceManager()
        
        # Run the async function in the current event loop
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            triggers = loop.run_until_complete(
                confidence_manager.check_fine_tuning_triggers()
            )
        finally:
            loop.close()
        
        # Log results
        logger.info(f"🔍 Fine-tuning trigger check completed")
        logger.info(f"   Low confidence trigger: {triggers['low_confidence_trigger']}")
        logger.info(f"   High automation trigger: {triggers['high_automation_trigger']}")
        logger.info(f"   Recommendations: {len(triggers['recommendations'])}")
        
        # If triggers are active, send alerts
        if triggers['low_confidence_trigger'] or triggers['high_automation_trigger']:
            alert_result = send_fine_tuning_alerts(triggers)
            triggers['alert_sent'] = alert_result
            logger.warning("🚨 Fine-tuning triggers ACTIVE - alerts sent to monitoring systems")
        else:
            triggers['alert_sent'] = False
            logger.info("✅ System performance within acceptable parameters")
            
        return triggers
        
    except Exception as exc:
        logger.error(f"❌ Fine-tuning trigger check failed: {exc}")
        # Don't re-raise to prevent celery from retrying indefinitely
        return {
            "error": str(exc),
            "status": "failed",
            "timestamp": datetime.utcnow().isoformat()
        }


def send_fine_tuning_alerts(triggers: Dict[str, Any]) -> bool:
    """
    Send alerts when fine-tuning is needed.
    
    Args:
        triggers: Dictionary with trigger results and recommendations
        
    Returns:
        True if alerts were sent successfully, False otherwise
    """
    try:
        # Prepare alert message
        alert_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "service": "visionflow-evaluation",
            "severity": "high" if triggers.get('low_confidence_trigger') else "medium",
            "summary": "VisionFlow fine-tuning triggers activated",
            "details": {
                "low_confidence_trigger": triggers.get('low_confidence_trigger', False),
                "high_automation_trigger": triggers.get('high_automation_trigger', False),
                "recommendations": triggers.get('recommendations', [])
            },
            "metrics": triggers.get('metrics', {})
        }
        
        # Log to structured format for monitoring systems to pick up
        logger.warning(f"FINE_TUNING_ALERT: {json.dumps(alert_data, indent=2)}")
        
        # TODO: Integrate with your alerting system (Slack, email, PagerDuty, etc.)
        # For now, we'll just log comprehensively
        
        if triggers.get('low_confidence_trigger'):
            logger.critical("🚨 URGENT: Model confidence degraded - immediate retraining recommended")
            
        if triggers.get('high_automation_trigger'):
            logger.warning("⚠️ High automation rate detected - quality gates may need tightening")
        
        # Print recommendations for easy visibility
        for i, rec in enumerate(triggers.get('recommendations', []), 1):
            logger.warning(f"   Recommendation {i}: {rec.get('type', 'Unknown')} ({rec.get('priority', 'medium')} priority)")
            logger.warning(f"     Reason: {rec.get('reason', 'No reason provided')}")
            
        return True
        
    except Exception as exc:
        logger.error(f"Failed to send fine-tuning alerts: {exc}")
        return False
