"""Real WAN 2.1 orchestrator with database updates."""

import asyncio
import os
from typing import Dict, Any
from ..shared.models import VideoGenerationRequest, JobStatus
from ..shared.monitoring import get_logger
from ..shared.database import get_session_factory

logger = get_logger(__name__)

class WanOrchestrator:
    """Real WAN 2.1 orchestrator with database integration."""
    
    def __init__(self):
        self.SessionLocal = get_session_factory()
        # Set HuggingFace token for the service
        if not os.getenv('HUGGINGFACE_TOKEN'):
            os.environ['HUGGINGFACE_TOKEN'] = 'hf_YOUR_HUGGINGFACE_TOKEN_HERE'
        logger.info("Real WAN 2.1 orchestrator initialized with database support")
    
    async def process_request(self, job_id: str, request: VideoGenerationRequest) -> Dict[str, Any]:
        """Dispatch video generation to celery worker."""
        logger.info(f"🎬 Dispatching WAN 2.1 generation job {job_id} to celery worker: '{request.prompt[:50]}...'")
        
        try:
            # Import celery tasks 
            from ..tasks import generate_video
            
            # Prepare task parameters
            task_kwargs = {
                "duration": request.duration,
                "fps": request.fps, 
                "resolution": request.resolution,
                "seed": request.seed,
                "guidance_scale": request.guidance_scale,
                "num_inference_steps": request.num_inference_steps,
                "quality": request.quality.value
            }
            
            # Dispatch to celery worker (async processing)
            logger.info(f"🚀 Dispatching job {job_id} to celery worker...")
            task = generate_video.delay(job_id, request.prompt, **task_kwargs)
            
            logger.info(f"✅ Job {job_id} dispatched to celery worker (task_id: {task.id})")
            
            return {
                "status": "dispatched",
                "message": "Video generation job dispatched to worker",
                "job_id": job_id,
                "celery_task_id": task.id
            }
            
        except Exception as e:
            logger.error(f"Error dispatching job {job_id} to celery: {e}")
            self._update_job_status(job_id, JobStatus.FAILED, str(e))
            return {
                "status": "failed",
                "message": f"Failed to dispatch job {job_id}: {e}",
                "job_id": job_id
            }
    
    def _update_job_status(self, job_id: str, status: JobStatus, error_message: str = None):
        """Update job status in database."""
        db = self.SessionLocal()
        try:
            from ..shared.database import VideoGenerationJob
            import uuid
            from datetime import datetime
            
            job = db.query(VideoGenerationJob).filter(VideoGenerationJob.id == uuid.UUID(job_id)).first()
            if job:
                job.status = status
                job.updated_at = datetime.utcnow()
                if status == JobStatus.COMPLETED:
                    job.completed_at = datetime.utcnow()
                if error_message:
                    job.error_message = error_message
                db.commit()
                logger.info(f"Database updated: Job {job_id} status = {status}")
            else:
                logger.error(f"Job {job_id} not found in database")
        except Exception as e:
            logger.error(f"Failed to update job {job_id} status: {e}")
            db.rollback()
        finally:
            db.close()

# Create the global orchestrator instance
orchestrator = WanOrchestrator()
