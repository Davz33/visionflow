"""Celery worker configuration for VisionFlow video generation tasks."""

import os
from celery import Celery
from celery.schedules import crontab
from .shared.config import get_settings

# Get application settings
settings = get_settings()

# Create Celery application
celery_app = Celery(
    "visionflow",
    broker=settings.redis.url,
    backend=settings.redis.url,
    include=["visionflow.tasks"]
)

# Celery configuration
celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_acks_late=True,
    worker_prefetch_multiplier=1,
    task_default_retry_delay=60,
    task_max_retries=3,
    result_expires=3600,  # 1 hour
    
    # Celery Beat Schedule for periodic tasks
    beat_schedule={
        'check-fine-tuning-triggers': {
            'task': 'check_fine_tuning_triggers',
            'schedule': crontab(minute=0),  # Every hour on the hour
            'options': {
                'expires': 30 * 60,  # Task expires after 30 minutes if not executed
            }
        },
        # You can add more periodic tasks here
        'system-health-check': {
            'task': 'check_fine_tuning_triggers',
            'schedule': crontab(hour=0, minute=0),  # Daily at midnight
            'kwargs': {'health_check': True},
            'options': {
                'expires': 60 * 60,  # Task expires after 1 hour if not executed
            }
        },
    },
    beat_scheduler='celery.beat:PersistentScheduler',
)

# Auto-discover tasks
celery_app.autodiscover_tasks(["visionflow.tasks"])

if __name__ == "__main__":
    celery_app.start()
