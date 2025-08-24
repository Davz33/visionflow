"""Database models and utilities."""

import uuid
from datetime import datetime
from typing import Any, Dict, Optional

from sqlalchemy import (
    JSON,
    Column,
    DateTime,
    Enum,
    Float,
    Integer,
    String,
    Text,
    create_engine,
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.dialects.postgresql import UUID

from .config import get_settings
from .models import JobStatus, VideoQuality

Base = declarative_base()


class VideoGenerationJob(Base):
    """Video generation job database model."""
    
    __tablename__ = "video_generation_jobs"
    
    # Primary fields
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    status = Column(Enum(JobStatus), default=JobStatus.PENDING, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    completed_at = Column(DateTime, nullable=True)
    
    # Request parameters
    prompt = Column(Text, nullable=False)
    duration = Column(Integer, nullable=False)
    quality = Column(Enum(VideoQuality), nullable=False)
    fps = Column(Integer, nullable=False)
    resolution = Column(String(20), nullable=False)
    seed = Column(Integer, nullable=True)
    guidance_scale = Column(Float, nullable=False)
    num_inference_steps = Column(Integer, nullable=False)
    
    # Processing results (stored as JSON)
    intent_analysis = Column(JSON, nullable=True)
    routing_decision = Column(JSON, nullable=True)
    prompt_optimization = Column(JSON, nullable=True)
    generation_result = Column(JSON, nullable=True)
    postprocessing_result = Column(JSON, nullable=True)
    
    # Error information
    error_message = Column(Text, nullable=True)
    error_code = Column(String(50), nullable=True)
    
    # Metrics
    processing_time = Column(Float, nullable=True)
    cost = Column(Float, nullable=True)
    
    def __repr__(self) -> str:
        return f"<VideoGenerationJob(id={self.id}, status={self.status})>"


class ServiceMetrics(Base):
    """Service metrics database model."""
    
    __tablename__ = "service_metrics"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    service_name = Column(String(100), nullable=False)
    metric_name = Column(String(100), nullable=False)
    metric_value = Column(Float, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False)
    labels = Column(JSON, nullable=True)
    
    def __repr__(self) -> str:
        return f"<ServiceMetrics(service={self.service_name}, metric={self.metric_name})>"


class CacheEntry(Base):
    """Cache entry database model for persistent caching."""
    
    __tablename__ = "cache_entries"
    
    key = Column(String(255), primary_key=True)
    value = Column(JSON, nullable=False)
    expires_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    accessed_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    access_count = Column(Integer, default=0, nullable=False)
    
    def __repr__(self) -> str:
        return f"<CacheEntry(key={self.key})>"


# Database engine and session management
_engine = None
_SessionLocal = None


def get_engine():
    """Get database engine."""
    global _engine
    if _engine is None:
        settings = get_settings()
        _engine = create_engine(
            settings.database.url,
            pool_pre_ping=True,
            pool_recycle=300,
            echo=settings.debug,
        )
    return _engine


def get_session_factory():
    """Get session factory."""
    global _SessionLocal
    if _SessionLocal is None:
        _SessionLocal = sessionmaker(
            autocommit=False,
            autoflush=False,
            bind=get_engine()
        )
    return _SessionLocal


def get_db():
    """Get database session."""
    SessionLocal = get_session_factory()
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def create_tables():
    """Create database tables."""
    engine = get_engine()
    Base.metadata.create_all(bind=engine)


def drop_tables():
    """Drop database tables."""
    engine = get_engine()
    Base.metadata.drop_all(bind=engine)


class DatabaseManager:
    """Database management utilities."""
    
    def __init__(self):
        self.SessionLocal = get_session_factory()
    
    def create_job(
        self,
        prompt: str,
        duration: int,
        quality: VideoQuality,
        fps: int,
        resolution: str,
        seed: Optional[int] = None,
        guidance_scale: float = 7.5,
        num_inference_steps: int = 20,
    ) -> VideoGenerationJob:
        """Create a new video generation job."""
        with self.SessionLocal() as db:
            job = VideoGenerationJob(
                prompt=prompt,
                duration=duration,
                quality=quality,
                fps=fps,
                resolution=resolution,
                seed=seed,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
            )
            db.add(job)
            db.commit()
            db.refresh(job)
            return job
    
    def get_job(self, job_id: uuid.UUID) -> Optional[VideoGenerationJob]:
        """Get job by ID."""
        with self.SessionLocal() as db:
            return db.query(VideoGenerationJob).filter(VideoGenerationJob.id == job_id).first()
    
    def update_job_status(
        self,
        job_id: uuid.UUID,
        status: JobStatus,
        error_message: Optional[str] = None,
        error_code: Optional[str] = None,
    ) -> bool:
        """Update job status."""
        with self.SessionLocal() as db:
            job = db.query(VideoGenerationJob).filter(VideoGenerationJob.id == job_id).first()
            if not job:
                return False
            
            job.status = status
            job.updated_at = datetime.utcnow()
            
            if status == JobStatus.COMPLETED:
                job.completed_at = datetime.utcnow()
            
            if error_message:
                job.error_message = error_message
            if error_code:
                job.error_code = error_code
            
            db.commit()
            return True
    
    def update_job_result(
        self,
        job_id: uuid.UUID,
        stage: str,
        result: Dict[str, Any],
    ) -> bool:
        """Update job processing result."""
        with self.SessionLocal() as db:
            job = db.query(VideoGenerationJob).filter(VideoGenerationJob.id == job_id).first()
            if not job:
                return False
            
            if stage == "intent_analysis":
                job.intent_analysis = result
            elif stage == "routing_decision":
                job.routing_decision = result
            elif stage == "prompt_optimization":
                job.prompt_optimization = result
            elif stage == "generation_result":
                job.generation_result = result
            elif stage == "postprocessing_result":
                job.postprocessing_result = result
            
            job.updated_at = datetime.utcnow()
            db.commit()
            return True
    
    def record_metric(
        self,
        service_name: str,
        metric_name: str,
        metric_value: float,
        labels: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Record a service metric."""
        with self.SessionLocal() as db:
            metric = ServiceMetrics(
                service_name=service_name,
                metric_name=metric_name,
                metric_value=metric_value,
                labels=labels,
            )
            db.add(metric)
            db.commit()
    
    def get_cache(self, key: str) -> Optional[Any]:
        """Get cached value."""
        with self.SessionLocal() as db:
            entry = db.query(CacheEntry).filter(CacheEntry.key == key).first()
            if not entry:
                return None
            
            # Check expiration
            if entry.expires_at and entry.expires_at < datetime.utcnow():
                db.delete(entry)
                db.commit()
                return None
            
            # Update access metrics
            entry.accessed_at = datetime.utcnow()
            entry.access_count += 1
            db.commit()
            
            return entry.value
    
    def set_cache(
        self,
        key: str,
        value: Any,
        expires_at: Optional[datetime] = None,
    ) -> None:
        """Set cached value."""
        with self.SessionLocal() as db:
            # Check if entry exists
            entry = db.query(CacheEntry).filter(CacheEntry.key == key).first()
            
            if entry:
                entry.value = value
                entry.expires_at = expires_at
                entry.accessed_at = datetime.utcnow()
            else:
                entry = CacheEntry(
                    key=key,
                    value=value,
                    expires_at=expires_at,
                )
                db.add(entry)
            
            db.commit()
