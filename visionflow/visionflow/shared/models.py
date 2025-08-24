"""Pydantic models for VisionFlow API."""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, validator


class VideoQuality(str, Enum):
    """Video quality options."""
    
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    ULTRA = "ultra"


class JobStatus(str, Enum):
    """Job processing status."""
    
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class VideoGenerationRequest(BaseModel):
    """Request model for video generation."""
    
    prompt: str = Field(..., min_length=1, max_length=500, description="Text prompt for video generation")
    duration: int = Field(default=5, ge=1, le=30, description="Video duration in seconds")
    quality: VideoQuality = Field(default=VideoQuality.MEDIUM, description="Video quality setting")
    fps: int = Field(default=24, ge=12, le=60, description="Frames per second")
    resolution: str = Field(default="512x512", description="Video resolution (WxH)")
    seed: Optional[int] = Field(default=None, description="Random seed for reproducibility")
    
    # Advanced parameters
    guidance_scale: float = Field(default=7.5, ge=1.0, le=20.0, description="Guidance scale for generation")
    num_inference_steps: int = Field(default=20, ge=10, le=100, description="Number of inference steps")
    
    @validator("resolution")
    def validate_resolution(cls, v: str) -> str:
        """Validate resolution format."""
        try:
            width, height = v.split("x")
            w, h = int(width), int(height)
            if w < 64 or h < 64 or w > 2048 or h > 2048:
                raise ValueError("Resolution must be between 64x64 and 2048x2048")
            return v
        except (ValueError, IndexError):
            raise ValueError("Resolution must be in format 'WIDTHxHEIGHT'")


class IntentAnalysis(BaseModel):
    """Intent analysis result."""
    
    intent_type: str = Field(..., description="Type of video generation intent")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Extracted parameters")
    entities: List[str] = Field(default_factory=list, description="Extracted entities")
    complexity_score: float = Field(..., ge=0.0, le=1.0, description="Request complexity assessment")


class RoutingDecision(BaseModel):
    """Routing decision result."""
    
    model_tier: str = Field(..., description="Selected model tier")
    estimated_cost: float = Field(..., ge=0.0, description="Estimated processing cost")
    estimated_duration: int = Field(..., ge=0, description="Estimated processing time in seconds")
    resource_requirements: Dict[str, Any] = Field(default_factory=dict, description="Required resources")
    priority: int = Field(default=5, ge=1, le=10, description="Processing priority")


class PromptOptimization(BaseModel):
    """Prompt optimization result."""
    
    original_prompt: str = Field(..., description="Original user prompt")
    optimized_prompt: str = Field(..., description="Optimized prompt for generation")
    optimization_strategy: str = Field(..., description="Applied optimization strategy")
    quality_score: float = Field(..., ge=0.0, le=1.0, description="Expected quality improvement")
    modifications: List[str] = Field(default_factory=list, description="List of applied modifications")


class VideoGenerationResult(BaseModel):
    """Video generation result."""
    
    video_path: str = Field(..., description="Path to generated video file")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Generation metadata")
    quality_metrics: Dict[str, float] = Field(default_factory=dict, description="Quality assessment metrics")
    generation_time: float = Field(..., ge=0.0, description="Generation time in seconds")
    model_version: str = Field(..., description="Model version used")


class GenerationResult(BaseModel):
    """Enhanced video generation result with additional metrics."""
    
    video_path: str = Field(..., description="Path to generated video file")
    model_used: str = Field(..., description="Model used for generation")
    generation_time: float = Field(..., ge=0.0, description="Generation time in seconds")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Generation parameters and metadata")
    estimated_cost: float = Field(..., ge=0.0, description="Estimated cost of generation")
    quality_metrics: Dict[str, float] = Field(default_factory=dict, description="Comprehensive quality metrics")


class PostProcessingResult(BaseModel):
    """Post-processing result."""
    
    processed_video_path: str = Field(..., description="Path to processed video file")
    original_size: int = Field(..., ge=0, description="Original file size in bytes")
    processed_size: int = Field(..., ge=0, description="Processed file size in bytes")
    compression_ratio: float = Field(..., ge=0.0, description="Compression ratio achieved")
    processing_time: float = Field(..., ge=0.0, description="Processing time in seconds")
    applied_effects: List[str] = Field(default_factory=list, description="Applied post-processing effects")


class JobResponse(BaseModel):
    """Job creation response."""
    
    job_id: str = Field(..., description="Unique job identifier")
    status: JobStatus = Field(..., description="Current job status")
    created_at: datetime = Field(..., description="Job creation timestamp")
    estimated_completion: Optional[datetime] = Field(default=None, description="Estimated completion time")
    progress: float = Field(default=0.0, ge=0.0, le=1.0, description="Completion progress")


class JobStatusResponse(BaseModel):
    """Job status response."""
    
    job_id: str = Field(..., description="Job identifier")
    status: JobStatus = Field(..., description="Current job status")
    progress: float = Field(..., ge=0.0, le=1.0, description="Completion progress")
    created_at: datetime = Field(..., description="Job creation timestamp")
    updated_at: datetime = Field(..., description="Last update timestamp")
    completed_at: Optional[datetime] = Field(default=None, description="Completion timestamp")
    
    # Processing details
    intent_analysis: Optional[IntentAnalysis] = Field(default=None, description="Intent analysis result")
    routing_decision: Optional[RoutingDecision] = Field(default=None, description="Routing decision")
    prompt_optimization: Optional[PromptOptimization] = Field(default=None, description="Prompt optimization")
    generation_result: Optional[VideoGenerationResult] = Field(default=None, description="Generation result")
    postprocessing_result: Optional[PostProcessingResult] = Field(default=None, description="Post-processing result")
    
    # Error information
    error_message: Optional[str] = Field(default=None, description="Error message if failed")
    error_code: Optional[str] = Field(default=None, description="Error code if failed")


class HealthCheck(BaseModel):
    """Health check response."""
    
    status: str = Field(..., description="Service status")
    timestamp: datetime = Field(..., description="Check timestamp")
    version: str = Field(..., description="Service version")
    uptime: float = Field(..., description="Service uptime in seconds")
    
    # Service-specific health
    database_connected: bool = Field(..., description="Database connection status")
    redis_connected: bool = Field(..., description="Redis connection status")
    storage_connected: bool = Field(..., description="Storage connection status")
    model_loaded: bool = Field(..., description="Model loading status")
    mlflow_connected: Optional[bool] = Field(None, description="MLFlow connection status")
    
    # Resource usage
    cpu_usage: float = Field(..., ge=0.0, le=100.0, description="CPU usage percentage")
    memory_usage: float = Field(..., ge=0.0, le=100.0, description="Memory usage percentage")
    disk_usage: float = Field(..., ge=0.0, le=100.0, description="Disk usage percentage")


class MetricsResponse(BaseModel):
    """Service metrics response."""
    
    # Request metrics
    total_requests: int = Field(..., ge=0, description="Total requests processed")
    successful_requests: int = Field(..., ge=0, description="Successful requests")
    failed_requests: int = Field(..., ge=0, description="Failed requests")
    average_response_time: float = Field(..., ge=0.0, description="Average response time in seconds")
    
    # Video generation metrics
    total_videos_generated: int = Field(..., ge=0, description="Total videos generated")
    average_generation_time: float = Field(..., ge=0.0, description="Average generation time")
    cache_hit_rate: float = Field(..., ge=0.0, le=1.0, description="Cache hit rate")
    
    # Resource metrics
    active_jobs: int = Field(..., ge=0, description="Currently active jobs")
    queue_length: int = Field(..., ge=0, description="Current queue length")
    gpu_utilization: float = Field(..., ge=0.0, le=100.0, description="GPU utilization percentage")


class ErrorResponse(BaseModel):
    """Error response model."""
    
    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    details: Optional[Dict[str, Any]] = Field(default=None, description="Additional error details")
    timestamp: datetime = Field(..., description="Error timestamp")
    request_id: Optional[str] = Field(default=None, description="Request identifier")
    
    class Config:
        schema_extra = {
            "example": {
                "error": "ValidationError",
                "message": "Invalid prompt length",
                "details": {"field": "prompt", "constraint": "max_length"},
                "timestamp": "2024-01-01T12:00:00Z",
                "request_id": "req_123456789"
            }
        }
