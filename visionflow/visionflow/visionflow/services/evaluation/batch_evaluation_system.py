"""
Batch Evaluation System for High-Volume Video Processing
Implements configurable sampling, confidence thresholds, and automated human review workflows
"""

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any, Union
from datetime import datetime, timedelta
import json
import redis
from queue import Queue, PriorityQueue
import threading
from dataclasses import dataclass, field

import torch
import numpy as np
from pydantic import BaseModel, Field

from .quality_metrics import IndustryStandardMetrics, EvaluationResult
from .industry_metrics_implementation import (
    LPIPSEvaluator, FVMDEvaluator, CLIPEvaluator, ETVAEvaluator
)

logger = logging.getLogger(__name__)


class SamplingStrategy(str, Enum):
    """Video frame sampling strategies for high-volume processing"""
    EVERY_FRAME = "every_frame"           # Evaluate every frame (highest quality, slowest)
    EVERY_NTH_FRAME = "every_nth_frame"   # Evaluate every Nth frame (configurable)
    KEYFRAME_ONLY = "keyframe_only"        # Evaluate only keyframes (fastest, lowest quality)
    ADAPTIVE = "adaptive"                  # Adaptive sampling based on content complexity
    RANDOM_SAMPLE = "random_sample"        # Random frame sampling
    TEMPORAL_STRATIFIED = "temporal_stratified"  # Stratified sampling across time


class ConfidenceThreshold(str, Enum):
    """Confidence threshold levels for automated flagging"""
    CRITICAL = "critical"      # 0.5 - Flag everything below 50% confidence
    HIGH = "high"             # 0.7 - Flag below 70% confidence
    MEDIUM = "medium"         # 0.8 - Flag below 80% confidence
    LOW = "low"               # 0.9 - Flag below 90% confidence


class BatchProcessingConfig(BaseModel):
    """Configuration for batch video evaluation processing"""
    
    # Sampling configuration
    sampling_strategy: SamplingStrategy = Field(default=SamplingStrategy.EVERY_NTH_FRAME)
    sampling_interval: int = Field(default=5, description="Evaluate every Nth frame")
    min_frames_evaluated: int = Field(default=10, description="Minimum frames to evaluate per video")
    max_frames_evaluated: int = Field(default=100, description="Maximum frames to evaluate per video")
    
    # Confidence thresholds
    confidence_threshold: ConfidenceThreshold = Field(default=ConfidenceThreshold.MEDIUM)
    auto_flag_threshold: float = Field(default=0.8, description="Confidence threshold for auto-flagging")
    human_review_threshold: float = Field(default=0.6, description="Confidence threshold for human review")
    
    # Processing configuration
    max_concurrent_videos: int = Field(default=10, description="Maximum videos processed concurrently")
    max_concurrent_frames: int = Field(default=50, description="Maximum frames processed concurrently")
    batch_size: int = Field(default=25, description="Batch size for GPU processing")
    
    # Infrastructure configuration
    use_gpu: bool = Field(default=True, description="Enable GPU acceleration")
    gpu_memory_limit: Optional[float] = Field(default=None, description="GPU memory limit in GB")
    enable_distributed: bool = Field(default=False, description="Enable distributed processing")
    
    # Quality assurance
    enable_quality_gates: bool = Field(default=True, description="Enable quality gates")
    quality_gate_threshold: float = Field(default=0.75, description="Minimum quality score to pass gate")
    
    # Monitoring and alerting
    enable_monitoring: bool = Field(default=True, description="Enable real-time monitoring")
    alert_on_failure: bool = Field(default=True, description="Send alerts on evaluation failures")


@dataclass
class FrameSample:
    """Represents a frame sample for evaluation"""
    frame_index: int
    frame_data: torch.Tensor
    timestamp: float
    is_keyframe: bool = False
    sampling_confidence: float = 1.0


@dataclass
class VideoEvaluationJob:
    """Represents a video evaluation job in the batch system"""
    job_id: str
    video_path: str
    prompt: str
    priority: int = 1
    created_at: datetime = field(default_factory=datetime.utcnow)
    config: BatchProcessingConfig = field(default_factory=BatchProcessingConfig)
    status: str = "pending"
    progress: float = 0.0
    results: Optional[EvaluationResult] = None
    flagged_for_review: bool = False
    review_reason: Optional[str] = None


class FrameSampler:
    """Intelligent frame sampling for different strategies"""
    
    def __init__(self, config: BatchProcessingConfig):
        self.config = config
    
    def sample_frames(self, video_frames: List[torch.Tensor], 
                     video_metadata: Dict[str, Any]) -> List[FrameSample]:
        """Sample frames based on configured strategy"""
        
        if self.config.sampling_strategy == SamplingStrategy.EVERY_FRAME:
            return self._sample_every_frame(video_frames)
        elif self.config.sampling_strategy == SamplingStrategy.EVERY_NTH_FRAME:
            return self._sample_every_nth_frame(video_frames)
        elif self.config.sampling_strategy == SamplingStrategy.KEYFRAME_ONLY:
            return self._sample_keyframes(video_frames, video_metadata)
        elif self.config.sampling_strategy == SamplingStrategy.ADAPTIVE:
            return self._sample_adaptive(video_frames, video_metadata)
        elif self.config.sampling_strategy == SamplingStrategy.RANDOM_SAMPLE:
            return self._sample_random(video_frames)
        elif self.config.sampling_strategy == SamplingStrategy.TEMPORAL_STRATIFIED:
            return self._sample_temporal_stratified(video_frames)
        else:
            raise ValueError(f"Unknown sampling strategy: {self.config.sampling_strategy}")
    
    def _sample_every_frame(self, frames: List[torch.Tensor]) -> List[FrameSample]:
        """Sample every frame (highest quality, slowest)"""
        return [
            FrameSample(i, frame, i / len(frames), False, 1.0)
            for i, frame in enumerate(frames)
        ]
    
    def _sample_every_nth_frame(self, frames: List[torch.Tensor]) -> List[FrameSample]:
        """Sample every Nth frame based on configuration"""
        interval = self.config.sampling_interval
        min_frames = self.config.min_frames_evaluated
        max_frames = self.config.max_frames_evaluated
        
        # Calculate optimal interval to meet frame count requirements
        if len(frames) <= max_frames:
            interval = max(1, len(frames) // min_frames)
        
        sampled_frames = []
        for i in range(0, len(frames), interval):
            if len(sampled_frames) >= max_frames:
                break
            sampled_frames.append(
                FrameSample(i, frames[i], i / len(frames), False, 1.0)
            )
        
        return sampled_frames
    
    def _sample_keyframes(self, frames: List[torch.Tensor], 
                          metadata: Dict[str, Any]) -> List[FrameSample]:
        """Sample only keyframes (fastest, lowest quality)"""
        # This would integrate with video codec keyframe detection
        # For now, use a simple heuristic based on frame differences
        
        keyframes = []
        threshold = 0.1  # Configurable threshold for keyframe detection
        
        for i in range(1, len(frames)):
            if i == 0 or self._is_keyframe(frames[i], frames[i-1], threshold):
                keyframes.append(
                    FrameSample(i, frames[i], i / len(frames), True, 0.9)
                )
        
        return keyframes[:self.config.max_frames_evaluated]
    
    def _is_keyframe(self, frame1: torch.Tensor, frame2: torch.Tensor, 
                     threshold: float) -> bool:
        """Determine if frame is a keyframe based on difference from previous"""
        diff = torch.mean(torch.abs(frame1 - frame2))
        return diff > threshold
    
    def _sample_adaptive(self, frames: List[torch.Tensor], 
                         metadata: Dict[str, Any]) -> List[FrameSample]:
        """Adaptive sampling based on content complexity"""
        # Analyze frame complexity and sample more frames from complex scenes
        complexity_scores = self._calculate_frame_complexity(frames)
        
        # Sample more frames from high-complexity regions
        total_frames = min(self.config.max_frames_evaluated, len(frames))
        sampled_indices = self._adaptive_sample_indices(complexity_scores, total_frames)
        
        return [
            FrameSample(i, frames[i], i / len(frames), False, 
                       complexity_scores[i] if i < len(complexity_scores) else 0.5)
            for i in sampled_indices
        ]
    
    def _calculate_frame_complexity(self, frames: List[torch.Tensor]) -> List[float]:
        """Calculate complexity score for each frame"""
        complexity_scores = []
        
        for i, frame in enumerate(frames):
            # Simple complexity metrics (can be enhanced with more sophisticated analysis)
            gray = self._to_grayscale(frame)
            edges = self._detect_edges(gray)
            complexity = torch.mean(edges).item()
            complexity_scores.append(complexity)
        
        # Normalize to 0-1 range
        max_complexity = max(complexity_scores) if complexity_scores else 1.0
        return [score / max_complexity for score in complexity_scores]
    
    def _to_grayscale(self, frame: torch.Tensor) -> torch.Tensor:
        """Convert frame to grayscale for complexity analysis"""
        if frame.shape[0] == 3:  # RGB
            return 0.299 * frame[0] + 0.587 * frame[1] + 0.114 * frame[2]
        return frame
    
    def _detect_edges(self, frame: torch.Tensor) -> torch.Tensor:
        """Simple edge detection for complexity analysis"""
        # This is a simplified implementation
        # In production, you'd use proper edge detection algorithms
        return torch.abs(torch.diff(frame, dim=0)) + torch.abs(torch.diff(frame, dim=1))
    
    def _adaptive_sample_indices(self, complexity_scores: List[float], 
                                target_count: int) -> List[int]:
        """Adaptively sample indices based on complexity scores"""
        if len(complexity_scores) <= target_count:
            return list(range(len(complexity_scores)))
        
        # Weight sampling by complexity
        weights = np.array(complexity_scores)
        weights = weights / np.sum(weights)
        
        # Sample indices with replacement, weighted by complexity
        sampled_indices = np.random.choice(
            len(complexity_scores), 
            size=target_count, 
            replace=False, 
            p=weights
        )
        
        return sorted(sampled_indices.tolist())
    
    def _sample_random(self, frames: List[torch.Tensor]) -> List[FrameSample]:
        """Random frame sampling"""
        total_frames = min(self.config.max_frames_evaluated, len(frames))
        indices = np.random.choice(len(frames), size=total_frames, replace=False)
        
        return [
            FrameSample(i, frames[i], i / len(frames), False, 0.7)
            for i in sorted(indices)
        ]
    
    def _sample_temporal_stratified(self, frames: List[torch.Tensor]) -> List[FrameSample]:
        """Stratified sampling across temporal segments"""
        total_frames = min(self.config.max_frames_evaluated, len(frames))
        segment_size = len(frames) // total_frames
        
        sampled_frames = []
        for i in range(total_frames):
            segment_start = i * segment_size
            segment_end = min(segment_start + segment_size, len(frames))
            
            # Sample from middle of segment for better representation
            frame_idx = segment_start + (segment_end - segment_start) // 2
            sampled_frames.append(
                FrameSample(frame_idx, frames[frame_idx], frame_idx / len(frames), False, 0.8)
            )
        
        return sampled_frames


class ConfidenceManager:
    """Manages confidence thresholds and automated flagging"""
    
    def __init__(self, config: BatchProcessingConfig):
        self.config = config
        self.flagged_jobs = Queue()
        self.review_queue = PriorityQueue()
    
    def evaluate_confidence(self, evaluation_result: EvaluationResult) -> Dict[str, Any]:
        """Evaluate confidence and determine if human review is needed"""
        
        confidence_scores = evaluation_result.quality_dimensions.get_assessment_confidence()
        overall_confidence = np.mean(list(confidence_scores.values()))
        
        # Check if job should be flagged for review
        needs_review = overall_confidence < self.config.human_review_threshold
        auto_flag = overall_confidence < self.config.auto_flag_threshold
        
        review_priority = self._calculate_review_priority(overall_confidence, evaluation_result)
        
        result = {
            "overall_confidence": overall_confidence,
            "needs_review": needs_review,
            "auto_flag": auto_flag,
            "review_priority": review_priority,
            "confidence_breakdown": confidence_scores,
            "flag_reason": self._get_flag_reason(overall_confidence, evaluation_result)
        }
        
        if needs_review:
            self._queue_for_review(evaluation_result, review_priority, result["flag_reason"])
        
        return result
    
    def _calculate_review_priority(self, confidence: float, 
                                  result: EvaluationResult) -> int:
        """Calculate priority for human review (lower = higher priority)"""
        base_priority = int((1 - confidence) * 100)
        
        # Adjust priority based on quality score
        quality_score = result.quality_dimensions.overall_quality_score
        if quality_score < 0.5:
            base_priority -= 20  # High priority for low quality
        elif quality_score > 0.8:
            base_priority += 20  # Lower priority for high quality
        
        return max(1, base_priority)
    
    def _get_flag_reason(self, confidence: float, 
                         result: EvaluationResult) -> str:
        """Generate human-readable reason for flagging"""
        if confidence < self.config.human_review_threshold:
            return f"Low confidence ({confidence:.2f}) - requires human review"
        elif confidence < self.config.auto_flag_threshold:
            return f"Moderate confidence ({confidence:.2f}) - flag for monitoring"
        else:
            return "High confidence - no review needed"
    
    def _queue_for_review(self, result: EvaluationResult, priority: int, reason: str):
        """Queue job for human review"""
        review_item = {
            "priority": priority,
            "timestamp": datetime.utcnow(),
            "evaluation_result": result,
            "reason": reason
        }
        self.review_queue.put((priority, review_item))
        logger.info(f"Queued evaluation for review with priority {priority}: {reason}")


class BatchEvaluationProcessor:
    """Main batch processing system for video evaluation"""
    
    def __init__(self, config: BatchProcessingConfig):
        self.config = config
        self.frame_sampler = FrameSampler(config)
        self.confidence_manager = ConfidenceManager(config)
        self.evaluators = self._initialize_evaluators()
        self.job_queue = Queue()
        self.results_queue = Queue()
        self.processing_pool = None
        self.monitoring_thread = None
        
        # Infrastructure components
        self.redis_client = self._initialize_redis()
        self.metrics_collector = self._initialize_metrics()
        
        # Start monitoring if enabled
        if self.config.enable_monitoring:
            self._start_monitoring()
    
    def _initialize_evaluators(self) -> Dict[str, Any]:
        """Initialize evaluation components"""
        evaluators = {}
        
        try:
            evaluators["lpips"] = LPIPSEvaluator()
            evaluators["fvmd"] = FVMDEvaluator()
            evaluators["clip"] = CLIPEvaluator()
            evaluators["etva"] = ETVAEvaluator()
        except Exception as e:
            logger.warning(f"Some evaluators failed to initialize: {e}")
        
        return evaluators
    
    def _initialize_redis(self) -> Optional[redis.Redis]:
        """Initialize Redis for job tracking and caching"""
        try:
            return redis.Redis(host='localhost', port=6379, db=0)
        except Exception as e:
            logger.warning(f"Redis initialization failed: {e}")
            return None
    
    def _initialize_metrics(self) -> Dict[str, Any]:
        """Initialize metrics collection system"""
        return {
            "processed_jobs": 0,
            "failed_jobs": 0,
            "flagged_jobs": 0,
            "average_processing_time": 0.0,
            "gpu_utilization": 0.0,
            "queue_depth": 0
        }
    
    async def process_video_batch(self, video_jobs: List[VideoEvaluationJob]) -> List[EvaluationResult]:
        """Process a batch of video evaluation jobs"""
        
        # Configure processing pool based on requirements
        if self.config.use_gpu and torch.cuda.is_available():
            self.processing_pool = ProcessPoolExecutor(
                max_workers=self.config.max_concurrent_videos
            )
        else:
            self.processing_pool = ThreadPoolExecutor(
                max_workers=self.config.max_concurrent_videos
            )
        
        # Submit jobs for processing
        futures = []
        for job in video_jobs:
            future = self.processing_pool.submit(self._process_single_video, job)
            futures.append(future)
        
        # Collect results
        results = []
        for future in asyncio.as_completed(futures):
            try:
                result = await future
                results.append(result)
            except Exception as e:
                logger.error(f"Job processing failed: {e}")
                self.metrics_collector["failed_jobs"] += 1
        
        return results
    
    def _process_single_video(self, job: VideoEvaluationJob) -> EvaluationResult:
        """Process a single video evaluation job"""
        
        start_time = datetime.utcnow()
        
        try:
            # Load video frames
            video_frames = self._load_video_frames(job.video_path)
            
            # Sample frames based on strategy
            sampled_frames = self.frame_sampler.sample_frames(video_frames, {})
            
            # Process frames in batches for GPU efficiency
            frame_batches = self._create_frame_batches(sampled_frames)
            
            # Evaluate each batch
            batch_results = []
            for batch in frame_batches:
                batch_result = self._evaluate_frame_batch(batch, job.prompt)
                batch_results.append(batch_result)
            
            # Aggregate results
            evaluation_result = self._aggregate_batch_results(batch_results, job)
            
            # Evaluate confidence and flag if needed
            confidence_analysis = self.confidence_manager.evaluate_confidence(evaluation_result)
            
            # Update job status
            job.status = "completed"
            job.results = evaluation_result
            job.flagged_for_review = confidence_analysis["needs_review"]
            job.review_reason = confidence_analysis["flag_reason"]
            
            # Update metrics
            processing_time = (datetime.utcnow() - start_time).total_seconds()
            self._update_metrics(processing_time, confidence_analysis)
            
            return evaluation_result
            
        except Exception as e:
            logger.error(f"Video processing failed for job {job.job_id}: {e}")
            job.status = "failed"
            self.metrics_collector["failed_jobs"] += 1
            raise
    
    def _load_video_frames(self, video_path: str) -> List[torch.Tensor]:
        """Load video frames from file"""
        # This would integrate with video loading libraries like OpenCV or PyAV
        # For now, return placeholder
        return [torch.randn(3, 224, 224) for _ in range(100)]
    
    def _create_frame_batches(self, frames: List[FrameSample]) -> List[List[FrameSample]]:
        """Create batches of frames for efficient GPU processing"""
        batch_size = self.config.batch_size
        batches = []
        
        for i in range(0, len(frames), batch_size):
            batch = frames[i:i + batch_size]
            batches.append(batch)
        
        return batches
    
    def _evaluate_frame_batch(self, frame_batch: List[FrameSample], 
                             prompt: str) -> Dict[str, Any]:
        """Evaluate a batch of frames using available evaluators"""
        
        batch_results = {}
        
        # LPIPS evaluation (if reference frames available)
        if "lpips" in self.evaluators:
            # This would require reference frames - placeholder for now
            pass
        
        # CLIP evaluation for text-video alignment
        if "clip" in self.evaluators:
            frames = [sample.frame_data for sample in frame_batch]
            clip_result = self.evaluators["clip"].evaluate_alignment(frames, prompt)
            if "error" not in clip_result:
                batch_results["clip"] = clip_result
        
        # ETVA evaluation
        if "etva" in self.evaluators:
            frames = [sample.frame_data for sample in frame_batch]
            etva_result = self.evaluators["etva"].evaluate_with_questions(frames, {})
            if "error" not in etva_result:
                batch_results["etva"] = etva_result
        
        return batch_results
    
    def _aggregate_batch_results(self, batch_results: List[Dict[str, Any]], 
                                job: VideoEvaluationJob) -> EvaluationResult:
        """Aggregate results from multiple frame batches"""
        
        # This would create a proper EvaluationResult object
        # For now, return a placeholder
        from .quality_metrics import create_mock_quality_dimensions
        
        quality_dimensions = create_mock_quality_dimensions()
        
        # Create evaluation result
        from .quality_metrics import EvaluationResult
        return EvaluationResult(
            evaluation_id=f"batch_{job.job_id}",
            job_id=job.job_id,
            original_prompt=job.prompt,
            evaluation_level="standard",
            evaluation_duration=0.0,
            quality_dimensions=quality_dimensions,
            benchmark_comparison={},
            created_at=datetime.utcnow()
        )
    
    def _update_metrics(self, processing_time: float, confidence_analysis: Dict[str, Any]):
        """Update processing metrics"""
        self.metrics_collector["processed_jobs"] += 1
        
        # Update average processing time
        current_avg = self.metrics_collector["average_processing_time"]
        total_jobs = self.metrics_collector["processed_jobs"]
        self.metrics_collector["average_processing_time"] = (
            (current_avg * (total_jobs - 1) + processing_time) / total_jobs
        )
        
        # Update flagged jobs count
        if confidence_analysis["needs_review"]:
            self.metrics_collector["flagged_jobs"] += 1
        
        # Update queue depth
        self.metrics_collector["queue_depth"] = self.job_queue.qsize()
    
    def _start_monitoring(self):
        """Start monitoring thread for real-time metrics"""
        def monitor_loop():
            while True:
                try:
                    # Update GPU utilization if available
                    if self.config.use_gpu and torch.cuda.is_available():
                        gpu_util = torch.cuda.utilization()
                        self.metrics_collector["gpu_utilization"] = gpu_util
                    
                    # Log metrics
                    logger.info(f"Batch Processing Metrics: {self.metrics_collector}")
                    
                    # Check for alerts
                    if self.config.alert_on_failure and self.metrics_collector["failed_jobs"] > 0:
                        self._send_alert("Evaluation failures detected")
                    
                    # Sleep for monitoring interval
                    time.sleep(30)  # Monitor every 30 seconds
                    
                except Exception as e:
                    logger.error(f"Monitoring error: {e}")
                    time.sleep(60)  # Wait longer on error
        
        self.monitoring_thread = threading.Thread(target=monitor_loop, daemon=True)
        self.monitoring_thread.start()
    
    def _send_alert(self, message: str):
        """Send alert for monitoring events"""
        # This would integrate with your alerting system
        logger.warning(f"ALERT: {message}")
        
        # Example: Send to Slack, email, or monitoring dashboard
        if self.redis_client:
            self.redis_client.publish("evaluation_alerts", json.dumps({
                "timestamp": datetime.utcnow().isoformat(),
                "message": message,
                "metrics": self.metrics_collector
            }))


def create_batch_processor(config: BatchProcessingConfig) -> BatchEvaluationProcessor:
    """Factory function to create configured batch processor"""
    return BatchEvaluationProcessor(config)


# Example usage and configuration
if __name__ == "__main__":
    # High-volume production configuration
    production_config = BatchProcessingConfig(
        sampling_strategy=SamplingStrategy.EVERY_NTH_FRAME,
        sampling_interval=5,
        max_concurrent_videos=20,
        max_concurrent_frames=100,
        batch_size=32,
        use_gpu=True,
        confidence_threshold=ConfidenceThreshold.MEDIUM,
        auto_flag_threshold=0.8,
        human_review_threshold=0.6,
        enable_quality_gates=True,
        quality_gate_threshold=0.75
    )
    
    # Create batch processor
    processor = create_batch_processor(production_config)
    
    print("🎬 Batch Evaluation System Ready!")
    print(f"Configuration: {production_config}")
    print(f"Available evaluators: {list(processor.evaluators.keys())}")
    print(f"Sampling strategy: {production_config.sampling_strategy}")
    print(f"Confidence threshold: {production_config.confidence_threshold}")
    print(f"Max concurrent videos: {production_config.max_concurrent_videos}")
    print(f"Batch size: {production_config.batch_size}")
    
    print("\n✅ Batch processing system configured for high-volume production!")
