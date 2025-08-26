"""
Video Evaluation Orchestrator - Multi-dimensional Autorater System
Based on autoraters and autoevals principles for text-to-video evaluation.

Implements the 6-dimensional evaluation framework:
1. Visual Quality (clarity, sharpness)
2. Perceptual Quality (image similarity across frames)
3. Motion Temporal Consistency (motion smoothness, flickering)
4. Text-Video Alignment (semantic and factual match)
5. Factual and Aesthetic Quality Consistency
6. Consistent Narrative Flow (of frames)
"""

import asyncio
import gc
import time
import uuid
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field

import cv2
import numpy as np
import torch
from PIL import Image

from ...shared.monitoring import get_logger

logger = get_logger(__name__)

class ConfidenceLevel(str, Enum):
    """Confidence levels for evaluation results"""
    EXCELLENT = "excellent"  # >0.9 - auto_approve
    HIGH = "high"            # 0.8-0.9 - auto_approve or flag_monitoring  
    MEDIUM = "medium"        # 0.6-0.8 - flag_review
    LOW = "low"              # 0.4-0.6 - queue_review
    CRITICAL = "critical"    # <0.4 - immediate_review

class SamplingStrategy(str, Enum):
    """Frame sampling strategies for scalable evaluation"""
    EVERY_FRAME = "every_frame"                    # Full evaluation (highest quality, slowest)
    EVERY_NTH_FRAME = "every_nth_frame"           # Evaluate every 5th frame (configurable)
    KEYFRAME_ONLY = "keyframe_only"               # Evaluate only keyframes (fastest)
    ADAPTIVE = "adaptive"                          # Intelligent sampling based on content complexity
    TEMPORAL_STRATIFIED = "temporal_stratified"    # Even distribution across video timeline
    RANDOM_SAMPLE = "random_sample"               # Statistical sampling for large datasets

class EvaluationDimension(str, Enum):
    """6 key evaluation dimensions for video generation"""
    VISUAL_QUALITY = "visual_quality"                    # clarity, sharpness
    PERCEPTUAL_QUALITY = "perceptual_quality"           # image similarity across frames
    MOTION_CONSISTENCY = "motion_consistency"           # motion smoothness, flickering
    TEXT_VIDEO_ALIGNMENT = "text_video_alignment"       # semantic and factual match
    AESTHETIC_QUALITY = "aesthetic_quality"             # factual and aesthetic consistency
    NARRATIVE_FLOW = "narrative_flow"                   # consistent narrative flow

@dataclass
class FrameSample:
    """Represents a sampled frame for evaluation"""
    frame_index: int
    frame_data: np.ndarray
    timestamp: float
    is_keyframe: bool = False
    sampling_confidence: float = 1.0

@dataclass
class DimensionScore:
    """Score for a single evaluation dimension"""
    dimension: EvaluationDimension
    score: float  # 0.0 to 1.0
    confidence: float  # 0.0 to 1.0
    method: str  # evaluation method used
    details: Dict[str, Any] = field(default_factory=dict)

@dataclass  
class EvaluationResult:
    """Complete evaluation result for a video"""
    video_path: str
    prompt: str
    evaluation_id: str
    timestamp: datetime
    
    # Individual dimension scores
    dimension_scores: List[DimensionScore]
    
    # Aggregated scores
    overall_score: float
    overall_confidence: float
    confidence_level: ConfidenceLevel
    
    # Metadata
    sampling_strategy: SamplingStrategy
    frames_evaluated: int
    evaluation_time: float
    
    # Decision and actions
    decision: str  # auto_approve, flag_review, etc.
    requires_human_review: bool
    review_priority: str
    
    # Additional details
    metadata: Dict[str, Any] = field(default_factory=dict)

class FrameSampler:
    """Intelligent frame sampling for different evaluation strategies"""
    
    def __init__(self, strategy: SamplingStrategy = SamplingStrategy.ADAPTIVE):
        self.strategy = strategy
        
    async def sample_frames(self, video_path: str, max_frames: int = 50) -> List[FrameSample]:
        """Sample frames from video based on strategy"""
        logger.info(f"🎬 Sampling frames using {self.strategy} strategy")
        
        # Load video
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0
        
        logger.info(f"Video info: {total_frames} frames, {fps:.1f} FPS, {duration:.1f}s")
        
        if self.strategy == SamplingStrategy.EVERY_FRAME:
            frames = await self._sample_every_frame(cap, total_frames, max_frames)
        elif self.strategy == SamplingStrategy.EVERY_NTH_FRAME:
            frames = await self._sample_every_nth_frame(cap, total_frames, max_frames)
        elif self.strategy == SamplingStrategy.KEYFRAME_ONLY:
            frames = await self._sample_keyframes(cap, total_frames, max_frames)
        elif self.strategy == SamplingStrategy.ADAPTIVE:
            frames = await self._sample_adaptive(cap, total_frames, max_frames, fps)
        elif self.strategy == SamplingStrategy.TEMPORAL_STRATIFIED:
            frames = await self._sample_temporal_stratified(cap, total_frames, max_frames, duration)
        elif self.strategy == SamplingStrategy.RANDOM_SAMPLE:
            frames = await self._sample_random(cap, total_frames, max_frames)
        else:
            frames = await self._sample_adaptive(cap, total_frames, max_frames, fps)
            
        cap.release()
        logger.info(f"✅ Sampled {len(frames)} frames using {self.strategy}")
        return frames
    
    async def _sample_every_frame(self, cap, total_frames: int, max_frames: int) -> List[FrameSample]:
        """Sample every frame (highest quality, slowest)"""
        frames = []
        frame_interval = max(1, total_frames // max_frames)
        
        for i in range(0, total_frames, frame_interval):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if ret:
                timestamp = i / cap.get(cv2.CAP_PROP_FPS)
                frames.append(FrameSample(i, frame, timestamp, False, 1.0))
                
        return frames[:max_frames]
    
    async def _sample_every_nth_frame(self, cap, total_frames: int, max_frames: int, n: int = 5) -> List[FrameSample]:
        """Sample every Nth frame (configurable interval)"""
        frames = []
        
        for i in range(0, total_frames, n):
            if len(frames) >= max_frames:
                break
                
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if ret:
                timestamp = i / cap.get(cv2.CAP_PROP_FPS)
                frames.append(FrameSample(i, frame, timestamp, False, 1.0))
                
        return frames
    
    async def _sample_keyframes(self, cap, total_frames: int, max_frames: int) -> List[FrameSample]:
        """Sample keyframes only (fastest approach)"""
        # Simple keyframe detection based on frame difference
        frames = []
        prev_frame = None
        keyframe_threshold = 0.3  # Threshold for frame difference
        
        for i in range(0, total_frames, max(1, total_frames // (max_frames * 2))):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if not ret:
                continue
                
            # Convert to grayscale for comparison
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            is_keyframe = False
            if prev_frame is None:
                is_keyframe = True  # First frame is always a keyframe
            else:
                # Calculate frame difference
                diff = cv2.absdiff(prev_frame, gray_frame)
                diff_ratio = np.sum(diff > 30) / diff.size
                is_keyframe = diff_ratio > keyframe_threshold
            
            if is_keyframe:
                timestamp = i / cap.get(cv2.CAP_PROP_FPS)
                frames.append(FrameSample(i, frame, timestamp, True, 1.0))
                
            prev_frame = gray_frame
            
            if len(frames) >= max_frames:
                break
                
        return frames
    
    async def _sample_adaptive(self, cap, total_frames: int, max_frames: int, fps: float) -> List[FrameSample]:
        """Intelligent sampling based on content complexity"""
        frames = []
        
        # Start with temporal stratification
        segment_size = total_frames // min(max_frames, 10)
        
        for segment in range(min(max_frames, 10)):
            start_frame = segment * segment_size
            end_frame = min((segment + 1) * segment_size, total_frames)
            
            # Sample middle frame of each segment for now
            # In a full implementation, this would analyze content complexity
            mid_frame = (start_frame + end_frame) // 2
            
            cap.set(cv2.CAP_PROP_POS_FRAMES, mid_frame)
            ret, frame = cap.read()
            if ret:
                timestamp = mid_frame / fps
                # Simple complexity score based on edge detection
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                edges = cv2.Canny(gray, 50, 150)
                complexity = np.sum(edges > 0) / edges.size
                
                frames.append(FrameSample(mid_frame, frame, timestamp, False, complexity))
                
        return frames
    
    async def _sample_temporal_stratified(self, cap, total_frames: int, max_frames: int, duration: float) -> List[FrameSample]:
        """Even distribution across video timeline"""
        frames = []
        time_intervals = np.linspace(0, duration, max_frames)
        
        for i, target_time in enumerate(time_intervals):
            frame_index = int(target_time * cap.get(cv2.CAP_PROP_FPS))
            frame_index = min(frame_index, total_frames - 1)
            
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ret, frame = cap.read()
            if ret:
                frames.append(FrameSample(frame_index, frame, target_time, False, 1.0))
                
        return frames
    
    async def _sample_random(self, cap, total_frames: int, max_frames: int) -> List[FrameSample]:
        """Statistical random sampling"""
        frames = []
        
        # Generate random frame indices
        frame_indices = np.random.choice(total_frames, min(max_frames, total_frames), replace=False)
        frame_indices = sorted(frame_indices)
        
        for frame_index in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ret, frame = cap.read()
            if ret:
                timestamp = frame_index / cap.get(cv2.CAP_PROP_FPS)
                frames.append(FrameSample(frame_index, frame, timestamp, False, 1.0))
                
        return frames

class VideoEvaluationOrchestrator:
    """
    Main orchestrator for multi-dimensional video evaluation
    Implements the autorater and autoeval framework
    """
    
    def __init__(self, 
                 sampling_strategy: SamplingStrategy = SamplingStrategy.ADAPTIVE,
                 max_frames_per_video: int = 50):
        self.sampling_strategy = sampling_strategy
        self.max_frames_per_video = max_frames_per_video
        self.frame_sampler = FrameSampler(sampling_strategy)
        
        # Evaluation weights (: weighted reliability)
        self.dimension_weights = {
            EvaluationDimension.VISUAL_QUALITY: 0.20,
            EvaluationDimension.PERCEPTUAL_QUALITY: 0.18,
            EvaluationDimension.MOTION_CONSISTENCY: 0.17,
            EvaluationDimension.TEXT_VIDEO_ALIGNMENT: 0.16,
            EvaluationDimension.AESTHETIC_QUALITY: 0.15,
            EvaluationDimension.NARRATIVE_FLOW: 0.14
        }
        
        # Confidence thresholds 
        self.confidence_thresholds = {
            ConfidenceLevel.EXCELLENT: 0.9,
            ConfidenceLevel.HIGH: 0.8,
            ConfidenceLevel.MEDIUM: 0.6,
            ConfidenceLevel.LOW: 0.4,
            ConfidenceLevel.CRITICAL: 0.0
        }
        
        logger.info(f"🎯 Video Evaluation Orchestrator initialized")
        logger.info(f"   Sampling Strategy: {sampling_strategy}")
        logger.info(f"   Max Frames: {max_frames_per_video}")
        
    async def evaluate_video(self, 
                           video_path: str, 
                           prompt: str,
                           evaluation_id: Optional[str] = None) -> EvaluationResult:
        """
        Main evaluation entry point - evaluates video across all 6 dimensions
        """
        if evaluation_id is None:
            evaluation_id = str(uuid.uuid4())
            
        logger.info(f"🎬 Starting video evaluation: {evaluation_id}")
        logger.info(f"   Video: {video_path}")
        logger.info(f"   Prompt: {prompt[:100]}{'...' if len(prompt) > 100 else ''}")
        
        start_time = time.time()
        
        try:
            # Step 1: Sample frames from video
            frames = await self.frame_sampler.sample_frames(video_path, self.max_frames_per_video)
            
            if not frames:
                raise ValueError("No frames could be sampled from video")
            
            # Step 2: Evaluate each dimension
            dimension_scores = []
            
            # Evaluate each dimension in parallel for efficiency
            dimension_tasks = []
            for dimension in EvaluationDimension:
                task = self._evaluate_dimension(dimension, frames, prompt, video_path)
                dimension_tasks.append(task)
            
            dimension_results = await asyncio.gather(*dimension_tasks)
            dimension_scores.extend(dimension_results)
            
            # Step 3: Aggregate scores and determine confidence
            overall_score, overall_confidence = self._aggregate_scores(dimension_scores)
            confidence_level = self._determine_confidence_level(overall_confidence)
            
            # Step 4: Make decision based on confidence
            decision, requires_review, priority = self._make_decision(confidence_level, overall_confidence)
            
            evaluation_time = time.time() - start_time
            
            # Step 5: Create evaluation result
            result = EvaluationResult(
                video_path=video_path,
                prompt=prompt,
                evaluation_id=evaluation_id,
                timestamp=datetime.utcnow(),
                dimension_scores=dimension_scores,
                overall_score=overall_score,
                overall_confidence=overall_confidence,
                confidence_level=confidence_level,
                sampling_strategy=self.sampling_strategy,
                frames_evaluated=len(frames),
                evaluation_time=evaluation_time,
                decision=decision,
                requires_human_review=requires_review,
                review_priority=priority,
                metadata={
                    "video_duration": frames[-1].timestamp if frames else 0,
                    "fps_estimate": len(frames) / frames[-1].timestamp if frames and frames[-1].timestamp > 0 else 0
                }
            )
            
            logger.info(f"✅ Evaluation completed: {evaluation_id}")
            logger.info(f"   Overall Score: {overall_score:.3f}")
            logger.info(f"   Confidence: {overall_confidence:.3f} ({confidence_level})")
            logger.info(f"   Decision: {decision}")
            logger.info(f"   Time: {evaluation_time:.2f}s")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Evaluation failed: {evaluation_id}: {e}")
            raise
        finally:
            # Cleanup
            gc.collect()
    
    async def _evaluate_dimension(self, 
                                dimension: EvaluationDimension, 
                                frames: List[FrameSample], 
                                prompt: str,
                                video_path: str) -> DimensionScore:
        """Evaluate a specific dimension using appropriate method"""
        logger.debug(f"🔍 Evaluating {dimension}")
        
        # For now, implement placeholder scoring
        # In full implementation, this would use:
        # - LPIPS for perceptual quality
        # - FVMD for motion consistency  
        # - CLIP for text-video alignment
        # - LLaVA for aesthetic quality and narrative flow
        # - Custom metrics for visual quality
        
        if dimension == EvaluationDimension.VISUAL_QUALITY:
            score, confidence = await self._evaluate_visual_quality(frames)
        elif dimension == EvaluationDimension.PERCEPTUAL_QUALITY:
            score, confidence = await self._evaluate_perceptual_quality(frames)
        elif dimension == EvaluationDimension.MOTION_CONSISTENCY:
            score, confidence = await self._evaluate_motion_consistency(frames)
        elif dimension == EvaluationDimension.TEXT_VIDEO_ALIGNMENT:
            score, confidence = await self._evaluate_text_alignment(frames, prompt)
        elif dimension == EvaluationDimension.AESTHETIC_QUALITY:
            score, confidence = await self._evaluate_aesthetic_quality(frames, prompt)
        elif dimension == EvaluationDimension.NARRATIVE_FLOW:
            score, confidence = await self._evaluate_narrative_flow(frames, prompt)
        else:
            score, confidence = 0.5, 0.5  # Default fallback
            
        return DimensionScore(
            dimension=dimension,
            score=score,
            confidence=confidence,
            method="placeholder",  # Will be specific method names
            details={"frames_analyzed": len(frames)}
        )
    
    async def _evaluate_visual_quality(self, frames: List[FrameSample]) -> Tuple[float, float]:
        """Evaluate visual quality (clarity, sharpness)"""
        # Placeholder implementation using simple sharpness metric
        sharpness_scores = []
        
        for frame in frames:
            gray = cv2.cvtColor(frame.frame_data, cv2.COLOR_BGR2GRAY)
            # Laplacian variance as sharpness measure
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            # Normalize to 0-1 scale (empirical values)
            normalized_score = min(1.0, laplacian_var / 1000.0)
            sharpness_scores.append(normalized_score)
        
        avg_score = np.mean(sharpness_scores)
        confidence = 1.0 - np.std(sharpness_scores)  # Higher consistency = higher confidence
        
        return avg_score, max(0.0, min(1.0, confidence))
    
    async def _evaluate_perceptual_quality(self, frames: List[FrameSample]) -> Tuple[float, float]:
        """Evaluate perceptual quality (image similarity across frames)"""
        # Placeholder: measure frame-to-frame similarity
        if len(frames) < 2:
            return 0.5, 0.5
            
        similarities = []
        for i in range(1, len(frames)):
            # Simple correlation coefficient between consecutive frames
            frame1 = cv2.cvtColor(frames[i-1].frame_data, cv2.COLOR_BGR2GRAY).flatten()
            frame2 = cv2.cvtColor(frames[i].frame_data, cv2.COLOR_BGR2GRAY).flatten()
            
            correlation = np.corrcoef(frame1, frame2)[0, 1]
            if not np.isnan(correlation):
                similarities.append(abs(correlation))
        
        if similarities:
            avg_similarity = np.mean(similarities)
            confidence = 1.0 - np.std(similarities)
            return avg_similarity, max(0.0, min(1.0, confidence))
        
        return 0.5, 0.5
    
    async def _evaluate_motion_consistency(self, frames: List[FrameSample]) -> Tuple[float, float]:
        """Evaluate motion temporal consistency using improved OpenCV optical flow"""
        if len(frames) < 2:
            return 0.5, 0.5
            
        try:
            flow_magnitudes = []
            for i in range(1, len(frames)):
                gray1 = cv2.cvtColor(frames[i-1].frame_data, cv2.COLOR_BGR2GRAY)
                gray2 = cv2.cvtColor(frames[i].frame_data, cv2.COLOR_BGR2GRAY)
                
                # First detect good features to track
                corners1 = cv2.goodFeaturesToTrack(
                    gray1, 
                    maxCorners=100, 
                    qualityLevel=0.01, 
                    minDistance=10,
                    blockSize=3
                )
                
                if corners1 is not None and len(corners1) > 0:
                    # Track features using Lucas-Kanade optical flow
                    corners2, status, error = cv2.calcOpticalFlowPyrLK(
                        gray1, gray2, corners1, None,
                        winSize=(15, 15),
                        maxLevel=2,
                        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
                    )
                    
                    # Filter successful tracks
                    if corners2 is not None and status is not None:
                        good_new = corners2[status == 1]
                        good_old = corners1[status == 1]
                        
                        if len(good_new) > 0 and len(good_old) > 0:
                            # Calculate motion vectors and magnitudes
                            motion_vectors = good_new - good_old
                            magnitudes = np.sqrt(motion_vectors[:, 0]**2 + motion_vectors[:, 1]**2)
                            avg_magnitude = np.mean(magnitudes)
                            flow_magnitudes.append(avg_magnitude)
                        else:
                            # No successful tracks, use frame difference as fallback
                            diff = cv2.absdiff(gray1, gray2)
                            avg_diff = np.mean(diff) / 255.0 * 10  # Scale to match flow magnitude range
                            flow_magnitudes.append(avg_diff)
                    else:
                        # Optical flow failed, use frame difference
                        diff = cv2.absdiff(gray1, gray2)
                        avg_diff = np.mean(diff) / 255.0 * 10
                        flow_magnitudes.append(avg_diff)
                else:
                    # No features detected, use frame difference
                    diff = cv2.absdiff(gray1, gray2)
                    avg_diff = np.mean(diff) / 255.0 * 10
                    flow_magnitudes.append(avg_diff)
            
            if flow_magnitudes and len(flow_magnitudes) > 1:
                # Consistency measured as inverse of variance (normalized)
                variance = np.var(flow_magnitudes)
                mean_magnitude = np.mean(flow_magnitudes)
                
                # Normalize consistency score
                if mean_magnitude > 0:
                    consistency = 1.0 / (1.0 + variance / (mean_magnitude + 1e-6))
                else:
                    consistency = 1.0  # No motion = perfect consistency
                    
                confidence = min(1.0, len(flow_magnitudes) / len(frames))
                return min(1.0, max(0.0, consistency)), confidence
            
        except Exception as e:
            logger.warning(f"Motion consistency evaluation failed, using fallback: {e}")
        
        # Fallback: simple frame difference
        try:
            differences = []
            for i in range(1, len(frames)):
                gray1 = cv2.cvtColor(frames[i-1].frame_data, cv2.COLOR_BGR2GRAY)
                gray2 = cv2.cvtColor(frames[i].frame_data, cv2.COLOR_BGR2GRAY)
                
                # Calculate frame difference
                diff = cv2.absdiff(gray1, gray2)
                diff_ratio = np.mean(diff) / 255.0
                differences.append(diff_ratio)
            
            if differences:
                # Consistency is inverse of difference variance
                consistency = 1.0 - min(1.0, np.var(differences))
                confidence = 0.6  # Lower confidence for fallback
                return consistency, confidence
        except Exception:
            pass
        
        return 0.5, 0.5
    
    async def _evaluate_text_alignment(self, frames: List[FrameSample], prompt: str) -> Tuple[float, float]:
        """Evaluate text-video alignment"""
        # Placeholder: will use CLIP in full implementation
        # For now, return moderate score with lower confidence due to complexity
        return 0.6, 0.7
    
    async def _evaluate_aesthetic_quality(self, frames: List[FrameSample], prompt: str) -> Tuple[float, float]:
        """Evaluate aesthetic quality"""
        # Placeholder: will use LLaVA in full implementation
        return 0.65, 0.75
    
    async def _evaluate_narrative_flow(self, frames: List[FrameSample], prompt: str) -> Tuple[float, float]:
        """Evaluate narrative flow consistency"""
        # Placeholder: will use LLaVA in full implementation
        return 0.6, 0.7
    
    def _aggregate_scores(self, dimension_scores: List[DimensionScore]) -> Tuple[float, float]:
        """
        Aggregate dimension scores using weighted reliability method
        (: weighted reliability aggregation)
        """
        if not dimension_scores:
            return 0.0, 0.0
        
        weighted_score = 0.0
        weighted_confidence = 0.0
        total_weight = 0.0
        
        for dim_score in dimension_scores:
            weight = self.dimension_weights.get(dim_score.dimension, 1.0)
            weighted_score += dim_score.score * weight * dim_score.confidence
            weighted_confidence += dim_score.confidence * weight
            total_weight += weight
        
        if total_weight > 0:
            final_score = weighted_score / total_weight
            final_confidence = weighted_confidence / total_weight
        else:
            final_score = 0.0
            final_confidence = 0.0
        
        return final_score, final_confidence
    
    def _determine_confidence_level(self, confidence: float) -> ConfidenceLevel:
        """Determine confidence level based on confidence score"""
        if confidence >= self.confidence_thresholds[ConfidenceLevel.EXCELLENT]:
            return ConfidenceLevel.EXCELLENT
        elif confidence >= self.confidence_thresholds[ConfidenceLevel.HIGH]:
            return ConfidenceLevel.HIGH
        elif confidence >= self.confidence_thresholds[ConfidenceLevel.MEDIUM]:
            return ConfidenceLevel.MEDIUM
        elif confidence >= self.confidence_thresholds[ConfidenceLevel.LOW]:
            return ConfidenceLevel.LOW
        else:
            return ConfidenceLevel.CRITICAL
    
    def _make_decision(self, confidence_level: ConfidenceLevel, confidence: float) -> Tuple[str, bool, str]:
        """
        Make decision based on confidence level 
        Returns: (decision, requires_human_review, priority)
        """
        if confidence_level == ConfidenceLevel.EXCELLENT:
            return "auto_approve", False, "none"
        elif confidence_level == ConfidenceLevel.HIGH:
            # Conditional approval based on exact confidence
            if confidence >= 0.85:
                return "auto_approve", False, "monitor"
            else:
                return "flag_monitoring", False, "low"
        elif confidence_level == ConfidenceLevel.MEDIUM:
            return "flag_review", True, "medium"
        elif confidence_level == ConfidenceLevel.LOW:
            return "queue_review", True, "high"
        else:  # CRITICAL
            return "immediate_review", True, "critical"
