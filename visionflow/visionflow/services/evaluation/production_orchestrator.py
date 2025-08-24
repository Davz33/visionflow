"""
Production Video Evaluation Orchestrator
Integrates all production models (LPIPS, CLIP, LLaVA, Motion) for comprehensive video evaluation.

This orchestrator provides enterprise-grade video evaluation with:
- Real model integration (not placeholders)
- Sophisticated score aggregation
- Production-ready confidence management
- Efficient resource utilization
"""

import asyncio
import time
import uuid
from dataclasses import dataclass
from enum import Enum
from typing import List, Dict, Any, Optional
import cv2
import numpy as np

from ...shared.monitoring import get_logger
from .production_models import ProductionEvaluationModels, get_production_models
from .confidence_manager import ConfidenceManager, ConfidenceLevel
from .score_aggregator import ScoreAggregator

logger = get_logger(__name__)

class EvaluationDimension(Enum):
    """Video evaluation dimensions"""
    VISUAL_QUALITY = "visual_quality"
    PERCEPTUAL_QUALITY = "perceptual_quality"  
    MOTION_CONSISTENCY = "motion_consistency"
    TEXT_VIDEO_ALIGNMENT = "text_video_alignment"
    AESTHETIC_QUALITY = "aesthetic_quality"
    NARRATIVE_FLOW = "narrative_flow"

class SamplingStrategy(Enum):
    """Frame sampling strategies for scalability"""
    EVERY_FRAME = "every_frame"
    EVERY_NTH_FRAME = "every_nth_frame"
    KEYFRAME_ONLY = "keyframe_only"
    ADAPTIVE = "adaptive"
    TEMPORAL_STRATIFIED = "temporal_stratified"
    RANDOM_SAMPLE = "random_sample"

@dataclass
class DimensionScore:
    """Score for a specific evaluation dimension"""
    dimension: EvaluationDimension
    score: float  # 0.0 to 1.0
    confidence: float  # 0.0 to 1.0
    details: Dict[str, Any]
    model_used: str
    processing_time: float

@dataclass
class VideoEvaluationResult:
    """Complete video evaluation result"""
    evaluation_id: str
    video_path: str
    prompt: str
    
    # Overall results
    overall_score: float
    overall_confidence: float
    confidence_level: ConfidenceLevel
    decision: str
    requires_human_review: bool
    
    # Dimension-wise results
    dimension_scores: List[DimensionScore]
    
    # Evaluation metadata
    sampling_strategy: SamplingStrategy
    frames_evaluated: int
    total_processing_time: float
    model_versions: Dict[str, str]
    
    # Aggregation details
    aggregation_method: str
    aggregation_details: Dict[str, Any]

class ProductionVideoEvaluationOrchestrator:
    """
    Production-grade video evaluation orchestrator with real models
    """
    
    def __init__(self, 
                 sampling_strategy: SamplingStrategy = SamplingStrategy.ADAPTIVE,
                 max_frames_per_video: int = 20,
                 device: str = 'auto',
                 enable_lightweight_mode: bool = False):
        
        self.sampling_strategy = sampling_strategy
        self.max_frames_per_video = max_frames_per_video
        self.device = device
        self.enable_lightweight_mode = enable_lightweight_mode
        
        # Initialize production models
        self.models = get_production_models(device=device, lightweight=enable_lightweight_mode)
        
        # Initialize confidence manager and score aggregator
        self.confidence_manager = ConfidenceManager()
        self.score_aggregator = ScoreAggregator()
        
        self.initialized = False
        
        logger.info(f"🏭 Production Video Evaluation Orchestrator initialized")
        logger.info(f"   Sampling Strategy: {sampling_strategy.value}")
        logger.info(f"   Max Frames: {max_frames_per_video}")
        logger.info(f"   Device: {device}")
        logger.info(f"   Lightweight Mode: {enable_lightweight_mode}")
    
    async def initialize(self):
        """Initialize all production models"""
        if not self.initialized:
            logger.info("🚀 Initializing production evaluation models...")
            await self.models.initialize()
            self.initialized = True
            logger.info("✅ Production orchestrator ready")
    
    def load_video_frames(self, video_path: str) -> List[np.ndarray]:
        """Load video frames from file"""
        frames = []
        
        try:
            cap = cv2.VideoCapture(video_path)
            
            if not cap.isOpened():
                logger.error(f"❌ Cannot open video: {video_path}")
                return frames
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frames.append(frame)
            
            cap.release()
            
            logger.debug(f"📹 Loaded {len(frames)} frames from {video_path}")
            
        except Exception as e:
            logger.error(f"❌ Error loading video frames: {e}")
        
        return frames
    
    def sample_frames(self, frames: List[np.ndarray], strategy: SamplingStrategy) -> List[np.ndarray]:
        """Sample frames according to the specified strategy"""
        
        if len(frames) <= self.max_frames_per_video:
            return frames
        
        if strategy == SamplingStrategy.EVERY_FRAME:
            return frames[:self.max_frames_per_video]
        
        elif strategy == SamplingStrategy.EVERY_NTH_FRAME:
            step = max(1, len(frames) // self.max_frames_per_video)
            return frames[::step][:self.max_frames_per_video]
        
        elif strategy == SamplingStrategy.KEYFRAME_ONLY:
            # Simple keyframe detection based on frame differences
            keyframes = [frames[0]]  # Always include first frame
            
            for i in range(1, len(frames)):
                if len(keyframes) >= self.max_frames_per_video:
                    break
                
                # Calculate difference from previous frame
                diff = cv2.absdiff(cv2.cvtColor(frames[i-1], cv2.COLOR_BGR2GRAY),
                                 cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY))
                
                if np.mean(diff) > 30:  # Threshold for keyframe detection
                    keyframes.append(frames[i])
            
            return keyframes
        
        elif strategy == SamplingStrategy.ADAPTIVE:
            # Adaptive sampling based on content complexity
            selected_frames = [frames[0]]  # Always include first
            
            complexity_scores = []
            for frame in frames:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                # Use Laplacian variance as complexity measure
                complexity = cv2.Laplacian(gray, cv2.CV_64F).var()
                complexity_scores.append(complexity)
            
            # Sort by complexity and select diverse frames
            frame_indices = list(range(len(frames)))
            sorted_indices = sorted(frame_indices, key=lambda i: complexity_scores[i], reverse=True)
            
            # Select frames with diverse time positions
            selected_indices = [0]  # First frame
            for idx in sorted_indices:
                if len(selected_indices) >= self.max_frames_per_video:
                    break
                
                # Ensure temporal diversity
                if all(abs(idx - selected_idx) > len(frames) // (self.max_frames_per_video * 2) 
                       for selected_idx in selected_indices):
                    selected_indices.append(idx)
            
            selected_indices.sort()
            return [frames[i] for i in selected_indices]
        
        elif strategy == SamplingStrategy.TEMPORAL_STRATIFIED:
            # Divide video into temporal segments and sample from each
            segment_size = len(frames) // self.max_frames_per_video
            sampled_frames = []
            
            for i in range(self.max_frames_per_video):
                segment_start = i * segment_size
                segment_end = min((i + 1) * segment_size, len(frames))
                
                if segment_start < len(frames):
                    # Sample middle frame from each segment
                    middle_idx = (segment_start + segment_end) // 2
                    sampled_frames.append(frames[middle_idx])
            
            return sampled_frames
        
        elif strategy == SamplingStrategy.RANDOM_SAMPLE:
            # Random sampling for statistical coverage
            import random
            indices = sorted(random.sample(range(len(frames)), self.max_frames_per_video))
            return [frames[i] for i in indices]
        
        else:
            # Default to uniform sampling
            step = len(frames) / self.max_frames_per_video
            indices = [int(i * step) for i in range(self.max_frames_per_video)]
            return [frames[i] for i in indices]
    
    async def evaluate_video(self, video_path: str, prompt: str) -> VideoEvaluationResult:
        """
        Perform comprehensive video evaluation using production models
        """
        evaluation_id = str(uuid.uuid4())
        start_time = time.time()
        
        logger.info(f"🎬 Starting production video evaluation: {evaluation_id}")
        logger.info(f"   Video: {video_path}")
        logger.info(f"   Prompt: {prompt}")
        
        if not self.initialized:
            await self.initialize()
        
        try:
            # Load video frames
            all_frames = self.load_video_frames(video_path)
            
            if not all_frames:
                raise ValueError(f"No frames loaded from video: {video_path}")
            
            # Sample frames according to strategy
            sampled_frames = self.sample_frames(all_frames, self.sampling_strategy)
            
            logger.info(f"🎬 Sampled {len(sampled_frames)} frames using {self.sampling_strategy.value}")
            
            # Run production model evaluations
            model_results = await self.models.evaluate_all_dimensions(sampled_frames, prompt)
            
            # Convert model results to dimension scores
            dimension_scores = await self._convert_model_results_to_scores(model_results)
            
            # Aggregate scores using production methods
            aggregation_result = await self.score_aggregator.aggregate_scores(dimension_scores)
            
            overall_score = aggregation_result['final_score']
            overall_confidence = aggregation_result['overall_confidence']
            
            # Determine confidence level and decision
            confidence_action = await self.confidence_manager.process_evaluation_confidence(
                overall_score, overall_confidence
            )
            
            total_time = time.time() - start_time
            
            # Create evaluation result
            result = VideoEvaluationResult(
                evaluation_id=evaluation_id,
                video_path=video_path,
                prompt=prompt,
                overall_score=overall_score,
                overall_confidence=overall_confidence,
                confidence_level=confidence_action.confidence_level,
                decision=confidence_action.action.value,
                requires_human_review=confidence_action.requires_review,
                dimension_scores=dimension_scores,
                sampling_strategy=self.sampling_strategy,
                frames_evaluated=len(sampled_frames),
                total_processing_time=total_time,
                model_versions=self._get_model_versions(),
                aggregation_method=aggregation_result['method_used'],
                aggregation_details=aggregation_result
            )
            
            logger.info(f"✅ Production evaluation completed: {evaluation_id}")
            logger.info(f"   Overall Score: {overall_score:.3f}")
            logger.info(f"   Confidence: {overall_confidence:.3f} ({confidence_action.confidence_level.value.upper()})")
            logger.info(f"   Decision: {confidence_action.action.value}")
            logger.info(f"   Time: {total_time:.2f}s")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Production evaluation failed: {e}")
            raise
    
    async def _convert_model_results_to_scores(self, model_results: Dict[str, Any]) -> List[DimensionScore]:
        """Convert production model results to standardized dimension scores"""
        
        dimension_scores = []
        
        # LPIPS -> Perceptual Quality
        if 'lpips' in model_results:
            lpips_result = model_results['lpips']
            score = DimensionScore(
                dimension=EvaluationDimension.PERCEPTUAL_QUALITY,
                score=lpips_result.get('consistency_score', 0.5),
                confidence=lpips_result.get('confidence', 0.8),
                details=lpips_result,
                model_used='LPIPS',
                processing_time=0.0  # Would be tracked in production
            )
            dimension_scores.append(score)
        
        # Motion Evaluator -> Motion Consistency
        if 'motion' in model_results:
            motion_result = model_results['motion']
            score = DimensionScore(
                dimension=EvaluationDimension.MOTION_CONSISTENCY,
                score=motion_result.get('motion_score', 0.5),
                confidence=motion_result.get('confidence', 0.7),
                details=motion_result,
                model_used='MotionEvaluator',
                processing_time=0.0
            )
            dimension_scores.append(score)
        
        # CLIP -> Text-Video Alignment
        if 'clip' in model_results:
            clip_result = model_results['clip']
            score = DimensionScore(
                dimension=EvaluationDimension.TEXT_VIDEO_ALIGNMENT,
                score=clip_result.get('alignment_score', 0.5),
                confidence=clip_result.get('confidence', 0.8),
                details=clip_result,
                model_used='CLIP',
                processing_time=0.0
            )
            dimension_scores.append(score)
        
        # LLaVA -> Aesthetic Quality and Narrative Flow
        if 'llava' in model_results:
            llava_results = model_results['llava']
            
            # Aesthetic Quality
            if 'aesthetic' in llava_results:
                aesthetic_result = llava_results['aesthetic']
                score = DimensionScore(
                    dimension=EvaluationDimension.AESTHETIC_QUALITY,
                    score=aesthetic_result.get('score', 0.5),
                    confidence=aesthetic_result.get('confidence', 0.7),
                    details=aesthetic_result,
                    model_used='LLaVA',
                    processing_time=0.0
                )
                dimension_scores.append(score)
            
            # Narrative Flow
            if 'narrative' in llava_results:
                narrative_result = llava_results['narrative']
                score = DimensionScore(
                    dimension=EvaluationDimension.NARRATIVE_FLOW,
                    score=narrative_result.get('score', 0.5),
                    confidence=narrative_result.get('confidence', 0.7),
                    details=narrative_result,
                    model_used='LLaVA',
                    processing_time=0.0
                )
                dimension_scores.append(score)
        
        # Add Visual Quality as a derived score from other metrics
        if dimension_scores:
            # Derive visual quality from perceptual quality and aesthetic quality
            perceptual_scores = [s for s in dimension_scores if s.dimension == EvaluationDimension.PERCEPTUAL_QUALITY]
            aesthetic_scores = [s for s in dimension_scores if s.dimension == EvaluationDimension.AESTHETIC_QUALITY]
            
            if perceptual_scores or aesthetic_scores:
                perceptual_score = perceptual_scores[0].score if perceptual_scores else 0.5
                aesthetic_score = aesthetic_scores[0].score if aesthetic_scores else 0.5
                
                visual_score = (perceptual_score * 0.6 + aesthetic_score * 0.4)
                visual_confidence = min(
                    perceptual_scores[0].confidence if perceptual_scores else 0.5,
                    aesthetic_scores[0].confidence if aesthetic_scores else 0.5
                )
                
                visual_dimension = DimensionScore(
                    dimension=EvaluationDimension.VISUAL_QUALITY,
                    score=visual_score,
                    confidence=visual_confidence,
                    details={'derived_from': ['perceptual_quality', 'aesthetic_quality']},
                    model_used='Derived',
                    processing_time=0.0
                )
                dimension_scores.append(visual_dimension)
        
        return dimension_scores
    
    def _get_model_versions(self) -> Dict[str, str]:
        """Get versions of production models"""
        return {
            'lpips': '0.1.4',
            'clip': '1.0.1', 
            'llava': 'v1.6-mistral-7b',
            'motion_evaluator': '1.0.0',
            'orchestrator': '1.0.0'
        }
    
    async def cleanup(self):
        """Cleanup all models and resources"""
        if self.models:
            self.models.cleanup()
        logger.info("🧹 Production orchestrator cleanup completed")

# Factory function for easy deployment
def get_production_orchestrator(
    sampling_strategy: SamplingStrategy = SamplingStrategy.ADAPTIVE,
    max_frames: int = 20,
    device: str = 'auto',
    lightweight: bool = False
) -> ProductionVideoEvaluationOrchestrator:
    """
    Get configured production evaluation orchestrator
    
    Args:
        sampling_strategy: Frame sampling strategy
        max_frames: Maximum frames per video
        device: Target device ('auto', 'cuda', 'mps', 'cpu')
        lightweight: Enable lightweight mode for resource-constrained environments
    """
    
    return ProductionVideoEvaluationOrchestrator(
        sampling_strategy=sampling_strategy,
        max_frames_per_video=max_frames,
        device=device,
        enable_lightweight_mode=lightweight
    )
