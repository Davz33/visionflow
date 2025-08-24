"""
Example: LLaVA Hybrid Video Evaluation

This example demonstrates how to use LLaVA for subjective evaluation
combined with industry-standard metrics for objective evaluation.
"""

import asyncio
import logging
from pathlib import Path
from typing import Dict, Any

# Import our evaluation components
from .llava_analyzer import create_llava_analyzer
from .industry_metrics_implementation import create_industry_metrics_evaluator
from .quality_metrics import EvaluationResult, QualityDimensions
from .llm_config import LLMConfig

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def evaluate_video_with_llava_hybrid(
    video_path: str,
    original_prompt: str,
    sampling_strategy: str = "EVERY_5TH_FRAME"
) -> EvaluationResult:
    """
    Evaluate video using LLaVA for subjective analysis and industry metrics for objective analysis.
    
    Args:
        video_path: Path to the video file
        original_prompt: Original text prompt used for video generation
        sampling_strategy: Frame sampling strategy
        
    Returns:
        Comprehensive evaluation result combining subjective and objective metrics
    """
    
    logger.info(f"Starting hybrid evaluation of {video_path}")
    
    # Initialize LLaVA analyzer for subjective evaluation
    llava_analyzer = create_llava_analyzer()
    
    # Initialize industry metrics evaluator for objective evaluation
    industry_evaluator = create_industry_metrics_evaluator()
    
    try:
        # 1. Subjective Evaluation with LLaVA
        logger.info("Running LLaVA subjective analysis...")
        
        # Extract frames based on sampling strategy
        frames = await _extract_frames(video_path, sampling_strategy)
        
        # Analyze each frame with LLaVA
        subjective_scores = []
        for i, frame in enumerate(frames):
            frame_analysis = await llava_analyzer.analyze_frame_content(
                frame, i, original_prompt
            )
            subjective_scores.append(frame_analysis)
        
        # Analyze sequence coherence
        sequence_analysis = await llava_analyzer.analyze_video_sequence(
            frames, original_prompt
        )
        
        # 2. Objective Evaluation with Industry Metrics
        logger.info("Running industry metrics objective analysis...")
        
        objective_scores = await industry_evaluator.evaluate_video(
            video_path, original_prompt
        )
        
        # 3. Combine Results
        logger.info("Combining subjective and objective results...")
        
        combined_result = await _combine_evaluation_results(
            subjective_scores,
            sequence_analysis,
            objective_scores,
            original_prompt
        )
        
        return combined_result
        
    finally:
        # Clean up resources
        await llava_analyzer.close()
    
async def _extract_frames(video_path: str, strategy: str) -> list:
    """Extract frames from video based on sampling strategy."""
    import cv2
    
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    
    if strategy == "EVERY_5TH_FRAME":
        # Sample every 5th frame
        for i in range(0, frame_count, 5):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if ret:
                frames.append(frame)
    
    elif strategy == "KEYFRAME_ONLY":
        # Sample keyframes (simplified - every 10th frame)
        for i in range(0, frame_count, 10):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if ret:
                frames.append(frame)
    
    else:
        # Default: sample every frame (for short videos)
        for i in range(min(frame_count, 30)):  # Limit to 30 frames
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if ret:
                frames.append(frame)
    
    cap.release()
    logger.info(f"Extracted {len(frames)} frames using strategy: {strategy}")
    return frames

async def _combine_evaluation_results(
    subjective_scores: list,
    sequence_analysis: Dict[str, Any],
    objective_scores: Dict[str, Any],
    original_prompt: str
) -> EvaluationResult:
    """Combine subjective and objective evaluation results."""
    
    # Calculate average subjective scores
    if subjective_scores:
        avg_prompt_adherence = sum(s.get("prompt_adherence_score", 0.5) for s in subjective_scores) / len(subjective_scores)
        avg_visual_quality = sum(s.get("visual_quality_score", 0.5) for s in subjective_scores) / len(subjective_scores)
        avg_aesthetic = sum(s.get("artistic_appeal", 0.5) for s in subjective_scores) / len(subjective_scores)
    else:
        avg_prompt_adherence = avg_visual_quality = avg_aesthetic = 0.5
    
    # Extract sequence scores
    temporal_coherence = sequence_analysis.get("temporal_coherence_score", 0.5)
    motion_smoothness = sequence_analysis.get("motion_smoothness", 0.5)
    
    # Extract objective scores
    lpips_score = objective_scores.get("lpips_score", 0.5)
    fvmd_score = objective_scores.get("fvmd_score", 0.5)
    clip_score = objective_scores.get("clip_alignment_score", 0.5)
    
    # Create quality dimensions
    technical_metrics = QualityDimensions.TechnicalMetrics(
        resolution_quality=0.8,  # Default
        framerate_consistency=0.8,  # Default
        encoding_quality=0.8,  # Default
        overall_technical_score=0.8  # Will be updated
    )
    
    content_metrics = QualityDimensions.ContentMetrics(
        prompt_adherence=avg_prompt_adherence,
        visual_coherence=temporal_coherence,
        narrative_flow=motion_smoothness,
        creativity_score=avg_aesthetic,
        detail_richness=avg_visual_quality,
        scene_composition=avg_aesthetic,
        object_accuracy=0.8,  # Default
        character_consistency=0.8,  # Default
        scene_transitions=0.8  # Default
    )
    
    aesthetic_metrics = QualityDimensions.AestheticMetrics(
        visual_appeal=avg_aesthetic,
        color_harmony=avg_visual_quality,
        lighting_quality=avg_visual_quality,
        composition_quality=avg_aesthetic,
        style_consistency=temporal_coherence
    )
    
    # Create evaluation result
    result = EvaluationResult(
        video_path=Path("example_video.mp4"),
        original_prompt=original_prompt,
        evaluation_timestamp="2024-01-01T00:00:00Z",
        quality_dimensions=QualityDimensions(
            technical=technical_metrics,
            content=content_metrics,
            aesthetic=aesthetic_metrics,
            user_experience=QualityDimensions.UserExperienceMetrics(),
            performance=QualityDimensions.PerformanceMetrics(),
            compliance=QualityDimensions.ComplianceMetrics()
        ),
        confidence_score=0.85,  # High confidence due to hybrid approach
        evaluation_metadata={
            "evaluation_method": "LLaVA Hybrid",
            "subjective_analyzer": "LLaVA Local Model",
            "objective_metrics": "LPIPS, FVMD, CLIP",
            "frame_sampling": "Adaptive",
            "total_frames_analyzed": len(subjective_scores)
        }
    )
    
    return result

async def main():
    """Main example function."""
    
    # Check current LLM configuration
    LLMConfig.print_config()
    
    # Example usage
    video_path = "path/to/your/video.mp4"
    original_prompt = "A serene mountain landscape at sunset with flowing clouds"
    
    try:
        result = await evaluate_video_with_llava_hybrid(
            video_path=video_path,
            original_prompt=original_prompt,
            sampling_strategy="EVERY_5TH_FRAME"
        )
        
        print("\n=== LLaVA Hybrid Evaluation Results ===")
        print(f"Overall Quality Score: {result.overall_quality_score:.2f}")
        print(f"Confidence Score: {result.confidence_score:.2f}")
        print(f"Content Quality: {result.quality_dimensions.content.overall_content_score:.2f}")
        print(f"Technical Quality: {result.quality_dimensions.technical.overall_technical_score:.2f}")
        print(f"Aesthetic Quality: {result.quality_dimensions.aesthetic.overall_aesthetic_score:.2f}")
        
        # Print human-readable summary
        print("\n=== Human-Readable Summary ===")
        print(result.to_human_readable())
        
    except FileNotFoundError:
        print(f"Video file not found: {video_path}")
        print("Please update the video_path variable with a valid video file.")
    
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        print(f"Evaluation failed: {e}")

if __name__ == "__main__":
    asyncio.run(main())
