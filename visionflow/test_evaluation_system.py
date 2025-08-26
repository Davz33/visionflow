#!/usr/bin/env python3
"""
Test script for the Multi-dimensional Video Evaluation System
Demonstrates the complete autorater and autoeval workflow.
"""

import asyncio
import os
import sys
import time
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from visionflow.services.evaluation.video_evaluation_orchestrator import (
    VideoEvaluationOrchestrator, SamplingStrategy
)
from visionflow.services.evaluation.confidence_manager import ConfidenceManager
from visionflow.services.evaluation.industry_metrics import IndustryMetrics

async def test_evaluation_system():
    """Test the complete evaluation system with a generated video"""
    
    print("🎯 Multi-dimensional Video Evaluation System Test")
    print("=" * 70)
    print("Testing autoraters and autoevals workflow ")
    print()
    
    # Find a generated video to evaluate
    generated_dir = project_root / "generated"
    video_files = list(generated_dir.glob("*.mp4"))
    
    if not video_files:
        print("❌ No video files found in generated/ directory")
        print("   Please run video generation first")
        return
    
    # Use the most recent video
    video_file = max(video_files, key=lambda p: p.stat().st_mtime)
    print(f"📹 Evaluating video: {video_file.name}")
    print(f"   File size: {video_file.stat().st_size / (1024*1024):.1f}MB")
    
    # Test prompt (simulating the original generation prompt)
    test_prompt = "A beautiful sunset over a calm ocean with gentle waves, cinematic lighting, peaceful atmosphere"
    print(f"📝 Test prompt: {test_prompt}")
    print()
    
    # Initialize evaluation components
    print("🚀 Initializing evaluation system...")
    
    # 1. Video Evaluation Orchestrator (main coordinator)
    orchestrator = VideoEvaluationOrchestrator(
        sampling_strategy=SamplingStrategy.ADAPTIVE,
        max_frames_per_video=20  # Reduced for testing
    )
    
    # 2. Confidence Manager (decision making)
    confidence_manager = ConfidenceManager(
        monitoring_window_hours=1,  # Short window for testing
        low_confidence_threshold=0.3,
        high_automation_threshold=0.9
    )
    
    # 3. Industry Metrics (optional - for full objective evaluation)
    industry_metrics = IndustryMetrics()
    await industry_metrics.initialize_models()
    
    print("✅ System initialized")
    print()
    
    # Run evaluation
    print("🔍 Starting multi-dimensional evaluation...")
    print("   Evaluating 6 dimensions:")
    print("   1. Visual Quality (clarity, sharpness)")
    print("   2. Perceptual Quality (image similarity across frames)")
    print("   3. Motion Consistency (motion smoothness, flickering)")
    print("   4. Text-Video Alignment (semantic and factual match)")
    print("   5. Aesthetic Quality (factual and aesthetic consistency)")
    print("   6. Narrative Flow (consistent narrative flow)")
    print()
    
    start_time = time.time()
    
    try:
        # Main evaluation
        evaluation_result = await orchestrator.evaluate_video(
            video_path=str(video_file),
            prompt=test_prompt
        )
        
        evaluation_time = time.time() - start_time
        
        # Process through confidence manager
        confidence_action = await confidence_manager.process_evaluation(evaluation_result)
        
        # Display results
        print("📊 EVALUATION RESULTS")
        print("=" * 50)
        print(f"Evaluation ID: {evaluation_result.evaluation_id}")
        print(f"Overall Score: {evaluation_result.overall_score:.3f}")
        print(f"Overall Confidence: {evaluation_result.overall_confidence:.3f}")
        print(f"Confidence Level: {evaluation_result.confidence_level.value.upper()}")
        print(f"Decision: {evaluation_result.decision}")
        print(f"Requires Review: {evaluation_result.requires_human_review}")
        print(f"Review Priority: {evaluation_result.review_priority}")
        print()
        
        print("📈 DIMENSION SCORES")
        print("-" * 50)
        for dim_score in evaluation_result.dimension_scores:
            print(f"{dim_score.dimension.value:25}: {dim_score.score:.3f} (conf: {dim_score.confidence:.3f})")
        print()
        
        print("⚙️  EVALUATION METADATA")
        print("-" * 50)
        print(f"Sampling Strategy: {evaluation_result.sampling_strategy.value}")
        print(f"Frames Evaluated: {evaluation_result.frames_evaluated}")
        print(f"Evaluation Time: {evaluation_result.evaluation_time:.2f}s")
        print(f"Video Duration: {evaluation_result.metadata.get('video_duration', 0):.1f}s")
        print()
        
        print("🎯 CONFIDENCE MANAGEMENT")
        print("-" * 50)
        print(f"Action Type: {confidence_action.action_type.value}")
        print(f"Review Priority: {confidence_action.review_priority.value}")
        print(f"Reason: {confidence_action.reason}")
        print()
        
        # Get performance metrics
        performance_metrics = await confidence_manager.get_performance_metrics()
        print("📊 SYSTEM PERFORMANCE")
        print("-" * 50)
        print(f"Total Evaluations: {performance_metrics.total_evaluations}")
        print(f"Auto Approved: {performance_metrics.auto_approved}")
        print(f"Flagged for Review: {performance_metrics.flagged_for_review}")
        print(f"Low Confidence Rate: {performance_metrics.low_confidence_rate:.1%}")
        print(f"High Automation Rate: {performance_metrics.high_automation_rate:.1%}")
        print(f"Average Confidence: {performance_metrics.avg_confidence:.3f}")
        print()
        
        # Check fine-tuning triggers
        triggers = await confidence_manager.check_fine_tuning_triggers()
        print("🔧 CONTINUOUS LEARNING TRIGGERS")
        print("-" * 50)
        print(f"Low Confidence Trigger: {triggers['low_confidence_trigger']}")
        print(f"High Automation Trigger: {triggers['high_automation_trigger']}")
        print(f"Recommendations: {len(triggers['recommendations'])}")
        
        for i, rec in enumerate(triggers['recommendations'], 1):
            print(f"  {i}. {rec['type']} ({rec['priority']} priority)")
            print(f"     Reason: {rec['reason']}")
        print()
        
        # Final assessment
        print("🎉 EVALUATION SUMMARY")
        print("=" * 50)
        
        if evaluation_result.confidence_level.value in ["excellent", "high"]:
            status_emoji = "✅"
            status_text = "PASSED - High Quality Video"
        elif evaluation_result.confidence_level.value == "medium":
            status_emoji = "⚠️"
            status_text = "NEEDS REVIEW - Medium Quality"
        else:
            status_emoji = "❌"
            status_text = "FAILED - Low Quality"
        
        print(f"{status_emoji} {status_text}")
        print(f"🔍 Confidence: {evaluation_result.overall_confidence:.1%}")
        print(f"📋 Decision: {evaluation_result.decision.replace('_', ' ').title()}")
        print(f"⏱️  Processing Time: {evaluation_time:.2f}s")
        
        if evaluation_result.requires_human_review:
            print(f"👤 Human review required (Priority: {evaluation_result.review_priority})")
        else:
            print("🤖 Automated processing approved")
        
        print()
        print("✅ Multi-dimensional evaluation completed successfully!")
        
        return evaluation_result
        
    except Exception as e:
        print(f"❌ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return None

async def main():
    """Main test function"""
    print("🎬 Testing Multi-dimensional Video Evaluation System")
    print("Based on autoraters and autoevals principles")
    print()
    
    result = await test_evaluation_system()
    
    if result:
        print("\n💡 Next Steps:")
        print("1. Review evaluation results above")
        print("2. Integrate with video generation pipeline")
        print("3. Set up continuous monitoring dashboard")
        print("4. Configure human review workflows")
        print("5. Implement fine-tuning triggers")
    else:
        print("\n🔧 Troubleshooting:")
        print("1. Ensure video files exist in generated/ directory")
        print("2. Check system dependencies")
        print("3. Verify frame sampling is working")

if __name__ == "__main__":
    asyncio.run(main())
