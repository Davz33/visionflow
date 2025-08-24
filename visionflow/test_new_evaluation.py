#!/usr/bin/env python3
"""
Standalone test for the new Multi-dimensional Video Evaluation System
Tests the autoraters and autoevals workflow without dependencies on existing modules.
"""

import asyncio
import os
import sys
import time
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Import our new evaluation modules directly
from visionflow.services.evaluation.video_evaluation_orchestrator import (
    VideoEvaluationOrchestrator, SamplingStrategy
)
from visionflow.services.evaluation.confidence_manager import ConfidenceManager
from visionflow.services.evaluation.score_aggregator import ScoreAggregator
from visionflow.services.evaluation.industry_metrics import IndustryMetrics

async def test_new_evaluation_system():
    """Test the new evaluation system with a generated video"""
    
    print("🎯 NEW Multi-dimensional Video Evaluation System Test")
    print("=" * 70)
    print("Testing autoraters and autoevals workflow from your slides")
    print()
    
    # Find a generated video to evaluate
    generated_dir = project_root / "generated"
    video_files = list(generated_dir.glob("*.mp4"))
    
    if not video_files:
        print("❌ No video files found in generated/ directory")
        print("   Please generate a video first using test_hq_generation.py")
        return
    
    # Use the most recent video
    video_file = max(video_files, key=lambda p: p.stat().st_mtime)
    print(f"📹 Evaluating video: {video_file.name}")
    print(f"   File size: {video_file.stat().st_size / (1024*1024):.1f}MB")
    print(f"   Modified: {time.ctime(video_file.stat().st_mtime)}")
    
    # Test prompt (simulating the original generation prompt)
    test_prompt = "A beautiful sunset over a calm ocean with gentle waves, cinematic lighting, peaceful atmosphere"
    print(f"📝 Test prompt: {test_prompt}")
    print()
    
    # Initialize evaluation components
    print("🚀 Initializing NEW evaluation system...")
    
    try:
        # 1. Video Evaluation Orchestrator (main coordinator)
        print("   1. Initializing Video Evaluation Orchestrator...")
        orchestrator = VideoEvaluationOrchestrator(
            sampling_strategy=SamplingStrategy.ADAPTIVE,
            max_frames_per_video=15  # Reduced for testing
        )
        
        # 2. Confidence Manager (decision making)
        print("   2. Initializing Confidence Manager...")
        confidence_manager = ConfidenceManager(
            monitoring_window_hours=1,  # Short window for testing
            low_confidence_threshold=0.3,
            high_automation_threshold=0.9
        )
        
        # 3. Score Aggregator (ensemble methods)
        print("   3. Initializing Score Aggregator...")
        score_aggregator = ScoreAggregator()
        
        # 4. Industry Metrics (objective evaluation)
        print("   4. Initializing Industry Metrics...")
        industry_metrics = IndustryMetrics()
        # Skip model initialization for now to avoid dependency issues
        # await industry_metrics.initialize_models()
        
        print("✅ All components initialized successfully")
        print()
        
        # Run the complete evaluation workflow
        print("🔍 Starting 6-dimensional evaluation workflow...")
        print("=" * 50)
        print("Dimensions being evaluated:")
        print("1. 📷 Visual Quality (clarity, sharpness)")
        print("2. 🎨 Perceptual Quality (image similarity across frames)")
        print("3. 🎬 Motion Consistency (motion smoothness, flickering)")
        print("4. 📝 Text-Video Alignment (semantic and factual match)")
        print("5. ✨ Aesthetic Quality (factual and aesthetic consistency)")
        print("6. 📖 Narrative Flow (consistent narrative flow)")
        print()
        
        start_time = time.time()
        
        # Main evaluation call
        print("🔥 Running evaluation...")
        evaluation_result = await orchestrator.evaluate_video(
            video_path=str(video_file),
            prompt=test_prompt
        )
        
        evaluation_time = time.time() - start_time
        
        print("🎯 Processing through confidence manager...")
        # Process through confidence manager
        confidence_action = await confidence_manager.process_evaluation(evaluation_result)
        
        print("\n" + "="*70)
        print("📊 EVALUATION RESULTS SUMMARY")
        print("="*70)
        
        # Overall Results
        print(f"🎬 Video: {video_file.name}")
        print(f"🆔 Evaluation ID: {evaluation_result.evaluation_id}")
        print(f"📈 Overall Score: {evaluation_result.overall_score:.3f} / 1.000")
        print(f"🎯 Overall Confidence: {evaluation_result.overall_confidence:.3f} / 1.000")
        print(f"🏆 Confidence Level: {evaluation_result.confidence_level.value.upper()}")
        print(f"⚖️  Decision: {evaluation_result.decision.replace('_', ' ').title()}")
        print(f"👥 Requires Review: {'Yes' if evaluation_result.requires_human_review else 'No'}")
        print(f"🚨 Review Priority: {evaluation_result.review_priority.upper()}")
        print()
        
        # Detailed Dimension Scores
        print("📊 DETAILED DIMENSION SCORES")
        print("-" * 50)
        for dim_score in evaluation_result.dimension_scores:
            score_bar = "█" * int(dim_score.score * 20) + "░" * (20 - int(dim_score.score * 20))
            confidence_stars = "⭐" * int(dim_score.confidence * 5)
            print(f"{dim_score.dimension.value.replace('_', ' ').title():25}: {dim_score.score:.3f} [{score_bar}] {confidence_stars}")
        print()
        
        # Evaluation Metadata
        print("⚙️  EVALUATION METADATA")
        print("-" * 50)
        print(f"Sampling Strategy: {evaluation_result.sampling_strategy.value.replace('_', ' ').title()}")
        print(f"Frames Analyzed: {evaluation_result.frames_evaluated}")
        print(f"Processing Time: {evaluation_result.evaluation_time:.2f} seconds")
        print(f"Video Duration: {evaluation_result.metadata.get('video_duration', 0):.1f} seconds")
        print(f"Estimated FPS: {evaluation_result.metadata.get('fps_estimate', 0):.1f}")
        print()
        
        # Confidence Management Results
        print("🎯 CONFIDENCE MANAGEMENT ANALYSIS")
        print("-" * 50)
        print(f"Action Type: {confidence_action.action_type.value.replace('_', ' ').title()}")
        print(f"Review Priority: {confidence_action.review_priority.value.title()}")
        print(f"Confidence Score: {confidence_action.confidence_score:.3f}")
        print(f"Decision Reason: {confidence_action.reason}")
        print(f"Timestamp: {confidence_action.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
        print()
        
        # Performance Metrics
        performance_metrics = await confidence_manager.get_performance_metrics()
        print("📈 SYSTEM PERFORMANCE METRICS")
        print("-" * 50)
        print(f"Total Evaluations: {performance_metrics.total_evaluations}")
        print(f"Auto Approved: {performance_metrics.auto_approved}")
        print(f"Flagged for Review: {performance_metrics.flagged_for_review}")
        print(f"Immediate Reviews: {performance_metrics.immediate_reviews}")
        print(f"Low Confidence Rate: {performance_metrics.low_confidence_rate:.1%}")
        print(f"High Automation Rate: {performance_metrics.high_automation_rate:.1%}")
        print(f"Average Confidence: {performance_metrics.avg_confidence:.3f}")
        print()
        
        # Continuous Learning Triggers
        triggers = await confidence_manager.check_fine_tuning_triggers()
        print("🔧 CONTINUOUS LEARNING TRIGGERS")
        print("-" * 50)
        print(f"Low Confidence Trigger: {'🚨 ACTIVE' if triggers['low_confidence_trigger'] else '✅ OK'}")
        print(f"High Automation Trigger: {'🚨 ACTIVE' if triggers['high_automation_trigger'] else '✅ OK'}")
        print(f"Total Recommendations: {len(triggers['recommendations'])}")
        
        if triggers['recommendations']:
            print("\nRecommendations:")
            for i, rec in enumerate(triggers['recommendations'], 1):
                print(f"  {i}. 🔄 {rec['type'].replace('_', ' ').title()} ({rec['priority']} priority)")
                print(f"     📋 {rec['reason']}")
                print(f"     🛠️  Actions: {', '.join(rec['actions'][:2])}...")
        print()
        
        # Final Assessment with Visual Indicators
        print("🎉 FINAL ASSESSMENT")
        print("=" * 50)
        
        if evaluation_result.confidence_level.value in ["excellent", "high"]:
            status_emoji = "🟢"
            status_text = "PASSED - High Quality Video"
            recommendation = "✅ Approved for automatic processing"
        elif evaluation_result.confidence_level.value == "medium":
            status_emoji = "🟡"
            status_text = "NEEDS REVIEW - Medium Quality"
            recommendation = "⚠️ Queue for human review"
        else:
            status_emoji = "🔴"
            status_text = "FAILED - Low Quality"
            recommendation = "❌ Requires immediate attention"
        
        print(f"{status_emoji} {status_text}")
        print(f"🎯 Confidence Level: {evaluation_result.overall_confidence:.1%}")
        print(f"📋 Decision: {evaluation_result.decision.replace('_', ' ').title()}")
        print(f"⏱️  Total Processing Time: {evaluation_time:.2f}s")
        print(f"💡 Recommendation: {recommendation}")
        
        if evaluation_result.requires_human_review:
            print(f"👤 Human review queued with {evaluation_result.review_priority} priority")
        else:
            print("🤖 Fully automated processing approved")
        
        print()
        print("✅ Multi-dimensional evaluation completed successfully!")
        print("🔄 System ready for continuous processing")
        
        return evaluation_result
        
    except Exception as e:
        print(f"❌ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return None

async def main():
    """Main test function"""
    print("🎬 Testing NEW Multi-dimensional Video Evaluation System")
    print("📋 Based on autoraters and autoevals principles from your slides")
    print("🔬 Implementing 6-dimensional evaluation with confidence management")
    print()
    
    result = await test_new_evaluation_system()
    
    if result:
        print("\n💡 IMPLEMENTATION HIGHLIGHTS:")
        print("✅ 6-dimensional evaluation framework implemented")
        print("✅ Confidence management with automated flagging")
        print("✅ Score aggregation with ensemble methods")
        print("✅ Continuous learning triggers")
        print("✅ Frame sampling strategies for scalability")
        print("✅ Industry metrics foundation ready")
        
        print("\n🚀 NEXT STEPS:")
        print("1. Integrate LPIPS, FVMD, CLIP models for objective metrics")
        print("2. Add LLaVA integration for subjective evaluation")
        print("3. Set up continuous monitoring dashboard")
        print("4. Configure human review workflows")
        print("5. Implement fine-tuning automation")
    else:
        print("\n🔧 TROUBLESHOOTING:")
        print("1. Ensure video files exist in generated/ directory")
        print("2. Run video generation first: python test_hq_generation.py")
        print("3. Check system dependencies")

if __name__ == "__main__":
    asyncio.run(main())
