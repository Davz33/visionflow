#!/usr/bin/env python3
"""
Test the complete evaluation system with proper metadata tracking.
Demonstrates industry best practices for video-prompt mapping.
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
from visionflow.services.generation.video_metadata_tracker import metadata_tracker

async def test_metadata_discovery():
    """Test metadata discovery and mapping functionality"""
    
    print("🔍 METADATA DISCOVERY AND MAPPING TEST")
    print("=" * 70)
    print("Testing industry best practices for video-prompt tracking")
    print()
    
    # Discover all existing videos and their metadata
    print("📁 Discovering existing videos...")
    discovered_videos = await metadata_tracker.discover_existing_videos()
    
    print(f"Found {len(discovered_videos)} video files:")
    print()
    
    videos_with_metadata = []
    videos_without_metadata = []
    
    for video_info in discovered_videos:
        filename = video_info['filename']
        has_metadata = video_info['has_metadata']
        file_size = video_info['file_size_mb']
        modified = video_info['modified_at'].strftime('%Y-%m-%d %H:%M:%S')
        
        status_emoji = "✅" if has_metadata else "❌"
        print(f"{status_emoji} {filename}")
        print(f"   Size: {file_size:.1f}MB | Modified: {modified}")
        
        if has_metadata:
            metadata = video_info['metadata']
            print(f"   📝 Prompt: {metadata.prompt[:60]}{'...' if len(metadata.prompt) > 60 else ''}")
            print(f"   🎯 Quality: {metadata.quality} | Duration: {metadata.duration}s")
            print(f"   🔧 Model: {metadata.model_name} | Device: {metadata.device}")
            videos_with_metadata.append(video_info)
        else:
            print(f"   ⚠️  No metadata found - cannot evaluate without prompt")
            videos_without_metadata.append(video_info)
        
        print()
    
    print("📊 METADATA SUMMARY")
    print("-" * 50)
    print(f"Videos with metadata: {len(videos_with_metadata)}")
    print(f"Videos without metadata: {len(videos_without_metadata)}")
    print(f"Metadata coverage: {len(videos_with_metadata)/len(discovered_videos)*100:.1f}%")
    print()
    
    return videos_with_metadata, videos_without_metadata

async def test_evaluation_with_metadata():
    """Test evaluation system using metadata-tracked videos"""
    
    print("🎯 EVALUATION WITH METADATA TRACKING")
    print("=" * 70)
    
    # Discover videos with metadata
    videos_with_metadata, videos_without_metadata = await test_metadata_discovery()
    
    if not videos_with_metadata:
        print("❌ No videos with metadata found!")
        print("💡 Generate a video first using the updated WAN service:")
        print("   python test_hq_generation.py")
        return
    
    # Initialize evaluation system
    print("🚀 Initializing evaluation system...")
    orchestrator = VideoEvaluationOrchestrator(
        sampling_strategy=SamplingStrategy.ADAPTIVE,
        max_frames_per_video=20
    )
    
    confidence_manager = ConfidenceManager(
        monitoring_window_hours=24,
        low_confidence_threshold=0.3,
        high_automation_threshold=0.9
    )
    
    print("✅ Evaluation system ready")
    print()
    
    # Evaluate each video with its tracked metadata
    evaluation_results = []
    
    for i, video_info in enumerate(videos_with_metadata[:3], 1):  # Limit to first 3 for demo
        metadata = video_info['metadata']
        video_path = video_info['video_path']
        
        print(f"📹 EVALUATING VIDEO {i}/{min(3, len(videos_with_metadata))}")
        print("=" * 50)
        print(f"File: {metadata.filename}")
        print(f"Prompt: {metadata.prompt}")
        print(f"Generation ID: {metadata.generation_id}")
        print(f"Quality Setting: {metadata.quality}")
        print(f"Duration: {metadata.duration}s | FPS: {metadata.fps}")
        print()
        
        start_time = time.time()
        
        try:
            # Run evaluation using the tracked prompt
            evaluation_result = await orchestrator.evaluate_video(
                video_path=video_path,
                prompt=metadata.prompt  # Using the tracked prompt!
            )
            
            # Process through confidence manager
            confidence_action = await confidence_manager.process_evaluation(evaluation_result)
            
            # Update metadata with evaluation results
            await metadata_tracker.update_evaluation_results(
                generation_id=metadata.generation_id,
                evaluation_result=evaluation_result
            )
            
            evaluation_time = time.time() - start_time
            
            # Display results
            print("📊 EVALUATION RESULTS")
            print("-" * 30)
            print(f"Overall Score: {evaluation_result.overall_score:.3f}")
            print(f"Confidence: {evaluation_result.overall_confidence:.3f}")
            print(f"Level: {evaluation_result.confidence_level.value.upper()}")
            print(f"Decision: {evaluation_result.decision}")
            print(f"Review Needed: {evaluation_result.requires_human_review}")
            print(f"Processing Time: {evaluation_time:.2f}s")
            
            # Show dimension breakdown
            print("\n📈 Dimension Scores:")
            for dim_score in evaluation_result.dimension_scores:
                score_bar = "█" * int(dim_score.score * 10) + "░" * (10 - int(dim_score.score * 10))
                print(f"  {dim_score.dimension.value.replace('_', ' ').title():20}: {dim_score.score:.3f} [{score_bar}]")
            
            evaluation_results.append({
                'metadata': metadata,
                'evaluation': evaluation_result,
                'confidence_action': confidence_action,
                'processing_time': evaluation_time
            })
            
            print(f"\n✅ Evaluation completed for {metadata.filename}")
            
        except Exception as e:
            print(f"❌ Evaluation failed for {metadata.filename}: {e}")
        
        print("\n" + "="*70 + "\n")
    
    # Summary analytics
    if evaluation_results:
        print("📊 BATCH EVALUATION SUMMARY")
        print("=" * 50)
        
        total_videos = len(evaluation_results)
        avg_score = sum(r['evaluation'].overall_score for r in evaluation_results) / total_videos
        avg_confidence = sum(r['evaluation'].overall_confidence for r in evaluation_results) / total_videos
        avg_processing_time = sum(r['processing_time'] for r in evaluation_results) / total_videos
        
        needs_review = sum(1 for r in evaluation_results if r['evaluation'].requires_human_review)
        auto_approved = total_videos - needs_review
        
        print(f"Videos Evaluated: {total_videos}")
        print(f"Average Score: {avg_score:.3f}")
        print(f"Average Confidence: {avg_confidence:.3f}")
        print(f"Average Processing Time: {avg_processing_time:.2f}s")
        print(f"Auto Approved: {auto_approved} ({auto_approved/total_videos*100:.1f}%)")
        print(f"Needs Review: {needs_review} ({needs_review/total_videos*100:.1f}%)")
        
        print("\n🎯 Quality Distribution:")
        confidence_levels = {}
        for result in evaluation_results:
            level = result['evaluation'].confidence_level.value
            confidence_levels[level] = confidence_levels.get(level, 0) + 1
        
        for level, count in confidence_levels.items():
            print(f"  {level.title()}: {count} videos ({count/total_videos*100:.1f}%)")
    
    print("\n✅ Metadata-driven evaluation completed!")
    
    return evaluation_results

async def test_metadata_workflow():
    """Test the complete metadata workflow"""
    
    print("🔄 COMPLETE METADATA WORKFLOW TEST")
    print("=" * 70)
    print("This demonstrates the full industry-standard workflow:")
    print("1. Video generation with metadata tracking")
    print("2. Metadata storage in multiple backends")
    print("3. Video discovery and prompt mapping")
    print("4. Evaluation using tracked prompts")
    print("5. Results linking back to generation metadata")
    print()
    
    # Step 1: Show current metadata state
    print("STEP 1: Current Metadata State")
    print("-" * 30)
    
    all_metadata = await metadata_tracker.storage_backends[0].list_all_metadata()
    print(f"Tracked generations: {len(all_metadata)}")
    
    if all_metadata:
        print("Recent generations:")
        for i, metadata in enumerate(all_metadata[:3], 1):
            print(f"  {i}. {metadata.filename}")
            print(f"     Prompt: {metadata.prompt[:50]}...")
            print(f"     Quality: {metadata.quality} | Model: {metadata.model_name}")
            if metadata.evaluation_id:
                print(f"     Evaluation: Score {metadata.overall_score:.3f} ({metadata.confidence_level})")
            else:
                print(f"     Evaluation: Not yet evaluated")
            print()
    
    # Step 2: Run evaluations
    print("STEP 2: Running Evaluations with Metadata")
    print("-" * 30)
    
    results = await test_evaluation_with_metadata()
    
    # Step 3: Show the complete lineage
    print("STEP 3: Generation-Evaluation Lineage")
    print("-" * 30)
    
    if results:
        for result in results:
            metadata = result['metadata']
            evaluation = result['evaluation']
            
            print(f"📹 {metadata.filename}")
            print(f"   Generation ID: {metadata.generation_id}")
            print(f"   Evaluation ID: {evaluation.evaluation_id}")
            print(f"   Created: {metadata.created_at.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"   Prompt: {metadata.prompt}")
            print(f"   Score: {evaluation.overall_score:.3f} | Confidence: {evaluation.overall_confidence:.3f}")
            print(f"   Decision: {evaluation.decision} | Review: {evaluation.requires_human_review}")
            print()
    
    print("✅ Complete workflow demonstrated!")
    print("\n💡 Key Benefits of this approach:")
    print("   ✅ Full traceability from prompt to evaluation")
    print("   ✅ Multiple storage backends for reliability")
    print("   ✅ Industry-standard metadata tracking")
    print("   ✅ Automatic evaluation-generation linking")
    print("   ✅ Support for analytics and debugging")
    print("   ✅ Compliance and audit trail support")

async def main():
    """Main test function"""
    
    print("🎬 VIDEO GENERATION METADATA TRACKING TEST")
    print("Based on industry best practices for ML model tracking")
    print("=" * 70)
    print()
    
    try:
        await test_metadata_workflow()
        
        print("\n🚀 NEXT STEPS FOR PRODUCTION:")
        print("1. Set up PostgreSQL for production database storage")
        print("2. Add user authentication and session tracking")
        print("3. Implement content versioning and model lineage")
        print("4. Set up monitoring dashboards for generation analytics")
        print("5. Add compliance features (content watermarking, audit logs)")
        print("6. Integrate with MLOps platforms (MLflow, Weights & Biases)")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())
