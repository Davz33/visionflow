#!/usr/bin/env python3
"""
Test the evaluation system with downloaded test videos.
Run this script to evaluate the downloaded test videos.
"""

import asyncio
from pathlib import Path
import sys

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).parent.parent))

from visionflow.services.evaluation.video_evaluation_orchestrator import (
    VideoEvaluationOrchestrator, SamplingStrategy
)
from visionflow.services.evaluation.confidence_manager import ConfidenceManager
from visionflow.services.evaluation.industry_metrics import IndustryMetrics

async def test_evaluation_with_videos():
    """Test the evaluation system with downloaded test videos"""
    
    # Initialize the evaluation system
    orchestrator = VideoEvaluationOrchestrator(
        sampling_strategy=SamplingStrategy.ADAPTIVE,
        max_frames=20
    )
    
    confidence_manager = ConfidenceManager()
    industry_metrics = IndustryMetrics()
    
    # Test videos directory
    test_videos_dir = Path("evaluation_datasets/test_videos")
    
    if not test_videos_dir.exists():
        print("❌ Test videos directory not found. Run download_test_videos.py first.")
        return
    
    # Load test dataset
    dataset_file = test_videos_dir / "test_dataset.json"
    if not dataset_file.exists():
        print("❌ Test dataset not found. Run download_test_videos.py first.")
        return
    
    with open(dataset_file, 'r') as f:
        dataset = json.load(f)
    
    print(f"🎬 Testing evaluation system with {len(dataset['videos'])} test videos")
    print("=" * 60)
    
    for video_info in dataset['videos']:
        video_path = test_videos_dir / video_info['filename']
        
        if not video_path.exists():
            print(f"⚠️  Video file not found: {video_path}")
            continue
        
        print(f"\n📹 Evaluating: {video_info['filename']}")
        print(f"   Category: {video_info['category']}")
        print(f"   Description: {video_info['description']}")
        
        try:
            # Create a test prompt based on the video category
            test_prompt = f"Test video: {video_info['description']}"
            
            # Run evaluation
            result = await orchestrator.evaluate_video(
                video_path=str(video_path),
                prompt=test_prompt
            )
            
            print(f"   ✅ Evaluation completed")
            print(f"   📊 Overall Score: {result.overall_score:.3f}")
            print(f"   🔍 Confidence: {result.overall_confidence:.3f}")
            print(f"   📋 Decision: {result.decision}")
            
        except Exception as e:
            print(f"   ❌ Evaluation failed: {e}")
    
    print("\n🎉 Test evaluation completed!")

if __name__ == "__main__":
    asyncio.run(test_evaluation_with_videos())
