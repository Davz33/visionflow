#!/usr/bin/env python3
"""
Download test videos for evaluation from reliable sources.
This script downloads short videos (10 seconds or less) for testing the evaluation system.
"""

import os
import json
import subprocess
from pathlib import Path
from typing import List, Dict, Any
import time

# Configuration
EVALUATION_DATASET_DIR = Path("evaluation_datasets")
TEST_VIDEOS_DIR = EVALUATION_DATASET_DIR / "test_videos"
TEST_VIDEOS_DIR.mkdir(exist_ok=True)

# Test video sources with reliable content
TEST_VIDEO_SOURCES = [
    {
        "name": "nature_landscape",
        "description": "A serene mountain landscape with flowing water",
        "url": "https://sample-videos.com/zip/10/mp4/SampleVideo_1280x720_1mb.mp4",
        "expected_duration": "10s",
        "category": "landscape",
        "evaluation_focus": ["spatial_consistency", "lighting_realism", "motion_smoothness"]
    },
    {
        "name": "urban_scene", 
        "description": "Urban cityscape with people and vehicles",
        "url": "https://sample-videos.com/zip/10/mp4/SampleVideo_1280x720_2mb.mp4",
        "expected_duration": "10s",
        "category": "urban",
        "evaluation_focus": ["motion_consistency", "object_tracking", "scene_complexity"]
    },
    {
        "name": "object_detail",
        "description": "Close-up of mechanical object with fine details",
        "url": "https://sample-videos.com/zip/10/mp4/SampleVideo_1280x720_5mb.mp4", 
        "expected_duration": "10s",
        "category": "object",
        "evaluation_focus": ["texture_detail", "spatial_stability", "lighting_consistency"]
    }
]

def download_video(url: str, output_path: Path) -> bool:
    """Download a video from URL using curl"""
    try:
        print(f"📥 Downloading: {output_path.name}")
        
        result = subprocess.run([
            "curl", "-L", "-o", str(output_path), url
        ], capture_output=True, text=True, timeout=300)
        
        if result.returncode == 0 and output_path.exists() and output_path.stat().st_size > 0:
            print(f"✅ Downloaded: {output_path.name}")
            return True
        else:
            print(f"❌ Failed to download: {url}")
            return False
                
    except Exception as e:
        print(f"❌ Error downloading {url}: {e}")
        return False

def create_test_video_dataset():
    """Create a test video dataset with metadata"""
    
    test_videos = []
    successful_downloads = 0
    
    for video_source in TEST_VIDEO_SOURCES:
        video_name = video_source["name"]
        video_file = TEST_VIDEOS_DIR / f"{video_name}.mp4"
        
        # Try to download the video
        if download_video(video_source["url"], video_file):
            successful_downloads += 1
            
            # Add to test videos list
            test_videos.append({
                "filename": video_file.name,
                "path": str(video_file.relative_to(Path.cwd())),
                "description": video_source["description"],
                "category": video_source["category"],
                "expected_duration": video_source["expected_duration"],
                "evaluation_focus": video_source["evaluation_focus"],
                "source": "sample-videos.com",
                "download_time": time.strftime("%Y-%m-%dT%H:%M:%S")
            })
        else:
            print(f"⚠️  Skipping {video_name} due to download failure")
    
    # Create the test dataset metadata
    test_dataset = {
        "metadata": {
            "name": "Test Video Evaluation Dataset",
            "description": "Short test videos for evaluating video generation quality assessment",
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "total_videos": len(test_videos),
            "successful_downloads": successful_downloads,
            "target_duration": "10 seconds or less",
            "purpose": "Testing evaluation system with real video content"
        },
        "videos": test_videos
    }
    
    # Save the test dataset metadata
    output_file = TEST_VIDEOS_DIR / "test_dataset.json"
    with open(output_file, 'w') as f:
        json.dump(test_dataset, f, indent=2)
    
    print(f"✅ Created test dataset: {output_file}")
    return test_dataset

def create_evaluation_prompts():
    """Create evaluation prompts for the test videos"""
    
    evaluation_prompts = {
        "nature_landscape": {
            "prompt": "A serene mountain landscape with flowing water, showing realistic natural lighting, atmospheric effects, and smooth motion. The scene should maintain spatial consistency and natural color grading throughout.",
            "evaluation_criteria": [
                "spatial_consistency", "lighting_realism", "motion_smoothness", "color_naturalness"
            ]
        },
        "urban_scene": {
            "prompt": "An urban cityscape with people and vehicles in motion, showing realistic urban dynamics, consistent object tracking, and natural movement patterns. The scene should maintain temporal coherence.",
            "evaluation_criteria": [
                "motion_consistency", "object_tracking", "scene_complexity", "temporal_coherence"
            ]
        },
        "object_detail": {
            "prompt": "A close-up view of a mechanical object with intricate details, showing consistent lighting, stable camera work, and clear visibility of fine features. The object should maintain spatial stability.",
            "evaluation_criteria": [
                "texture_detail", "spatial_stability", "lighting_consistency", "focus_stability"
            ]
        }
    }
    
    # Save evaluation prompts
    output_file = TEST_VIDEOS_DIR / "evaluation_prompts.json"
    with open(output_file, 'w') as f:
        json.dump(evaluation_prompts, f, indent=2)
    
    print(f"✅ Created evaluation prompts: {output_file}")
    return evaluation_prompts

def create_evaluation_test_script():
    """Create a script to test the evaluation system with the downloaded videos"""
    
    test_script_content = '''#!/usr/bin/env python3
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
        
        print(f"\\n📹 Evaluating: {video_info['filename']}")
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
    
    print("\\n🎉 Test evaluation completed!")

if __name__ == "__main__":
    asyncio.run(test_evaluation_with_videos())
'''
    
    output_file = TEST_VIDEOS_DIR / "test_evaluation.py"
    with open(output_file, 'w') as f:
        f.write(test_script_content)
    
    # Make it executable
    output_file.chmod(0o755)
    
    print(f"✅ Created evaluation test script: {output_file}")
    return output_file

def main():
    """Main function to download test videos and create evaluation setup"""
    print("🚀 Setting up test video evaluation dataset...")
    
    # Create test videos directory
    TEST_VIDEOS_DIR.mkdir(exist_ok=True)
    print(f"📁 Created test videos directory: {TEST_VIDEOS_DIR}")
    
    # Download test videos
    test_dataset = create_test_video_dataset()
    
    # Create evaluation prompts
    evaluation_prompts = create_evaluation_prompts()
    
    # Create evaluation test script
    test_script = create_evaluation_test_script()
    
    # Create a README for the test dataset
    readme_content = f"""# Test Video Evaluation Dataset

This directory contains test videos for evaluating the video generation quality assessment system.

## Dataset Information

- **Total Videos**: {len(test_dataset['videos'])}
- **Target Duration**: 10 seconds or less
- **Purpose**: Testing evaluation system with real video content
- **Source**: Sample videos from reliable sources

## Files

- `test_dataset.json` - Metadata for all test videos
- `evaluation_prompts.json` - Evaluation prompts for each video
- `test_evaluation.py` - Script to test the evaluation system

## Usage

1. **Download videos**: Run `download_test_videos.py` to download test videos
2. **Test evaluation**: Run `test_evaluation.py` to evaluate the videos
3. **Review results**: Check the evaluation output for quality assessment

## Video Categories

- **Landscape**: Natural scenes with environmental elements
- **Urban**: City scenes with people and vehicles  
- **Object**: Close-up views of detailed objects

## Evaluation Focus

Each video category has specific evaluation criteria:
- Spatial consistency
- Motion smoothness
- Lighting realism
- Object tracking
- Texture detail
- Temporal coherence

Run the test evaluation script to see how the system performs on these videos.
"""
    
    readme_file = TEST_VIDEOS_DIR / "README.md"
    with open(readme_file, 'w') as f:
        f.write(readme_content)
    
    print(f"✅ Created README: {readme_file}")
    print("\n🎉 Test video dataset setup completed!")
    print(f"📁 Test videos location: {TEST_VIDEOS_DIR.absolute()}")
    print(f"📊 Total videos: {len(test_dataset['videos'])}")
    print(f"📋 Evaluation prompts: {len(evaluation_prompts)}")
    print(f"🧪 Test script: {test_script}")
    print("\n💡 Next steps:")
    print("1. Run the test evaluation script: python test_videos/test_evaluation.py")
    print("2. Review evaluation results")
    print("3. Adjust evaluation criteria if needed")

if __name__ == "__main__":
    main()
