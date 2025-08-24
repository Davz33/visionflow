#!/usr/bin/env python3
"""
Create Test Dataset for Video Evaluation
Downloads or creates sample videos with known prompts for testing the evaluation system.

This script creates a test dataset of videos with their associated prompts,
following industry best practices for ML evaluation datasets.
"""

import asyncio
import json
import os
import requests
import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from visionflow.services.generation.video_metadata_tracker import (
    metadata_tracker, VideoGenerationMetadata
)
from visionflow.shared.models import VideoQuality

# Sample video-prompt pairs for testing (common text-to-video prompts)
SAMPLE_PROMPTS = [
    {
        "prompt": "A beautiful sunset over a calm ocean with gentle waves, cinematic lighting, peaceful atmosphere",
        "quality": "high",
        "duration": 3.0,
        "tags": ["nature", "sunset", "ocean", "cinematic"]
    },
    {
        "prompt": "A cat walking through a garden filled with colorful flowers, soft natural lighting",
        "quality": "medium", 
        "duration": 5.0,
        "tags": ["animals", "cat", "garden", "flowers"]
    },
    {
        "prompt": "Rain falling on a window with blurred city lights in the background, moody atmosphere",
        "quality": "high",
        "duration": 4.0,
        "tags": ["rain", "urban", "moody", "atmospheric"]
    },
    {
        "prompt": "A field of wheat swaying in the wind under a blue sky with white clouds",
        "quality": "medium",
        "duration": 6.0,
        "tags": ["nature", "wheat", "field", "sky"]
    },
    {
        "prompt": "Fire crackling in a fireplace, warm orange glow, cozy atmosphere",
        "quality": "high",
        "duration": 8.0,
        "tags": ["fire", "fireplace", "cozy", "warm"]
    },
    {
        "prompt": "Birds flying across a clear blue sky, peaceful and serene",
        "quality": "medium",
        "duration": 4.0,
        "tags": ["birds", "sky", "peaceful", "nature"]
    },
    {
        "prompt": "Abstract geometric shapes morphing and changing colors, artistic style",
        "quality": "low",
        "duration": 3.0,
        "tags": ["abstract", "geometric", "artistic", "colorful"]
    },
    {
        "prompt": "Snow falling gently in a forest during winter, quiet and serene atmosphere",
        "quality": "high",
        "duration": 5.0,
        "tags": ["snow", "winter", "forest", "serene"]
    }
]

# URLs to sample WAN-generated videos (if available)
SAMPLE_VIDEO_URLS = [
    # These would be actual URLs to WAN-generated sample videos
    # For now, we'll create placeholder entries
    {
        "url": None,  # Would be actual URL
        "filename": "sample_sunset_ocean.mp4",
        "prompt_index": 0
    },
    {
        "url": None,
        "filename": "sample_cat_garden.mp4", 
        "prompt_index": 1
    },
    {
        "url": None,
        "filename": "sample_rain_window.mp4",
        "prompt_index": 2
    }
]

async def create_dummy_videos():
    """
    Create dummy video files for testing when we don't have actual WAN samples
    These are minimal MP4 files just for testing the evaluation pipeline
    """
    
    print("🎬 Creating dummy test videos...")
    
    test_dir = project_root / "generated" / "test_dataset"
    test_dir.mkdir(parents=True, exist_ok=True)
    
    created_videos = []
    
    for i, prompt_data in enumerate(SAMPLE_PROMPTS[:4]):  # Create 4 test videos
        filename = f"test_video_{i+1:02d}.mp4"
        video_path = test_dir / filename
        
        # Create a minimal MP4 file (this is just for testing - not a real video)
        # In practice, you'd download actual WAN-generated samples
        try:
            # Create a tiny valid MP4 file using ffmpeg if available
            import subprocess
            
            # Create a simple test video with ffmpeg
            cmd = [
                "ffmpeg", "-y",  # -y to overwrite
                "-f", "lavfi",
                "-i", f"testsrc2=duration={prompt_data['duration']}:size=640x480:rate=12",
                "-c:v", "libx264",
                "-preset", "ultrafast",
                "-pix_fmt", "yuv420p",
                str(video_path)
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                print(f"   ✅ Created {filename} ({video_path.stat().st_size / 1024:.1f}KB)")
                created_videos.append((str(video_path), prompt_data))
            else:
                print(f"   ❌ Failed to create {filename}: {result.stderr}")
                
        except Exception as e:
            print(f"   ⚠️ ffmpeg not available, creating placeholder: {e}")
            
            # Create a minimal placeholder file
            with open(video_path, 'wb') as f:
                # Write minimal MP4 header (not a real video, just for testing)
                f.write(b'\x00\x00\x00\x1cftypisom\x00\x00\x02\x00isomiso2mp41')
                f.write(b'\x00' * 1000)  # Padding
            
            print(f"   📝 Created placeholder {filename}")
            created_videos.append((str(video_path), prompt_data))
    
    return created_videos

async def download_sample_videos():
    """Download actual WAN sample videos if URLs are available"""
    
    print("📥 Downloading sample WAN videos...")
    
    downloaded_videos = []
    
    for sample in SAMPLE_VIDEO_URLS:
        if sample["url"]:
            try:
                response = requests.get(sample["url"], stream=True)
                response.raise_for_status()
                
                video_path = project_root / "generated" / "test_dataset" / sample["filename"]
                video_path.parent.mkdir(parents=True, exist_ok=True)
                
                with open(video_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                
                prompt_data = SAMPLE_PROMPTS[sample["prompt_index"]]
                downloaded_videos.append((str(video_path), prompt_data))
                
                print(f"   ✅ Downloaded {sample['filename']}")
                
            except Exception as e:
                print(f"   ❌ Failed to download {sample['filename']}: {e}")
        else:
            print(f"   ⚠️ No URL available for {sample['filename']}")
    
    return downloaded_videos

async def create_metadata_for_test_videos(video_prompt_pairs: List[Tuple[str, Dict]]):
    """Create proper metadata for test videos"""
    
    print("📋 Creating metadata for test videos...")
    
    created_count = 0
    
    for video_path, prompt_data in video_prompt_pairs:
        try:
            # Create comprehensive metadata
            video_path_obj = Path(video_path)
            file_stats = video_path_obj.stat()
            
            generation_id = str(uuid.uuid4())
            video_id = str(uuid.uuid4())
            
            metadata = VideoGenerationMetadata(
                generation_id=generation_id,
                video_id=video_id,
                video_path=str(video_path_obj.absolute()),
                filename=video_path_obj.name,
                file_size_bytes=file_stats.st_size,
                file_created_at=datetime.fromtimestamp(file_stats.st_mtime),
                
                # Prompt and generation parameters
                prompt=prompt_data["prompt"],
                negative_prompt=None,
                duration=prompt_data["duration"],
                fps=12,  # Standard test FPS
                resolution="640x480",  # Standard test resolution
                quality=prompt_data["quality"],
                seed=42,  # Fixed seed for reproducibility
                guidance_scale=7.5,
                num_inference_steps=25,
                
                # Model information
                model_name="wan2.1-t2v-1.3b",
                model_version="test_samples",
                device="test",
                
                # Generation results
                actual_duration=prompt_data["duration"],
                actual_resolution="640x480",
                actual_fps=12.0,
                num_frames=int(prompt_data["duration"] * 12),
                generation_time=30.0,  # Simulated generation time
                
                # System metadata
                created_at=datetime.utcnow(),
                completed_at=datetime.utcnow(),
                user_id="test_user",
                session_id="test_session",
                
                # Tags and notes
                tags=["test_dataset", "evaluation_sample"] + prompt_data.get("tags", []),
                notes=f"Test video for evaluation system testing - {prompt_data['quality']} quality",
                metadata_version="1.0"
            )
            
            # Store metadata
            success_count = 0
            for backend in metadata_tracker.storage_backends:
                try:
                    if await backend.store_metadata(metadata):
                        success_count += 1
                except Exception as e:
                    print(f"      ⚠️ Backend failed: {e}")
            
            if success_count > 0:
                created_count += 1
                print(f"   ✅ Metadata created for {video_path_obj.name}")
            else:
                print(f"   ❌ Failed to store metadata for {video_path_obj.name}")
                
        except Exception as e:
            print(f"   ❌ Error creating metadata for {video_path}: {e}")
    
    print(f"📊 Created metadata for {created_count}/{len(video_prompt_pairs)} videos")
    return created_count

async def create_evaluation_dataset():
    """Create a complete evaluation dataset with videos and metadata"""
    
    print("🎯 CREATING EVALUATION TEST DATASET")
    print("=" * 60)
    print("This creates a test dataset for evaluating the autorater system")
    print()
    
    # Step 1: Try to download actual samples, fall back to dummy videos
    print("STEP 1: Acquiring test videos")
    print("-" * 30)
    
    downloaded_videos = await download_sample_videos()
    
    if not downloaded_videos:
        print("No sample videos available online, creating test videos...")
        test_videos = await create_dummy_videos()
    else:
        test_videos = downloaded_videos
    
    if not test_videos:
        print("❌ Failed to create any test videos!")
        return False
    
    print(f"✅ Acquired {len(test_videos)} test videos")
    print()
    
    # Step 2: Create metadata for all videos
    print("STEP 2: Creating metadata")
    print("-" * 30)
    
    metadata_count = await create_metadata_for_test_videos(test_videos)
    
    if metadata_count == 0:
        print("❌ Failed to create metadata!")
        return False
    
    print()
    
    # Step 3: Create dataset manifest
    print("STEP 3: Creating dataset manifest")
    print("-" * 30)
    
    manifest = {
        "dataset_name": "WAN Video Evaluation Test Dataset",
        "created_at": datetime.utcnow().isoformat(),
        "total_videos": len(test_videos),
        "metadata_coverage": metadata_count,
        "description": "Test dataset for evaluating WAN video generation autorater system",
        "videos": []
    }
    
    for video_path, prompt_data in test_videos:
        video_info = {
            "filename": Path(video_path).name,
            "path": video_path,
            "prompt": prompt_data["prompt"],
            "quality": prompt_data["quality"],
            "duration": prompt_data["duration"],
            "tags": prompt_data.get("tags", [])
        }
        manifest["videos"].append(video_info)
    
    # Save manifest
    manifest_path = project_root / "generated" / "test_dataset" / "dataset_manifest.json"
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    
    print(f"✅ Dataset manifest saved: {manifest_path}")
    print()
    
    # Step 4: Summary
    print("📊 DATASET SUMMARY")
    print("=" * 30)
    print(f"Videos created: {len(test_videos)}")
    print(f"Metadata records: {metadata_count}")
    print(f"Quality distribution:")
    
    quality_counts = {}
    for _, prompt_data in test_videos:
        quality = prompt_data["quality"]
        quality_counts[quality] = quality_counts.get(quality, 0) + 1
    
    for quality, count in quality_counts.items():
        print(f"  {quality}: {count} videos")
    
    print(f"\nDataset location: {project_root / 'generated' / 'test_dataset'}")
    print("✅ Test dataset ready for evaluation!")
    
    return True

async def verify_dataset():
    """Verify the created dataset"""
    
    print("\n🔍 VERIFYING DATASET")
    print("=" * 30)
    
    # Check if videos exist and have metadata
    discovered = await metadata_tracker.discover_existing_videos()
    test_videos = [v for v in discovered if "test_dataset" in v["video_path"]]
    
    print(f"Test videos found: {len(test_videos)}")
    
    for video_info in test_videos:
        filename = Path(video_info["video_path"]).name
        has_metadata = video_info["has_metadata"]
        status = "✅" if has_metadata else "❌"
        
        print(f"  {status} {filename}")
        
        if has_metadata:
            metadata = video_info["metadata"]
            print(f"     Prompt: {metadata.prompt[:50]}...")
            print(f"     Quality: {metadata.quality}")
    
    print()
    coverage = len([v for v in test_videos if v["has_metadata"]]) / len(test_videos) * 100 if test_videos else 0
    print(f"Metadata coverage: {coverage:.1f}%")
    
    if coverage >= 100:
        print("✅ Dataset verification passed!")
        return True
    else:
        print("⚠️ Dataset verification failed - missing metadata")
        return False

async def main():
    """Main function"""
    
    print("🎬 WAN VIDEO EVALUATION DATASET CREATOR")
    print("Creating test dataset for autorater evaluation system")
    print("=" * 60)
    print()
    
    try:
        # Create the dataset
        success = await create_evaluation_dataset()
        
        if success:
            # Verify the dataset
            await verify_dataset()
            
            print("\n🚀 NEXT STEPS:")
            print("1. Run evaluation test: python test_evaluation_with_metadata.py")
            print("2. Test autorater system with known prompts")
            print("3. Validate multi-dimensional scoring")
            print("4. Check confidence management decisions")
        else:
            print("\n❌ Dataset creation failed!")
            print("🔧 Troubleshooting:")
            print("1. Check if ffmpeg is installed for video creation")
            print("2. Verify write permissions in generated/ directory")
            print("3. Check system resources")
    
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())
