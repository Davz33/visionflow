#!/usr/bin/env python3
"""
Test script for high-quality 10-second WAN 2.1 video generation.
"""

import asyncio
import os
import sys
import time
from pathlib import Path

# Add the visionflow directory to Python path
visionflow_dir = Path(__file__).parent.parent
sys.path.insert(0, str(visionflow_dir))

from visionflow.services.generation.wan_video_service import WanVideoGenerationService
from visionflow.shared.models import VideoGenerationRequest, VideoQuality, VideoResolution

async def test_high_quality_generation():
    """Test high-quality 10-second video generation."""
    
    # Ensure we're in the correct working directory
    os.chdir(visionflow_dir)
    print(f"📁 Working directory: {os.getcwd()}")
    
    # Initialize the service
    print("🚀 Initializing WAN Video Generation Service...")
    service = WanVideoGenerationService()
    
    # Create a complex, high-quality request
    request = VideoGenerationRequest(
        prompt="A majestic dragon soaring through a mystical forest at sunset, with detailed scales, flowing wings, and magical particles floating around, cinematic lighting, high detail",
        duration=10,  # 10 seconds
        resolution=VideoResolution.HD_1080P,  # 1920x1080
        fps=24,  # Higher FPS for smooth motion
        quality=VideoQuality.ULTRA,  # Ultra quality
        guidance_scale=7.5,  # Balanced creativity vs adherence
        num_inference_steps=50  # More steps for higher quality
    )
    
    print(f"🎬 High-Quality Generation Request:")
    print(f"   📝 Prompt: {request.prompt}")
    print(f"   ⏱️  Duration: {request.duration}s")
    print(f"   🖼️  Resolution: {request.resolution}")
    print(f"   🎞️  FPS: {request.fps}")
    print(f"   ⭐ Quality: {request.quality}")
    print(f"   🎯 Guidance Scale: {request.guidance_scale}")
    print(f"   🔄 Inference Steps: {request.num_inference_steps}")
    
    try:
        print("\n🎬 Starting high-quality video generation...")
        start_time = time.time()
        
        # Generate the video
        result = await service.generate_video(request)
        
        end_time = time.time()
        generation_time = end_time - start_time
        
        if isinstance(result, dict) and result.get("status") == "completed":
            print(f"\n✅ High-quality generation successful!")
            print(f"⏱️  Total time: {generation_time:.2f} seconds")
            print(f"📁 Video saved to: {result.get('video_path', 'N/A')}")
            print(f"📊 Memory usage: {result.get('memory_usage', 'N/A')}")
            
            # Check if video file exists and get its size
            video_path = result.get('video_path')
            if video_path and os.path.exists(video_path):
                file_size = os.path.getsize(video_path) / (1024 * 1024)  # MB
                print(f"📏 Video file size: {file_size:.1f} MB")
        else:
            print(f"\n❌ High-quality generation failed:")
            print(f"   Error: {result.get('error', 'Unknown error')}")
            if isinstance(result, dict):
                print(f"   Result: {result}")
                
    except Exception as e:
        print(f"\n💥 Generation error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("🚀 Testing High-Quality WAN 2.1 Video Generation")
    print("=" * 60)
    
    asyncio.run(test_high_quality_generation())
