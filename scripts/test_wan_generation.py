#!/usr/bin/env python3
"""Test Wan2.2 video generation on RunPod"""

import asyncio
import os
import time
import traceback
from pathlib import Path
from visionflow.services.generation.wan_video_service import WanVideoGenerationService
from visionflow.shared.models import VideoGenerationRequest

async def test_generation():
    print("Testing Wan2.2 video generation...")
    
    # Ensure we're in the correct working directory
    current_dir = Path.cwd()
    visionflow_dir = current_dir.parent if current_dir.name == "scripts" else current_dir
    if visionflow_dir.name != "visionflow":
        visionflow_dir = current_dir / "visionflow" if (current_dir / "visionflow").exists() else current_dir
    
    print(f"📁 Working directory: {visionflow_dir}")
    os.chdir(visionflow_dir)
    
    # Initialize service
    service = WanVideoGenerationService()
    
    # Create test request
    request = VideoGenerationRequest(
        prompt="A cat playing with a ball of yarn, high quality",
        model_key="ti2v-5B",
        duration=1,
        resolution="1280x704",
        fps=24,
        quality="medium",
        guidance_scale=5.0,
        num_inference_steps=10,
        seed=42,
    )
    
    print(f"📝 Prompt: {request.prompt}")
    print(f"🎬 Duration: {request.duration}s, Resolution: {request.resolution}, FPS: {request.fps}")
    
    # Generate video
    start_time = time.time()
    try:
        result = await service.generate_video(request)
        end_time = time.time()
        
        if isinstance(result, dict) and result.get("status") == "completed":
            print(f"✅ Generation successful!")
            print(f"⏱️  Time taken: {end_time - start_time:.2f} seconds")
            print(f"📁 Video saved to: {result.get('video_path')}")
            print(f"📊 Memory usage: {result.get('memory_usage', 'N/A')}")
        else:
            print(f"❌ Video generation failed: {result}")
        
        return result
        
    except Exception as e:
        print(f"❌ Generation failed: {e}")
        traceback.print_exc()
        return None

if __name__ == "__main__":
    asyncio.run(test_generation())