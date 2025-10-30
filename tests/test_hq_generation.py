#!/usr/bin/env python3
"""
Test HQ video generation with new Apple Silicon resource restrictions
"""

import asyncio
import os
import sys
import time
import uuid
from pathlib import Path

# Set Mac-optimized environment variables BEFORE importing torch/ML libraries
print("🍎 Setting up Apple Silicon MPS optimizations...")

# Mac-specific optimizations
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'

# Apply new conservative resource limits for Apple Silicon
os.environ['WAN_MPS_MEMORY_FRACTION'] = '0.5'        # Conservative 50% for unified memory
os.environ['WAN_MAX_SYSTEM_RAM_GB'] = '8.0'          # 8GB limit
os.environ['WAN_RAM_WARNING_THRESHOLD'] = '75.0'     # Lower threshold for Mac
os.environ['WAN_MAX_VIDEO_DURATION'] = '10'          # Allow up to 10 seconds
os.environ['WAN_MAX_RESOLUTION_PIXELS'] = '921600'   # 1280x720 max
os.environ['WAN_ENABLE_AGGRESSIVE_CLEANUP'] = 'true'
os.environ['WAN_CLEANUP_INTERVAL'] = '1'

# Cache configuration
CACHE_DIR = '/Users/dav/coding/wan-open-eval/visionflow/cache/huggingface'
os.environ['HF_HOME'] = CACHE_DIR
os.environ['TRANSFORMERS_CACHE'] = CACHE_DIR
os.environ['HF_HUB_CACHE'] = CACHE_DIR

print(f"🔒 Applied resource limits:")
print(f"   MPS Memory Fraction: {os.environ['WAN_MPS_MEMORY_FRACTION']}")
print(f"   Max System RAM: {os.environ['WAN_MAX_SYSTEM_RAM_GB']}GB")
print(f"   RAM Warning Threshold: {os.environ['WAN_RAM_WARNING_THRESHOLD']}%")
print(f"   Cache Directory: {CACHE_DIR}")

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Import after environment setup
from visionflow.services.generation.wan_video_service import WanVideoGenerationService
from visionflow.shared.models import VideoGenerationRequest, VideoQuality

async def test_hq_generation():
    """Test high-quality video generation with resource restrictions."""
    
    print("\n🎬 Starting HQ Video Generation Test")
    print("=" * 60)
    
    # Initialize service with resource limits
    print("🍎 Initializing WAN Video Service with Apple Silicon optimizations...")
    video_service = WanVideoGenerationService()
    
    # Check model status before generation
    print("\n📊 Checking system status...")
    status = await video_service.get_model_status()
    print(f"Device: {status['device']}")
    print(f"Current model: {status.get('current_model', 'None loaded')}")
    print(f"Memory usage: {status['memory_usage']}")
    print(f"Resource limits: {status['resource_limits']}")
    
    # Create high-quality generation request
    job_id = str(uuid.uuid4())
    prompt = "A beautiful sunset over a calm ocean with gentle waves, cinematic lighting, peaceful atmosphere"
    
    hq_request = VideoGenerationRequest(
        prompt=prompt,
        duration=3,  # 3 seconds as requested
        fps=24,      # Standard fps
        resolution="832x480",  # Conservative resolution for HQ model
        seed=42,     # Fixed seed for reproducibility
        guidance_scale=7.5,    # Good quality guidance
        num_inference_steps=25, # Higher steps for HQ
        quality=VideoQuality.HIGH  # HQ quality as requested
    )
    
    print(f"\n🚀 Generation Parameters:")
    print(f"   Job ID: {job_id}")
    print(f"   Prompt: {prompt}")
    print(f"   Duration: {hq_request.duration}s")
    print(f"   Resolution: {hq_request.resolution}")
    print(f"   Quality: {hq_request.quality}")
    print(f"   FPS: {hq_request.fps}")
    print(f"   Inference Steps: {hq_request.num_inference_steps}")
    
    # Monitor system resources before generation
    import psutil
    memory_before = psutil.virtual_memory()
    print(f"\n📈 System resources before generation:")
    print(f"   RAM: {memory_before.used / (1024**3):.1f}GB / {memory_before.total / (1024**3):.1f}GB ({memory_before.percent:.1f}%)")
    
    # Start generation
    start_time = time.time()
    print(f"\n🔥 Starting HQ video generation at {time.strftime('%H:%M:%S')}...")
    
    try:
        result = await video_service.generate_video(hq_request)
        generation_time = time.time() - start_time
        
        print(f"\n✅ Generation completed in {generation_time:.1f}s")
        print(f"Status: {result['status']}")
        
        if result['status'] == 'completed':
            print(f"Video path: {result['video_path']}")
            print(f"Model used: {result['model_used']}")
            print(f"Final resolution: {result['resolution']}")
            print(f"Total frames: {result['num_frames']}")
            print(f"Generation count: {result['generation_count']}")
            
            # Check if file exists and get size
            video_path = Path(result['video_path'])
            if video_path.exists():
                file_size = video_path.stat().st_size / (1024**2)  # Size in MB
                print(f"File size: {file_size:.1f}MB")
            else:
                print("⚠️ Video file not found!")
                
        else:
            print(f"❌ Generation failed: {result.get('error', 'Unknown error')}")
            
        # Monitor system resources after generation
        memory_after = psutil.virtual_memory()
        print(f"\n📈 System resources after generation:")
        print(f"   RAM: {memory_after.used / (1024**3):.1f}GB / {memory_after.total / (1024**3):.1f}GB ({memory_after.percent:.1f}%)")
        print(f"   Memory change: {(memory_after.used - memory_before.used) / (1024**3):+.1f}GB")
        
        if 'memory_usage' in result:
            print(f"\n🔍 Final memory usage reported by service:")
            for key, value in result['memory_usage'].items():
                if isinstance(value, float):
                    if 'gb' in key.lower():
                        print(f"   {key}: {value:.1f}GB")
                    elif 'percent' in key.lower():
                        print(f"   {key}: {value:.1f}%")
                    else:
                        print(f"   {key}: {value}")
                else:
                    print(f"   {key}: {value}")
        
        return result
        
    except Exception as e:
        generation_time = time.time() - start_time
        print(f"\n❌ Generation failed after {generation_time:.1f}s")
        print(f"Error: {e}")
        
        # Still monitor memory after failure
        memory_after = psutil.virtual_memory()
        print(f"\n📈 System resources after failure:")
        print(f"   RAM: {memory_after.used / (1024**3):.1f}GB / {memory_after.total / (1024**3):.1f}GB ({memory_after.percent:.1f}%)")
        
        return {"status": "failed", "error": str(e)}

async def main():
    """Main test function."""
    print("🍎 Apple Silicon WAN Video Generation Test")
    print("Testing HQ quality, 3-second video with new resource restrictions")
    print("=" * 70)
    
    # Run the generation test
    result = await test_hq_generation()
    
    print(f"\n🎯 Test Summary:")
    print(f"Status: {result['status']}")
    if result['status'] == 'completed':
        print("✅ HQ video generation successful with resource restrictions!")
    else:
        print("❌ Generation failed - check resource limits or system capacity")
    
    print("\n💡 To adjust resource limits, modify environment variables:")
    print("   export WAN_MPS_MEMORY_FRACTION=0.4  # More conservative")
    print("   export WAN_MAX_SYSTEM_RAM_GB=6.0    # Lower RAM limit")
    print("   export WAN_MAX_VIDEO_DURATION=2     # Shorter videos")

if __name__ == "__main__":
    asyncio.run(main())
