#!/usr/bin/env python3
"""
RunPod Handler for WAN 2.1 Generation
This script runs inside RunPod containers and handles video generation requests
"""

import os
import sys
import json
import time
import asyncio
import logging
from pathlib import Path
from typing import Dict, Any, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add workspace to Python path
sys.path.append('/workspace')

# Global service instance (initialized once for faster subsequent calls)
wan_service = None

def init_wan_service():
    """Initialize WAN service once per container"""
    global wan_service
    
    if wan_service is not None:
        return wan_service
    
    try:
        logger.info("Initializing WAN 2.1 service...")
        
        # Import your WAN service
        from visionflow.services.generation.wan_video_service import WanVideoGenerationService
        from visionflow.shared.models import VideoGenerationRequest
        
        # Initialize the service
        wan_service = WanVideoGenerationService()
        
        logger.info("✅ WAN 2.1 service initialized successfully")
        return wan_service
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize WAN service: {e}")
        raise

def validate_request(job_input: Dict[str, Any]) -> Dict[str, Any]:
    """Validate and normalize request parameters"""
    
    # Required parameter
    if 'prompt' not in job_input:
        raise ValueError("Missing required parameter: 'prompt'")
    
    # Set defaults for optional parameters
    validated = {
        'prompt': str(job_input['prompt']),
        'duration': int(job_input.get('duration', 5)),
        'fps': int(job_input.get('fps', 24)),
        'resolution': str(job_input.get('resolution', '512x512')),
        'quality': str(job_input.get('quality', 'medium')),
        'guidance_scale': float(job_input.get('guidance_scale', 7.5)),
        'num_inference_steps': int(job_input.get('num_inference_steps', 25)),
        'seed': job_input.get('seed')  # Can be None
    }
    
    # Validate ranges
    if validated['duration'] < 1 or validated['duration'] > 30:
        raise ValueError("Duration must be between 1 and 30 seconds")
    
    if validated['fps'] < 8 or validated['fps'] > 60:
        raise ValueError("FPS must be between 8 and 60")
    
    if validated['guidance_scale'] < 1.0 or validated['guidance_scale'] > 20.0:
        raise ValueError("Guidance scale must be between 1.0 and 20.0")
    
    if validated['num_inference_steps'] < 10 or validated['num_inference_steps'] > 100:
        raise ValueError("Inference steps must be between 10 and 100")
    
    # Validate resolution format
    try:
        width, height = validated['resolution'].split('x')
        width, height = int(width), int(height)
        if width < 256 or height < 256 or width > 1024 or height > 1024:
            raise ValueError("Resolution must be between 256x256 and 1024x1024")
    except:
        raise ValueError("Invalid resolution format. Use 'WIDTHxHEIGHT' (e.g., '512x512')")
    
    return validated

async def generate_video_handler(job: Dict[str, Any]) -> Dict[str, Any]:
    """
    Main handler for RunPod video generation jobs
    
    Args:
        job: RunPod job object with 'input' containing generation parameters
        
    Returns:
        Result dictionary for RunPod
    """
    start_time = time.time()
    
    try:
        logger.info("🎬 Starting video generation job")
        
        # Extract and validate input
        job_input = job.get("input", {})
        validated_params = validate_request(job_input)
        
        logger.info(f"📝 Prompt: {validated_params['prompt'][:100]}...")
        logger.info(f"⚙️  Parameters: {validated_params['duration']}s, {validated_params['resolution']}, {validated_params['num_inference_steps']} steps")
        
        # Initialize service (cached after first call)
        service = init_wan_service()
        
        # Import here to avoid issues if import fails
        from visionflow.shared.models import VideoGenerationRequest
        
        # Create VideoGenerationRequest object
        request = VideoGenerationRequest(
            prompt=validated_params['prompt'],
            duration=validated_params['duration'],
            fps=validated_params['fps'],
            resolution=validated_params['resolution'],
            quality=validated_params['quality'],
            guidance_scale=validated_params['guidance_scale'],
            num_inference_steps=validated_params['num_inference_steps'],
            seed=validated_params['seed']
        )
        
        # Generate video using your exact WAN service
        logger.info("🚀 Starting video generation...")
        generation_start = time.time()
        
        result = await service.generate_video(request)
        
        generation_time = time.time() - generation_start
        total_time = time.time() - start_time
        
        logger.info(f"✅ Generation completed in {generation_time:.1f}s (total: {total_time:.1f}s)")
        
        # Format response for RunPod
        response = {
            "status": "success",
            "video_path": result.get("video_path"),
            "generation_time": generation_time,
            "total_time": total_time,
            "metadata": {
                "prompt": validated_params['prompt'],
                "resolution": validated_params['resolution'],
                "duration": validated_params['duration'],
                "fps": validated_params['fps'],
                "guidance_scale": validated_params['guidance_scale'],
                "num_inference_steps": validated_params['num_inference_steps'],
                "seed": validated_params['seed'],
                **result.get("metadata", {})
            }
        }
        
        # Add video file to response if it exists
        video_path = result.get("video_path")
        if video_path and Path(video_path).exists():
            response["video_size_mb"] = Path(video_path).stat().st_size / (1024 * 1024)
            
            # For RunPod, we might need to encode the video as base64 or upload to storage
            # For now, just return the path
            response["video_available"] = True
        else:
            response["video_available"] = False
            logger.warning("Video file not found or not generated")
        
        return response
        
    except Exception as e:
        error_time = time.time() - start_time
        error_msg = str(e)
        
        logger.error(f"❌ Generation failed after {error_time:.1f}s: {error_msg}")
        
        return {
            "status": "error",
            "error": error_msg,
            "error_type": type(e).__name__,
            "execution_time": error_time,
            "metadata": {
                "input": job.get("input", {}),
                "error_details": str(e)
            }
        }

def sync_handler(job: Dict[str, Any]) -> Dict[str, Any]:
    """
    Synchronous wrapper for async handler
    Required by RunPod serverless
    """
    try:
        # Run async handler in event loop
        return asyncio.run(generate_video_handler(job))
    except Exception as e:
        logger.error(f"Handler wrapper error: {e}")
        return {
            "status": "error",
            "error": f"Handler wrapper error: {str(e)}",
            "error_type": "HandlerError"
        }

def test_handler():
    """Test the handler locally"""
    logger.info("🧪 Testing handler locally...")
    
    test_job = {
        "input": {
            "prompt": "A red ball bouncing on grass, test video",
            "duration": 3,
            "fps": 24,
            "resolution": "512x512",
            "num_inference_steps": 15,
            "seed": 42
        }
    }
    
    result = sync_handler(test_job)
    logger.info(f"Test result: {json.dumps(result, indent=2)}")
    
    return result

# RunPod serverless entry point
if __name__ == "__main__":
    logger.info("🚀 Starting RunPod WAN 2.1 Handler")
    
    # Check if this is a test run
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        test_handler()
        sys.exit(0)
    
    try:
        import runpod
        
        logger.info("Starting RunPod serverless...")
        
        # Start RunPod serverless with our handler
        runpod.serverless.start({
            "handler": sync_handler,
            "return_aggregate_stream": True  # Enable streaming for large responses
        })
        
    except ImportError:
        logger.error("RunPod package not found. Install with: pip install runpod")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Failed to start RunPod serverless: {e}")
        sys.exit(1)
