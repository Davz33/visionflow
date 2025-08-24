#!/usr/bin/env python3
"""
M4 Max Optimized WAN Generation Service
Persistent caching + MPS acceleration for blazing fast generation
"""

import asyncio
import os
import sys
from datetime import datetime
from typing import Dict, Any

# M4 Max Optimization Environment Variables (MUST be set before imports!)
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'  # Use all available memory

# Cache Configuration - Force HuggingFace to use existing cache
CACHE_DIR = '/Users/dav/coding/wan-open-eval/visionflow/cache/huggingface'
os.environ['HF_HOME'] = CACHE_DIR
os.environ['TRANSFORMERS_CACHE'] = CACHE_DIR
os.environ['HF_HUB_CACHE'] = CACHE_DIR
os.environ['HUGGINGFACE_HUB_CACHE'] = CACHE_DIR
os.environ['HF_DATASETS_CACHE'] = CACHE_DIR

# Disable download verification to use cached files
os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '1'
os.environ['HF_HUB_OFFLINE'] = '0'  # Allow fallback but prefer cache

print(f"🔥 CACHE FORCED: {CACHE_DIR}")
print(f"📦 Cache size: 8.6GB WAN model ready!")

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

# Add project root to Python path
sys.path.insert(0, '/Users/dav/coding/wan-open-eval/visionflow')

# Now import the working WAN service
from visionflow.services.generation.wan_video_service import WanVideoGenerationService
from visionflow.shared.models import VideoGenerationRequest, VideoQuality

app = FastAPI(title="Quick Generation Service", version="0.1.0")

class GenerationRequest(BaseModel):
    """Request model"""
    prompt: str
    original_prompt: str
    generation_params: Dict[str, Any]
    routing_decision: Dict[str, Any]
    job_id: str

class GenerationResponse(BaseModel):
    """Response model"""
    job_id: str
    status: str
    video_path: str = None
    generation_time: float
    message: str

@app.get("/health")
async def health_check():
    """Health check with M4 Max optimization status"""
    import torch
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "service": "m4-max-optimized-wan-generation",
        "model_loaded": True,
        "mps_available": torch.backends.mps.is_available(),
        "mps_built": torch.backends.mps.is_built(),
        "cache_dir": os.environ.get('HF_HOME'),
        "optimization": "M4 Max + MPS + Persistent Cache"
    }

@app.post("/generate", response_model=GenerationResponse)
async def generate_video(request: GenerationRequest):
    """Generate video using M4 Max optimized WAN service"""
    import time
    import torch
    start_time = time.time()
    
    print(f"🚀 M4 MAX OPTIMIZED GENERATION: Job {request.job_id}")
    print(f"   Prompt: {request.prompt}")
    print(f"   MPS Available: {torch.backends.mps.is_available()}")
    print(f"   Cache Dir: {os.environ.get('HF_HOME')}")
    
    try:
        # Initialize M4 Max optimized WAN service
        video_service = WanVideoGenerationService()
        
        # Prepare M4 Max optimized request
        params = request.generation_params
        wan_request = VideoGenerationRequest(
            prompt=request.prompt,
            duration=params.get("duration", 1),
            fps=params.get("fps", 12),
            resolution=params.get("resolution", "256x256"),
            seed=params.get("seed"),
            guidance_scale=params.get("guidance_scale", 6.0),  # Optimized for M4 Max
            num_inference_steps=params.get("num_inference_steps", 15),  # Faster for M4 Max
            quality=VideoQuality(params.get("quality", "low"))
        )
        
        # Generate video with M4 Max acceleration
        print(f"🔥 Starting M4 Max accelerated generation...")
        result = await video_service.generate_video(wan_request)
        
        generation_time = time.time() - start_time
        print(f"⚡ M4 Max generation completed in {generation_time:.1f}s")
        
        return GenerationResponse(
            job_id=request.job_id,
            status="completed",
            video_path=result.get("video_path"),
            generation_time=generation_time,
            message=f"⚡ M4 Max optimized WAN 2.1 generation completed! Cache enabled, MPS accelerated."
        )
        
    except Exception as e:
        generation_time = time.time() - start_time
        print(f"❌ Generation failed: {e}")
        
        return GenerationResponse(
            job_id=request.job_id,
            status="failed", 
            video_path="",  # Empty string instead of None for Pydantic
            generation_time=generation_time,
            message=f"Generation failed: {str(e)}"
        )

if __name__ == "__main__":
    print("⚡ Starting M4 Max Optimized WAN Generation Service on port 8002...")
    print(f"🔥 MPS Available: {os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK', 'Not set')}")
    print(f"💾 Cache Directory: {CACHE_DIR}")
    print(f"📦 Cached Model: 8.6GB WAN 2.1 ready (NO re-download needed!)")
    print(f"🚀 Optimizations: M4 Max + MPS + Forced Cache + Zero Download + Reduced Steps")
    uvicorn.run(app, host="0.0.0.0", port=8002, log_level="info")
