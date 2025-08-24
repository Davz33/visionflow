#!/usr/bin/env python3
"""
Simple generation service for architecture testing
Runs as a standalone FastAPI service
"""

import asyncio
import time
from datetime import datetime
from typing import Dict, Any, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

# Create FastAPI app
app = FastAPI(
    title="Simple Generation Service", 
    description="Lightweight service for testing architecture",
    version="0.1.0"
)

class GenerationRequest(BaseModel):
    """Request model for generation"""
    prompt: str
    original_prompt: str
    generation_params: Dict[str, Any]
    routing_decision: Dict[str, Any]
    job_id: str

class GenerationResponse(BaseModel):
    """Response model for generation"""
    job_id: str
    status: str
    video_path: Optional[str] = None
    generation_time: float
    message: str

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "service": "simple-generation",
        "gpu_available": False,  # Mock for now
        "model_loaded": True     # Mock for now
    }

@app.post("/generate", response_model=GenerationResponse)
async def generate_video(request: GenerationRequest):
    """Generate video endpoint - currently returns mock result"""
    start_time = time.time()
    
    print(f"🎬 Received generation request for job {request.job_id}")
    print(f"   Prompt: {request.prompt}")
    print(f"   Params: {request.generation_params}")
    
    # Simulate some processing time based on parameters
    duration = request.generation_params.get("duration", 1)
    processing_time = min(duration * 2, 10)  # Max 10 seconds for testing
    
    print(f"   ⏳ Simulating {processing_time}s of processing...")
    await asyncio.sleep(processing_time)
    
    generation_time = time.time() - start_time
    mock_video_path = f"/generated/mock_video_{request.job_id}.mp4"
    
    print(f"   ✅ Mock generation complete in {generation_time:.1f}s")
    
    return GenerationResponse(
        job_id=request.job_id,
        status="completed",
        video_path=mock_video_path,
        generation_time=generation_time,
        message=f"Mock video generated successfully for '{request.prompt[:50]}...'"
    )

if __name__ == "__main__":
    print("🚀 Starting Simple Generation Service on port 8002...")
    uvicorn.run(app, host="0.0.0.0", port=8002, log_level="info")
