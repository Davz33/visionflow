"""
Mock Remote WAN2.1 Endpoint
For testing the remote WAN integration locally
"""

import asyncio
import time
import uuid
from typing import Dict, Any
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import uvicorn

from ...shared.monitoring import get_logger

logger = get_logger("remote_wan_mock")

app = FastAPI(
    title="Mock Remote WAN2.1 Service",
    description="Mock endpoint for testing remote WAN integration",
    version="0.1.0"
)


class GenerationRequest(BaseModel):
    """Request model for generation"""
    prompt: str
    original_prompt: str
    generation_params: Dict[str, Any]
    routing_decision: Dict[str, Any]
    job_id: str
    optimization: Dict[str, Any] = Field(default_factory=dict)


class GenerationResponse(BaseModel):
    """Response model for generation"""
    status: str
    job_id: str
    generation_time: float
    video_path: str = None
    video_url: str = None
    model: str = "wan2.1-mock"
    quality: str = "medium"
    resolution: str = "512x512"
    error: str = None


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "mock-remote-wan"}


@app.post("/generate", response_model=GenerationResponse)
async def generate_video(request: GenerationRequest) -> GenerationResponse:
    """Mock video generation endpoint"""
    
    logger.info(f"🎬 Mock generating video: '{request.prompt[:50]}...'")
    start_time = time.time()
    
    try:
        # Simulate processing time
        await asyncio.sleep(2)
        
        # Create a mock video file
        output_dir = Path("./generated")
        output_dir.mkdir(exist_ok=True)
        
        video_filename = f"{request.job_id}.mp4"
        video_path = output_dir / video_filename
        
        # Create a simple mock video file (empty file for testing)
        video_path.touch()
        
        generation_time = time.time() - start_time
        
        logger.info(f"✅ Mock generation completed in {generation_time:.2f}s")
        
        return GenerationResponse(
            status="completed",
            job_id=request.job_id,
            generation_time=generation_time,
            video_path=str(video_path),
            model="wan2.1-mock",
            quality=request.generation_params.get("quality", "medium"),
            resolution=request.generation_params.get("resolution", "512x512")
        )
        
    except Exception as e:
        logger.error(f"❌ Mock generation failed: {e}")
        
        return GenerationResponse(
            status="failed",
            job_id=request.job_id,
            generation_time=time.time() - start_time,
            error=str(e)
        )


if __name__ == "__main__":
    uvicorn.run(
        "remote_wan_mock:app",
        host="0.0.0.0",
        port=8002,
        reload=True
    )

