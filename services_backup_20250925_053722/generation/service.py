"""Simple video generation service for testing."""

import asyncio
from typing import Dict, Any
from ...shared.models import VideoGenerationRequest
from ...shared.monitoring import get_logger

logger = get_logger(__name__)

class MockGenerationService:
    """Mock video generation service for testing API endpoints."""
    
    def __init__(self):
        logger.info("Mock generation service initialized")
    
    async def generate_video(self, request: VideoGenerationRequest) -> Dict[str, Any]:
        """Mock video generation that simulates processing."""
        logger.info(f"Mock generating video for prompt: {request.prompt}")
        
        # Simulate some processing time
        await asyncio.sleep(2)
        
        # Return mock result
        return {
            "status": "completed",
            "message": "Mock video generated successfully",
            "video_path": f"/generated/mock_video_{hash(request.prompt)}.mp4",
            "duration": request.duration,
            "quality": request.quality,
            "fps": request.fps,
            "resolution": request.resolution
        }

# Create service instance
generation_service = MockGenerationService()
