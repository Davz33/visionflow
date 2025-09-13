"""
Remote WAN 2.1 Client Service
Communicates with external WAN2.1 endpoints via HTTP API
"""

import asyncio
import httpx
import time
from typing import Dict, Any, Optional
from pathlib import Path

from ...shared.config import get_settings
from ...shared.models import (
    VideoGenerationRequest,
    GenerationResult,
    VideoQuality
)
from ...shared.monitoring import get_logger

logger = get_logger("remote_wan_client")


class RemoteWANClient:
    """Client for communicating with remote WAN2.1 endpoints"""
    
    def __init__(self):
        self.settings = get_settings()
        self.base_url = self.settings.model.remote_wan_url
        self.api_key = self.settings.model.remote_wan_api_key
        self.timeout = self.settings.model.remote_wan_timeout
        self.max_retries = self.settings.model.remote_wan_max_retries
        
        logger.info(f"Remote WAN client initialized with endpoint: {self.base_url}")
    
    async def generate_video(
        self,
        request: VideoGenerationRequest,
        prompt_optimization: Dict[str, Any],
        routing_decision: Dict[str, Any],
    ) -> GenerationResult:
        """Generate video using remote WAN2.1 endpoint"""
        
        logger.info(f"🎬 Generating video via remote WAN2.1: '{request.prompt[:50]}...'")
        start_time = time.time()
        
        try:
            # Prepare the request payload
            payload = self._prepare_request_payload(request, prompt_optimization, routing_decision)
            
            # Make the API call with retries
            result = await self._make_api_call(payload)
            
            # Process the response
            generation_result = self._process_response(result, request)
            
            generation_time = time.time() - start_time
            logger.info(f"✅ Remote generation completed in {generation_time:.2f}s")
            
            return generation_result
            
        except Exception as e:
            logger.error(f"❌ Remote WAN generation failed: {e}")
            return GenerationResult(
                job_id=request.job_id,
                status="failed",
                error=str(e),
                generation_time=time.time() - start_time,
                video_path=None,
                metadata={"source": "remote_wan", "error": str(e)}
            )
    
    def _prepare_request_payload(
        self,
        request: VideoGenerationRequest,
        prompt_optimization: Dict[str, Any],
        routing_decision: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Prepare the request payload for the remote API"""
        
        # Parse resolution
        width, height = map(int, request.resolution.split('x'))
        
        payload = {
            "prompt": request.prompt,
            "original_prompt": prompt_optimization.get("original_prompt", request.prompt),
            "generation_params": {
                "duration": request.duration,
                "fps": request.fps,
                "resolution": request.resolution,
                "width": width,
                "height": height,
                "seed": request.seed,
                "guidance_scale": request.guidance_scale,
                "num_inference_steps": request.num_inference_steps,
                "quality": request.quality.value,
            },
            "routing_decision": routing_decision,
            "job_id": request.job_id,
            "optimization": prompt_optimization,
        }
        
        return payload
    
    async def _make_api_call(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Make the API call to the remote WAN endpoint with retries"""
        
        headers = {
            "Content-Type": "application/json",
            "User-Agent": "VisionFlow-RemoteClient/1.0"
        }
        
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            for attempt in range(self.max_retries):
                try:
                    logger.info(f"🔄 Attempting remote API call (attempt {attempt + 1}/{self.max_retries})")
                    
                    response = await client.post(
                        f"{self.base_url}/generate",
                        json=payload,
                        headers=headers
                    )
                    
                    response.raise_for_status()
                    result = response.json()
                    
                    logger.info("✅ Remote API call successful")
                    return result
                    
                except httpx.TimeoutException:
                    logger.warning(f"⏰ API call timeout on attempt {attempt + 1}")
                    if attempt == self.max_retries - 1:
                        raise Exception("API call timed out after all retries")
                    
                except httpx.HTTPStatusError as e:
                    logger.warning(f"🌐 HTTP error {e.response.status_code} on attempt {attempt + 1}")
                    if attempt == self.max_retries - 1:
                        raise Exception(f"API call failed with status {e.response.status_code}")
                    
                except Exception as e:
                    logger.warning(f"❌ API call failed on attempt {attempt + 1}: {e}")
                    if attempt == self.max_retries - 1:
                        raise
                
                # Wait before retry
                await asyncio.sleep(2 ** attempt)  # Exponential backoff
    
    def _process_response(self, response: Dict[str, Any], request: VideoGenerationRequest) -> GenerationResult:
        """Process the response from the remote API"""
        
        status = response.get("status", "unknown")
        
        if status == "completed":
            # Download the video file if provided as URL
            video_path = self._handle_video_output(response, request)
            
            return GenerationResult(
                job_id=request.job_id,
                status="completed",
                video_path=video_path,
                generation_time=response.get("generation_time", 0),
                metadata={
                    "source": "remote_wan",
                    "endpoint": self.base_url,
                    "model": response.get("model", "unknown"),
                    "quality": response.get("quality", "unknown"),
                    "resolution": response.get("resolution", "unknown"),
                }
            )
        
        else:
            error_msg = response.get("error", "Unknown error from remote service")
            return GenerationResult(
                job_id=request.job_id,
                status="failed",
                error=error_msg,
                generation_time=response.get("generation_time", 0),
                metadata={"source": "remote_wan", "error": error_msg}
            )
    
    def _handle_video_output(self, response: Dict[str, Any], request: VideoGenerationRequest) -> Optional[str]:
        """Handle video output from remote service"""
        
        video_url = response.get("video_url")
        video_path = response.get("video_path")
        
        if video_path:
            # Video is already saved locally
            return video_path
        
        elif video_url:
            # Download video from URL
            return self._download_video(video_url, request.job_id)
        
        else:
            logger.warning("No video output found in remote response")
            return None
    
    def _download_video(self, video_url: str, job_id: str) -> Optional[str]:
        """Download video from remote URL"""
        
        try:
            # Create output directory
            output_dir = Path(self.settings.model.cache_dir) / "generated"
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Generate filename
            video_path = output_dir / f"{job_id}.mp4"
            
            logger.info(f"📥 Downloading video from {video_url}")
            
            # Download the video (this would need to be implemented with proper async download)
            # For now, return the URL as the path
            return str(video_path)
            
        except Exception as e:
            logger.error(f"Failed to download video: {e}")
            return None
    
    async def health_check(self) -> bool:
        """Check if the remote WAN endpoint is healthy"""
        
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(f"{self.base_url}/health")
                return response.status_code == 200
        except Exception as e:
            logger.warning(f"Health check failed: {e}")
            return False


# Global instance
remote_wan_client = RemoteWANClient()

