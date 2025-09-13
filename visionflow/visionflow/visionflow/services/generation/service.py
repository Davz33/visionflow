"""Video generation service using WAN 2.1 model."""

from .wan_adapter import wan_adapter
from ...shared.models import (
    PromptOptimization,
    RoutingDecision,
    VideoGenerationRequest,
    GenerationResult,
)
from ...shared.monitoring import get_logger

logger = get_logger("generation_service")


class VideoGenerationService:
    """Wrapper service that uses the WAN adapter for local or remote generation."""
    
    def __init__(self):
        logger.info("Video generation service wrapper initialized with adapter")
    
    async def generate_video(
        self,
        request: VideoGenerationRequest,
        prompt_optimization: PromptOptimization,
        routing_decision: RoutingDecision,
    ) -> GenerationResult:
        """Generate video using WAN adapter (local or remote)."""
        
        logger.info("Delegating to WAN adapter service")
        
        return await wan_adapter.generate_video(
            request,
            prompt_optimization,
            routing_decision
        )


# Service instance - now uses adapter
generation_service = VideoGenerationService()
