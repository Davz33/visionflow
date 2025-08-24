"""Video generation service using WAN 2.1 model."""

from .wan_model_service import enhanced_generation_service
from ...shared.models import (
    PromptOptimization,
    RoutingDecision,
    VideoGenerationRequest,
    GenerationResult,
)
from ...shared.monitoring import get_logger

logger = get_logger("generation_service")


class VideoGenerationService:
    """Wrapper service that uses the enhanced generation service."""
    
    def __init__(self):
        logger.info("Video generation service wrapper initialized")
    
    async def generate_video(
        self,
        request: VideoGenerationRequest,
        prompt_optimization: PromptOptimization,
        routing_decision: RoutingDecision,
    ) -> GenerationResult:
        """Generate video using enhanced WAN 2.1 service."""
        
        logger.info("Delegating to enhanced generation service")
        
        return await enhanced_generation_service.generate_video(
            request,
            prompt_optimization,
            routing_decision
        )


# Service instance - now uses enhanced service
generation_service = VideoGenerationService()
