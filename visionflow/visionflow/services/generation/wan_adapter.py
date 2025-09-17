"""
WAN Service Adapter
Switches between local and remote WAN2.1 services based on configuration
"""

from typing import Dict, Any
from ...shared.config import get_settings
from ...shared.models import (
    VideoGenerationRequest,
    GenerationResult,
    PromptOptimization,
    RoutingDecision,
)
from ...shared.monitoring import get_logger

logger = get_logger("wan_adapter")


class WANServiceAdapter:
    """Adapter that switches between local and remote WAN services"""
    
    def __init__(self):
        self.settings = get_settings()
        self._local_service = None
        self._remote_service = None
        
        logger.info(f"WAN adapter initialized - Remote mode: {self.settings.model.use_remote_wan}")
    
    async def generate_video(
        self,
        request: VideoGenerationRequest,
        prompt_optimization: PromptOptimization,
        routing_decision: RoutingDecision,
    ) -> GenerationResult:
        """Generate video using either local or remote WAN service"""
        
        if self.settings.model.use_remote_wan:
            return await self._generate_remote(request, prompt_optimization, routing_decision)
        else:
            return await self._generate_local(request, prompt_optimization, routing_decision)
    
    async def _generate_remote(
        self,
        request: VideoGenerationRequest,
        prompt_optimization: PromptOptimization,
        routing_decision: RoutingDecision,
    ) -> GenerationResult:
        """Generate video using remote WAN service"""
        
        logger.info("🌐 Using remote WAN2.1 service")
        
        if self._remote_service is None:
            from .remote_wan_client import remote_wan_client
            self._remote_service = remote_wan_client
        
        # Convert Pydantic models to dicts for remote service
        prompt_opt_dict = prompt_optimization.dict() if hasattr(prompt_optimization, 'dict') else prompt_optimization
        routing_dict = routing_decision.dict() if hasattr(routing_decision, 'dict') else routing_decision
        
        return await self._remote_service.generate_video(
            request,
            prompt_opt_dict,
            routing_dict
        )
    
    async def _generate_local(
        self,
        request: VideoGenerationRequest,
        prompt_optimization: PromptOptimization,
        routing_decision: RoutingDecision,
    ) -> GenerationResult:
        """Generate video using local WAN service"""
        
        logger.info("🏠 Using local WAN2.1 service")
        
        if self._local_service is None:
            from .wan_model_service import enhanced_generation_service
            self._local_service = enhanced_generation_service
        
        return await self._local_service.generate_video(
            request,
            prompt_optimization,
            routing_decision
        )
    
    async def health_check(self) -> Dict[str, Any]:
        """Check health of the active service"""
        
        if self.settings.model.use_remote_wan:
            if self._remote_service is None:
                from .remote_wan_client import remote_wan_client
                self._remote_service = remote_wan_client
            
            is_healthy = await self._remote_service.health_check()
            return {
                "service": "remote_wan",
                "healthy": is_healthy,
                "endpoint": self.settings.model.remote_wan_url
            }
        else:
            # For local service, we assume it's healthy if it can be imported
            return {
                "service": "local_wan",
                "healthy": True,
                "device": self.settings.model.device
            }


# Global adapter instance
wan_adapter = WANServiceAdapter()





