"""Routing service for optimal resource allocation and model selection."""

from typing import Any, Dict

from ...shared.config import get_settings
from ...shared.models import IntentAnalysis, RoutingDecision, VideoGenerationRequest, VideoQuality
from ...shared.monitoring import get_logger, track_request_metrics

logger = get_logger("router_service")


class RoutingService:
    """Service for routing decisions and resource allocation."""
    
    def __init__(self):
        self.settings = get_settings()
        
        # Model tier configurations
        self.model_tiers = {
            "basic": {
                "max_duration": 10,
                "max_resolution": "512x512",
                "cost_per_second": 0.05,
                "gpu_memory_gb": 4,
                "estimated_time_multiplier": 1.0,
            },
            "standard": {
                "max_duration": 20,
                "max_resolution": "1024x1024",
                "cost_per_second": 0.12,
                "gpu_memory_gb": 8,
                "estimated_time_multiplier": 1.5,
            },
            "premium": {
                "max_duration": 30,
                "max_resolution": "1920x1080",
                "cost_per_second": 0.25,
                "gpu_memory_gb": 16,
                "estimated_time_multiplier": 2.0,
            },
            "ultra": {
                "max_duration": 60,
                "max_resolution": "2048x2048",
                "cost_per_second": 0.50,
                "gpu_memory_gb": 24,
                "estimated_time_multiplier": 3.0,
            },
        }
        
        # Priority mapping based on intent and complexity
        self.priority_mapping = {
            ("scene_creation", "low"): 5,
            ("scene_creation", "medium"): 6,
            ("scene_creation", "high"): 7,
            ("character_animation", "low"): 6,
            ("character_animation", "medium"): 7,
            ("character_animation", "high"): 8,
            ("object_focus", "low"): 4,
            ("object_focus", "medium"): 5,
            ("object_focus", "high"): 6,
            ("abstract_artistic", "low"): 3,
            ("abstract_artistic", "medium"): 4,
            ("abstract_artistic", "high"): 5,
            ("nature_documentary", "low"): 7,
            ("nature_documentary", "medium"): 8,
            ("nature_documentary", "high"): 9,
        }
    
    @track_request_metrics("router_service")
    async def make_routing_decision(
        self,
        request: VideoGenerationRequest,
        intent: IntentAnalysis,
    ) -> RoutingDecision:
        """Make routing decision based on request and intent analysis."""
        
        # Select appropriate model tier
        model_tier = self._select_model_tier(request, intent)
        
        # Calculate cost estimation
        estimated_cost = self._estimate_cost(request, model_tier)
        
        # Estimate processing duration
        estimated_duration = self._estimate_duration(request, intent, model_tier)
        
        # Determine resource requirements
        resource_requirements = self._get_resource_requirements(model_tier, request)
        
        # Calculate priority
        priority = self._calculate_priority(intent, request)
        
        logger.info(
            "Routing decision made",
            model_tier=model_tier,
            estimated_cost=estimated_cost,
            estimated_duration=estimated_duration,
            priority=priority,
            intent_type=intent.intent_type,
            complexity=intent.complexity_score,
        )
        
        return RoutingDecision(
            model_tier=model_tier,
            estimated_cost=estimated_cost,
            estimated_duration=estimated_duration,
            resource_requirements=resource_requirements,
            priority=priority,
        )
    
    def _select_model_tier(self, request: VideoGenerationRequest, intent: IntentAnalysis) -> str:
        """Select the appropriate model tier based on requirements."""
        
        # Start with quality-based selection
        quality_tier_mapping = {
            VideoQuality.LOW: "basic",
            VideoQuality.MEDIUM: "standard", 
            VideoQuality.HIGH: "premium",
            VideoQuality.ULTRA: "ultra",
        }
        
        base_tier = quality_tier_mapping[request.quality]
        
        # Adjust based on complexity
        if intent.complexity_score > 0.8:
            # High complexity requires better tier
            tier_upgrades = {
                "basic": "standard",
                "standard": "premium", 
                "premium": "ultra",
                "ultra": "ultra",
            }
            base_tier = tier_upgrades[base_tier]
        elif intent.complexity_score < 0.3:
            # Low complexity can use lower tier
            tier_downgrades = {
                "ultra": "premium",
                "premium": "standard",
                "standard": "basic",
                "basic": "basic",
            }
            base_tier = tier_downgrades[base_tier]
        
        # Check if request parameters fit within tier limits
        tier_config = self.model_tiers[base_tier]
        
        # Check duration constraints
        if request.duration > tier_config["max_duration"]:
            # Need to upgrade to higher tier
            for tier_name, config in self.model_tiers.items():
                if (config["max_duration"] >= request.duration and
                    self._tier_priority(tier_name) > self._tier_priority(base_tier)):
                    base_tier = tier_name
                    break
        
        # Check resolution constraints
        width, height = map(int, request.resolution.split("x"))
        max_width, max_height = map(int, tier_config["max_resolution"].split("x"))
        
        if width > max_width or height > max_height:
            # Need to upgrade to higher tier
            for tier_name, config in self.model_tiers.items():
                tier_width, tier_height = map(int, config["max_resolution"].split("x"))
                if (tier_width >= width and tier_height >= height and
                    self._tier_priority(tier_name) > self._tier_priority(base_tier)):
                    base_tier = tier_name
                    break
        
        return base_tier
    
    def _tier_priority(self, tier: str) -> int:
        """Get numerical priority of tier for comparison."""
        priorities = {"basic": 1, "standard": 2, "premium": 3, "ultra": 4}
        return priorities.get(tier, 1)
    
    def _estimate_cost(self, request: VideoGenerationRequest, model_tier: str) -> float:
        """Estimate the cost of processing the request."""
        tier_config = self.model_tiers[model_tier]
        base_cost = tier_config["cost_per_second"] * request.duration
        
        # Factor in inference steps (more steps = higher cost)
        steps_multiplier = 1.0 + (request.num_inference_steps - 20) * 0.01
        
        # Factor in resolution
        width, height = map(int, request.resolution.split("x"))
        resolution_pixels = width * height
        resolution_multiplier = 1.0 + (resolution_pixels - 512 * 512) / (1024 * 1024)
        
        # Factor in guidance scale (higher guidance = more compute)
        guidance_multiplier = 1.0 + (request.guidance_scale - 7.5) * 0.05
        
        total_cost = base_cost * steps_multiplier * resolution_multiplier * guidance_multiplier
        return round(total_cost, 4)
    
    def _estimate_duration(
        self,
        request: VideoGenerationRequest,
        intent: IntentAnalysis,
        model_tier: str,
    ) -> int:
        """Estimate processing duration in seconds."""
        tier_config = self.model_tiers[model_tier]
        
        # Base time per second of video
        base_time_per_second = 10  # seconds
        
        # Apply tier multiplier
        tier_multiplier = tier_config["estimated_time_multiplier"]
        
        # Factor in complexity
        complexity_multiplier = 1.0 + intent.complexity_score
        
        # Factor in inference steps
        steps_multiplier = request.num_inference_steps / 20
        
        # Factor in resolution
        width, height = map(int, request.resolution.split("x"))
        resolution_multiplier = (width * height) / (512 * 512)
        
        estimated_time = (
            base_time_per_second *
            request.duration *
            tier_multiplier *
            complexity_multiplier *
            steps_multiplier *
            resolution_multiplier
        )
        
        return max(10, int(estimated_time))  # Minimum 10 seconds
    
    def _get_resource_requirements(self, model_tier: str, request: VideoGenerationRequest) -> Dict[str, Any]:
        """Get resource requirements for the model tier."""
        tier_config = self.model_tiers[model_tier]
        
        # Calculate GPU memory requirements
        width, height = map(int, request.resolution.split("x"))
        video_memory_mb = (width * height * request.duration * request.fps * 4) // (1024 * 1024)  # Rough estimate
        
        return {
            "gpu_memory_gb": tier_config["gpu_memory_gb"],
            "estimated_video_memory_mb": video_memory_mb,
            "cpu_cores": 2 if model_tier in ["basic", "standard"] else 4,
            "ram_gb": tier_config["gpu_memory_gb"] // 2,  # Roughly half of GPU memory
            "storage_gb": max(1, request.duration // 10),  # 1GB per 10 seconds of video
            "network_bandwidth_mbps": 100 if model_tier in ["premium", "ultra"] else 50,
        }
    
    def _calculate_priority(self, intent: IntentAnalysis, request: VideoGenerationRequest) -> int:
        """Calculate processing priority based on intent and request."""
        
        # Get base priority from intent and complexity
        complexity_level = (
            "high" if intent.complexity_score > 0.7
            else "medium" if intent.complexity_score > 0.4
            else "low"
        )
        
        base_priority = self.priority_mapping.get(
            (intent.intent_type, complexity_level), 5
        )
        
        # Adjust based on quality (higher quality = higher priority)
        quality_adjustment = {
            VideoQuality.LOW: -1,
            VideoQuality.MEDIUM: 0,
            VideoQuality.HIGH: 1,
            VideoQuality.ULTRA: 2,
        }
        
        final_priority = base_priority + quality_adjustment[request.quality]
        
        # Clamp to valid range
        return max(1, min(10, final_priority))


# Service instance
routing_service = RoutingService()
