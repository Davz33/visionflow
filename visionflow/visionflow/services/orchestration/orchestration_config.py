"""
Enhanced Orchestration Configuration for Multi-Agent Systems
Based on 2024 best practices for adaptive agent coordination
"""

from enum import Enum
from typing import Dict, List, Optional
from pathlib import Path
from pydantic import BaseModel, Field

from ...shared.config import get_settings
from ...shared.monitoring import get_logger

logger = get_logger("orchestration_config")
settings = get_settings()


class OrchestrationMode(str, Enum):
    """Available orchestration modes for video generation"""
    ENHANCED_MULTI_AGENT = "enhanced_multi_agent"
    SWARM_INTELLIGENCE = "swarm_orchestrator"
    LANGGRAPH_WORKFLOW = "langgraph_workflow"
    AGENT_PATTERN = "agent_pattern"
    LEGACY_SAGE = "legacy_sage"


class ComplexityLevel(str, Enum):
    """Request complexity levels for adaptive routing"""
    SIMPLE = "simple"
    MEDIUM = "medium"
    COMPLEX = "complex"
    EXPERT = "expert"


class AgentSpecialization(str, Enum):
    """Available agent specializations"""
    TECHNICAL_EXPERT = "technical_expert"
    QUALITY_SPECIALIST = "quality_specialist"
    EFFICIENCY_OPTIMIZER = "efficiency_optimizer"
    CREATIVE_DIRECTOR = "creative_director"
    USER_EXPERIENCE = "user_experience"


class OrchestrationStrategy(BaseModel):
    """Configuration for orchestration strategy"""
    mode: OrchestrationMode = Field(description="Primary orchestration mode")
    fallback_modes: List[OrchestrationMode] = Field(description="Fallback orchestration modes")
    complexity_threshold: Dict[ComplexityLevel, OrchestrationMode] = Field(
        description="Complexity-based mode selection"
    )
    performance_requirements: Dict[str, float] = Field(
        description="Performance requirements for mode selection"
    )
    quality_thresholds: Dict[str, float] = Field(
        description="Quality thresholds for different modes"
    )


class SwarmConfiguration(BaseModel):
    """Configuration for swarm intelligence orchestration"""
    agent_count: int = Field(default=3, description="Number of agents in swarm")
    consensus_threshold: float = Field(default=0.8, description="Consensus requirement")
    max_collaboration_rounds: int = Field(default=3, description="Maximum collaboration rounds")
    specializations: List[AgentSpecialization] = Field(
        description="Required agent specializations"
    )
    load_balancing: str = Field(default="dynamic", description="Load balancing strategy")
    conflict_resolution: str = Field(default="consensus_voting", description="Conflict resolution method")


class SupervisorConfiguration(BaseModel):
    """Configuration for supervisor-based orchestration"""
    coordination_pattern: str = Field(default="hierarchical", description="Coordination pattern")
    agent_specializations: Dict[str, List[str]] = Field(
        description="Agent specialization mappings"
    )
    task_assignment_strategy: str = Field(default="intelligent", description="Task assignment strategy")
    quality_gates: List[str] = Field(description="Quality checkpoints")
    escalation_rules: Dict[str, str] = Field(description="Escalation rules")


class PerformanceMetrics(BaseModel):
    """Performance metrics for orchestration evaluation"""
    response_time_targets: Dict[str, float] = Field(description="Response time targets by complexity")
    quality_score_minimums: Dict[str, float] = Field(description="Minimum quality scores")
    resource_utilization_targets: Dict[str, float] = Field(description="Resource utilization targets")
    user_satisfaction_targets: Dict[str, float] = Field(description="User satisfaction targets")
    cost_efficiency_targets: Dict[str, float] = Field(description="Cost efficiency targets")


class AdaptiveOrchestrationConfig(BaseModel):
    """Comprehensive adaptive orchestration configuration"""
    
    # Core configuration
    default_strategy: OrchestrationStrategy
    swarm_config: SwarmConfiguration
    supervisor_config: SupervisorConfiguration
    performance_metrics: PerformanceMetrics
    
    # Adaptive behavior
    enable_adaptive_routing: bool = Field(default=True, description="Enable adaptive mode selection")
    learning_enabled: bool = Field(default=True, description="Enable performance learning")
    feedback_integration: bool = Field(default=True, description="Integrate user feedback")
    
    # Advanced features
    hybrid_orchestration: bool = Field(default=False, description="Allow hybrid orchestration modes")
    real_time_optimization: bool = Field(default=True, description="Enable real-time optimization")
    predictive_scaling: bool = Field(default=True, description="Enable predictive resource scaling")
    
    # Monitoring and observability
    detailed_logging: bool = Field(default=True, description="Enable detailed orchestration logging")
    metrics_collection: bool = Field(default=True, description="Enable metrics collection")
    performance_tracking: bool = Field(default=True, description="Enable performance tracking")


class OrchestrationModeSelector:
    """Intelligent orchestration mode selector based on request characteristics"""
    
    def __init__(self, config: AdaptiveOrchestrationConfig):
        self.config = config
        self.mode_performance_history: Dict[str, List[float]] = {}
        self.request_patterns: Dict[str, int] = {}
    
    def analyze_request_complexity(self, request: str) -> ComplexityLevel:
        """Analyze request complexity for mode selection"""
        
        # Length-based analysis
        word_count = len(request.split())
        
        # Keyword-based analysis
        complex_keywords = [
            "multiple", "various", "different", "complex", "detailed", "sophisticated",
            "professional", "cinematic", "narrative", "character", "scene", "style"
        ]
        
        expert_keywords = [
            "optimization", "performance", "technical", "advanced", "custom",
            "experimental", "research", "academic", "professional"
        ]
        
        # Calculate complexity score
        complexity_score = 0
        
        # Word count factor
        if word_count > 30:
            complexity_score += 3
        elif word_count > 15:
            complexity_score += 2
        elif word_count > 5:
            complexity_score += 1
        
        # Keyword factors
        complex_count = sum(1 for kw in complex_keywords if kw in request.lower())
        expert_count = sum(1 for kw in expert_keywords if kw in request.lower())
        
        complexity_score += complex_count * 2 + expert_count * 3
        
        # Determine complexity level
        if complexity_score >= 8:
            return ComplexityLevel.EXPERT
        elif complexity_score >= 5:
            return ComplexityLevel.COMPLEX
        elif complexity_score >= 2:
            return ComplexityLevel.MEDIUM
        else:
            return ComplexityLevel.SIMPLE
    
    def select_orchestration_mode(
        self, 
        request: str, 
        user_preferences: Optional[Dict] = None,
        performance_history: Optional[Dict] = None
    ) -> OrchestrationMode:
        """Select optimal orchestration mode based on multiple factors"""
        
        complexity = self.analyze_request_complexity(request)
        
        # Check complexity-based thresholds
        if complexity in self.config.default_strategy.complexity_threshold:
            recommended_mode = self.config.default_strategy.complexity_threshold[complexity]
        else:
            recommended_mode = self.config.default_strategy.mode
        
        # Consider user preferences
        if user_preferences:
            preferred_mode = user_preferences.get("orchestration_mode")
            if preferred_mode and OrchestrationMode(preferred_mode):
                logger.info(f"Using user preferred mode: {preferred_mode}")
                return OrchestrationMode(preferred_mode)
        
        # Consider performance history
        if performance_history and self.config.learning_enabled:
            mode_scores = self._calculate_mode_scores(complexity, performance_history)
            if mode_scores:
                best_mode = max(mode_scores.items(), key=lambda x: x[1])[0]
                logger.info(f"Performance-based mode selection: {best_mode}")
                return OrchestrationMode(best_mode)
        
        # Adaptive selection based on current load
        if self.config.enable_adaptive_routing:
            adapted_mode = self._adaptive_mode_selection(complexity, recommended_mode)
            if adapted_mode != recommended_mode:
                logger.info(f"Adaptive routing: {recommended_mode} -> {adapted_mode}")
                return adapted_mode
        
        logger.info(f"Selected orchestration mode: {recommended_mode} for complexity: {complexity}")
        return recommended_mode
    
    def _calculate_mode_scores(self, complexity: ComplexityLevel, history: Dict) -> Dict[str, float]:
        """Calculate mode scores based on historical performance"""
        scores = {}
        
        for mode in OrchestrationMode:
            mode_history = history.get(mode.value, {})
            
            # Performance factors
            avg_response_time = mode_history.get("avg_response_time", 60)
            avg_quality_score = mode_history.get("avg_quality_score", 0.8)
            success_rate = mode_history.get("success_rate", 0.9)
            user_satisfaction = mode_history.get("user_satisfaction", 0.8)
            
            # Calculate composite score
            time_score = max(0, 1 - (avg_response_time / 120))  # Normalize to 120s max
            quality_weight = 0.3
            speed_weight = 0.25
            reliability_weight = 0.25
            satisfaction_weight = 0.2
            
            composite_score = (
                avg_quality_score * quality_weight +
                time_score * speed_weight +
                success_rate * reliability_weight +
                user_satisfaction * satisfaction_weight
            )
            
            scores[mode.value] = composite_score
        
        return scores
    
    def _adaptive_mode_selection(
        self, 
        complexity: ComplexityLevel, 
        recommended_mode: OrchestrationMode
    ) -> OrchestrationMode:
        """Adaptively select mode based on current system state"""
        
        # Mock system load analysis (in production, this would check actual metrics)
        current_load = 0.6  # Mock 60% load
        
        # Adapt based on load
        if current_load > 0.8:
            # High load - prefer simpler, faster modes
            if complexity in [ComplexityLevel.COMPLEX, ComplexityLevel.EXPERT]:
                logger.info("High load detected, simplifying orchestration mode")
                return OrchestrationMode.LANGGRAPH_WORKFLOW
            elif complexity == ComplexityLevel.MEDIUM:
                return OrchestrationMode.AGENT_PATTERN
        elif current_load < 0.3:
            # Low load - can use more sophisticated modes
            if complexity == ComplexityLevel.SIMPLE:
                logger.info("Low load detected, upgrading to enhanced mode")
                return OrchestrationMode.ENHANCED_MULTI_AGENT
        
        return recommended_mode
    
    def update_performance_metrics(
        self, 
        mode: OrchestrationMode, 
        metrics: Dict[str, float]
    ) -> None:
        """Update performance metrics for mode selection learning"""
        
        if mode.value not in self.mode_performance_history:
            self.mode_performance_history[mode.value] = []
        
        # Add new performance data
        composite_score = (
            metrics.get("quality_score", 0.8) * 0.4 +
            (1 - min(metrics.get("response_time", 60) / 120, 1)) * 0.3 +
            metrics.get("success_rate", 1.0) * 0.3
        )
        
        self.mode_performance_history[mode.value].append(composite_score)
        
        # Keep only recent history (last 100 entries)
        if len(self.mode_performance_history[mode.value]) > 100:
            self.mode_performance_history[mode.value] = \
                self.mode_performance_history[mode.value][-100:]
        
        logger.info(f"Updated performance metrics for {mode.value}: {composite_score:.3f}")


def create_default_orchestration_config() -> AdaptiveOrchestrationConfig:
    """Create default orchestration configuration with 2024 best practices"""
    
    return AdaptiveOrchestrationConfig(
        default_strategy=OrchestrationStrategy(
            mode=OrchestrationMode.ENHANCED_MULTI_AGENT,
            fallback_modes=[
                OrchestrationMode.SWARM_INTELLIGENCE,
                OrchestrationMode.LANGGRAPH_WORKFLOW,
                OrchestrationMode.AGENT_PATTERN,
                OrchestrationMode.LEGACY_SAGE
            ],
            complexity_threshold={
                ComplexityLevel.SIMPLE: OrchestrationMode.AGENT_PATTERN,
                ComplexityLevel.MEDIUM: OrchestrationMode.ENHANCED_MULTI_AGENT,
                ComplexityLevel.COMPLEX: OrchestrationMode.SWARM_INTELLIGENCE,
                ComplexityLevel.EXPERT: OrchestrationMode.SWARM_INTELLIGENCE
            },
            performance_requirements={
                "max_response_time": 90.0,
                "min_quality_score": 0.8,
                "min_success_rate": 0.95
            },
            quality_thresholds={
                "excellent": 0.9,
                "good": 0.8,
                "acceptable": 0.7
            }
        ),
        swarm_config=SwarmConfiguration(
            agent_count=3,
            consensus_threshold=0.8,
            max_collaboration_rounds=3,
            specializations=[
                AgentSpecialization.TECHNICAL_EXPERT,
                AgentSpecialization.QUALITY_SPECIALIST,
                AgentSpecialization.EFFICIENCY_OPTIMIZER
            ],
            load_balancing="dynamic",
            conflict_resolution="consensus_voting"
        ),
        supervisor_config=SupervisorConfiguration(
            coordination_pattern="hierarchical",
            agent_specializations={
                "video_generation": ["technical_optimization", "parameter_tuning"],
                "quality_assurance": ["validation", "compliance", "user_satisfaction"],
                "performance": ["efficiency", "resource_management", "optimization"]
            },
            task_assignment_strategy="intelligent",
            quality_gates=["requirements_check", "technical_validation", "quality_assessment"],
            escalation_rules={
                "quality_failure": "quality_specialist",
                "performance_issue": "efficiency_optimizer",
                "technical_problem": "technical_expert"
            }
        ),
        performance_metrics=PerformanceMetrics(
            response_time_targets={
                "simple": 30.0,
                "medium": 60.0,
                "complex": 90.0,
                "expert": 120.0
            },
            quality_score_minimums={
                "simple": 0.75,
                "medium": 0.80,
                "complex": 0.85,
                "expert": 0.90
            },
            resource_utilization_targets={
                "cpu": 0.75,
                "memory": 0.80,
                "gpu": 0.85
            },
            user_satisfaction_targets={
                "simple": 0.85,
                "medium": 0.88,
                "complex": 0.90,
                "expert": 0.92
            },
            cost_efficiency_targets={
                "cost_per_generation": 0.50,
                "resource_efficiency": 0.80
            }
        ),
        enable_adaptive_routing=True,
        learning_enabled=True,
        feedback_integration=True,
        hybrid_orchestration=False,
        real_time_optimization=True,
        predictive_scaling=True,
        detailed_logging=True,
        metrics_collection=True,
        performance_tracking=True
    )


# Global configuration instance
_orchestration_config = None

def get_orchestration_config() -> AdaptiveOrchestrationConfig:
    """Get or create orchestration configuration"""
    global _orchestration_config
    if _orchestration_config is None:
        _orchestration_config = create_default_orchestration_config()
    return _orchestration_config


def get_mode_selector() -> OrchestrationModeSelector:
    """Get orchestration mode selector"""
    config = get_orchestration_config()
    return OrchestrationModeSelector(config)
