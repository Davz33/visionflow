"""
Video evaluation services for quality assessment and analysis.

This package provides comprehensive video evaluation capabilities including:
- Industry-standard metrics (LPIPS, FVMD, CLIP, ETVA)
- Subjective evaluation using LLaVA or other LLMs
- Batch processing with configurable sampling strategies
- Confidence-based automated flagging
- Quality assessment and scoring
"""

# Core quality metrics and data models
from .quality_metrics import (
    QualityDimensions,
    TechnicalMetrics,
    ContentMetrics,
    AestheticMetrics,
    UserExperienceMetrics,
    PerformanceMetrics,
    ComplianceMetrics,
    IndustryStandardMetrics,
    EvaluationResult,
    RubricCriteria,
    QualitativeAssessment,
    get_default_video_quality_rubrics,
    create_qualitative_assessment,
    validate_evaluation_consistency
)

# Industry metrics implementation
from .industry_metrics_implementation import (
    LPIPSEvaluator,
    FVMDEvaluator,
    CLIPEvaluator,
    ETVAEvaluator,
    create_industry_metrics_evaluator,
    evaluate_video_with_industry_metrics
)

# Batch evaluation system
from .batch_evaluation_system import (
    SamplingStrategy,
    ConfidenceThreshold,
    BatchProcessingConfig,
    FrameSample,
    VideoEvaluationJob,
    FrameSampler,
    ConfidenceManager,
    BatchEvaluationProcessor,
    create_batch_processor
)

# LLaVA-based subjective evaluation
from .llava_analyzer import (
    LLaVAAnalyzer,
    create_llava_analyzer
)

# LLM configuration and provider management
from .llm_config import (
    LLMProvider,
    LLMConfig
)

# Legacy components (maintained for backward compatibility)
from .autoraters import (
    ContentAnalyzer,
    TechnicalAnalyzer,
    AestheticAnalyzer,
    UserExperienceAnalyzer,
    PerformanceAnalyzer,
    ComplianceAnalyzer,
    create_content_analyzer,
    create_technical_analyzer,
    create_aesthetic_analyzer,
    create_user_experience_analyzer,
    create_performance_analyzer,
    create_compliance_analyzer
)

from .vision_models import (
    GeminiVisionAnalyzer,
    get_video_quality_orchestrator
)

from .benchmarks import (
    VideoQualityBenchmark,
    BenchmarkResult,
    run_benchmark_suite
)

from .continuous_learning import (
    ContinuousLearningSystem,
    FeedbackProcessor,
    ModelUpdater
)

from .evaluation_orchestrator import (
    EvaluationOrchestrator,
    create_evaluation_orchestrator
)

# Quality metrics utilities
from .quality_metrics import (
    calculate_overall_quality_score,
    generate_quality_report,
    export_evaluation_results
)

__all__ = [
    # Core quality metrics
    "QualityDimensions",
    "TechnicalMetrics", 
    "ContentMetrics",
    "AestheticMetrics",
    "UserExperienceMetrics",
    "PerformanceMetrics",
    "ComplianceMetrics",
    "IndustryStandardMetrics",
    "EvaluationResult",
    "RubricCriteria",
    "QualitativeAssessment",
    "get_default_video_quality_rubrics",
    "create_qualitative_assessment",
    "validate_evaluation_consistency",
    
    # Industry metrics
    "LPIPSEvaluator",
    "FVMDEvaluator", 
    "CLIPEvaluator",
    "ETVAEvaluator",
    "create_industry_metrics_evaluator",
    "evaluate_video_with_industry_metrics",
    
    # Batch evaluation
    "SamplingStrategy",
    "ConfidenceThreshold",
    "BatchProcessingConfig", 
    "FrameSample",
    "VideoEvaluationJob",
    "FrameSampler",
    "ConfidenceManager",
    "BatchEvaluationProcessor",
    "create_batch_processor",
    
    # LLaVA analyzer
    "LLaVAAnalyzer",
    "create_llava_analyzer",
    
    # LLM configuration
    "LLMProvider",
    "LLMConfig",
    
    # Legacy components
    "ContentAnalyzer",
    "TechnicalAnalyzer",
    "AestheticAnalyzer", 
    "UserExperienceAnalyzer",
    "PerformanceAnalyzer",
    "ComplianceAnalyzer",
    "create_content_analyzer",
    "create_technical_analyzer",
    "create_aesthetic_analyzer",
    "create_user_experience_analyzer",
    "create_performance_analyzer",
    "create_compliance_analyzer",
    
    "GeminiVisionAnalyzer",
    "get_video_quality_orchestrator",
    
    "VideoQualityBenchmark",
    "BenchmarkResult",
    "run_benchmark_suite",
    
    "ContinuousLearningSystem",
    "FeedbackProcessor",
    "ModelUpdater",
    
    "EvaluationOrchestrator",
    "create_evaluation_orchestrator",
    
    # Utilities
    "calculate_overall_quality_score",
    "generate_quality_report", 
    "export_evaluation_results"
]
