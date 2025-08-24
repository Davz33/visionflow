"""
Video evaluation services for quality assessment and analysis.

This package provides comprehensive video evaluation capabilities including:
- Multi-dimensional evaluation framework (6 dimensions)
- Confidence management with automated flagging  
- Score aggregation with ensemble methods
- Industry-standard metrics integration
- Continuous learning and fine-tuning triggers
"""

# Only import the new evaluation modules to avoid dependency issues
__all__ = [
    "VideoEvaluationOrchestrator",
    "SamplingStrategy", 
    "EvaluationDimension",
    "ConfidenceLevel",
    "ConfidenceManager",
    "ScoreAggregator",
    "IndustryMetrics"
]