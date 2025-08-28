# Add these metrics to your evaluation service

from prometheus_client import Counter, Histogram, Gauge, start_http_server
import time

# Evaluation metrics
evaluation_total = Counter(
    'visionflow_evaluations_total', 
    'Total evaluations', 
    ['category', 'decision', 'confidence_level', 'requires_review']
)

evaluation_score = Histogram(
    'visionflow_evaluation_score', 
    'Evaluation scores', 
    ['category', 'dimension']
)

evaluation_confidence = Histogram(
    'visionflow_evaluation_confidence', 
    'Evaluation confidence', 
    ['category']
)

evaluation_duration = Histogram(
    'visionflow_evaluation_duration_seconds', 
    'Evaluation processing time',
    ['category']
)

evaluation_dimension_score = Histogram(
    'visionflow_evaluation_dimension_score',
    'Individual dimension scores',
    ['dimension', 'category']
)

# Start metrics server
def start_metrics_server(port: int = 9091):
    """Start Prometheus metrics server"""
    start_http_server(port)
    print(f"📊 Metrics server started on port {port}")

# Usage in evaluation service:
def record_evaluation_metrics(evaluation_result, category: str):
    """Record evaluation metrics"""
    # Record total evaluation
    evaluation_total.labels(
        category=category,
        decision=evaluation_result.decision,
        confidence_level=evaluation_result.confidence_level.value,
        requires_review=str(evaluation_result.requires_human_review)
    ).inc()
    
    # Record overall score
    evaluation_score.labels(
        category=category,
        dimension='overall'
    ).observe(evaluation_result.overall_score)
    
    # Record confidence
    evaluation_confidence.labels(category=category).observe(
        evaluation_result.overall_confidence
    )
    
    # Record processing time
    evaluation_duration.labels(category=category).observe(
        evaluation_result.evaluation_time
    )
    
    # Record dimension scores
    for dim_score in evaluation_result.dimension_scores:
        evaluation_dimension_score.labels(
            dimension=dim_score.dimension.value,
            category=category
        ).observe(dim_score.score)
