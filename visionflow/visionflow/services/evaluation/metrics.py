"""
Evaluation Metrics Module for VisionFlow
Provides Prometheus metrics for video evaluation performance tracking.
"""

from prometheus_client import Counter, Histogram, Gauge, start_http_server
from typing import Dict, Any
import time

# Evaluation Counters
evaluation_total = Counter(
    'visionflow_evaluations_total', 
    'Total evaluations performed', 
    ['category', 'decision', 'confidence_level', 'requires_review', 'sampling_strategy']
)

evaluation_success = Counter(
    'visionflow_evaluations_success_total',
    'Total successful evaluations',
    ['category', 'decision']
)

evaluation_failure = Counter(
    'visionflow_evaluations_failure_total',
    'Total failed evaluations',
    ['category', 'error_type']
)

# Evaluation Histograms
evaluation_score = Histogram(
    'visionflow_evaluation_score', 
    'Evaluation quality scores', 
    ['category', 'dimension', 'decision']
)

evaluation_confidence = Histogram(
    'visionflow_evaluation_confidence', 
    'Evaluation confidence levels', 
    ['category', 'confidence_level']
)

evaluation_duration = Histogram(
    'visionflow_evaluation_duration_seconds', 
    'Evaluation processing time',
    ['category', 'sampling_strategy', 'frames_evaluated']
)

evaluation_dimension_score = Histogram(
    'visionflow_evaluation_dimension_score',
    'Individual dimension scores',
    ['dimension', 'category', 'decision']
)

# Evaluation Gauges
evaluation_queue_size = Gauge(
    'visionflow_evaluation_queue_size',
    'Current evaluation queue size',
    ['priority']
)

evaluation_worker_status = Gauge(
    'visionflow_evaluation_worker_status',
    'Evaluation worker status (1=active, 0=inactive)',
    ['worker_id', 'status']
)

# Performance metrics
evaluation_throughput = Gauge(
    'visionflow_evaluation_throughput_evaluations_per_second',
    'Current evaluation throughput',
    ['category']
)

# Start metrics server
def start_metrics_server(port: int = 9091):
    """Start Prometheus metrics server"""
    try:
        start_http_server(port)
        print(f"📊 Evaluation metrics server started on port {port}")
        return True
    except Exception as e:
        print(f"❌ Failed to start metrics server: {e}")
        return False

def record_evaluation_metrics(evaluation_result: Any, category: str = "unknown"):
    """Record comprehensive evaluation metrics"""
    try:
        # Extract decision and confidence info
        decision = getattr(evaluation_result, 'decision', 'unknown')
        confidence_level = getattr(evaluation_result, 'confidence_level', None)
        confidence_value = confidence_level.value if confidence_level else 'unknown'
        requires_review = getattr(evaluation_result, 'requires_human_review', False)
        sampling_strategy = getattr(evaluation_result, 'sampling_strategy', None)
        strategy_name = sampling_strategy.value if sampling_strategy else 'unknown'
        
        # Record total evaluation
        evaluation_total.labels(
            category=category,
            decision=decision,
            confidence_level=confidence_value,
            requires_review=str(requires_review),
            sampling_strategy=strategy_name
        ).inc()
        
        # Record success
        evaluation_success.labels(
            category=category,
            decision=decision
        ).inc()
        
        # Record overall score
        overall_score = getattr(evaluation_result, 'overall_score', 0.0)
        evaluation_score.labels(
            category=category,
            dimension='overall',
            decision=decision
        ).observe(overall_score)
        
        # Record confidence
        confidence = getattr(evaluation_result, 'overall_confidence', 0.0)
        evaluation_confidence.labels(
            category=category,
            confidence_level=confidence_value
        ).observe(confidence)
        
        # Record processing time
        processing_time = getattr(evaluation_result, 'evaluation_time', 0.0)
        frames_evaluated = getattr(evaluation_result, 'frames_evaluated', 0)
        evaluation_duration.labels(
            category=category,
            sampling_strategy=strategy_name,
            frames_evaluated=str(frames_evaluated)
        ).observe(processing_time)
        
        # Record dimension scores if available
        dimension_scores = getattr(evaluation_result, 'dimension_scores', [])
        if hasattr(dimension_scores, '__iter__'):
            for dim_score in dimension_scores:
                if hasattr(dim_score, 'dimension') and hasattr(dim_score, 'score'):
                    dimension_name = dim_score.dimension.value if hasattr(dim_score.dimension, 'value') else str(dim_score.dimension)
                    score_value = dim_score.score
                    evaluation_dimension_score.labels(
                        dimension=dimension_name,
                        category=category,
                        decision=decision
                    ).observe(score_value)
        
        print(f"✅ Recorded metrics for evaluation: {category} - {decision}")
        
    except Exception as e:
        print(f"❌ Failed to record evaluation metrics: {e}")
        # Record failure
        evaluation_failure.labels(
            category=category,
            error_type="metrics_recording_error"
        ).inc()

def record_evaluation_failure(category: str, error_type: str, error_message: str = ""):
    """Record evaluation failure metrics"""
    evaluation_failure.labels(
        category=category,
        error_type=error_type
    ).inc()
    print(f"❌ Recorded failure metrics: {category} - {error_type}")

def update_queue_metrics(queue_size: int, priority: str = "normal"):
    """Update evaluation queue metrics"""
    evaluation_queue_size.labels(priority=priority).set(queue_size)

def update_worker_status(worker_id: str, status: str):
    """Update evaluation worker status"""
    status_value = 1 if status == "active" else 0
    evaluation_worker_status.labels(
        worker_id=worker_id,
        status=status
    ).set(status_value)

def update_throughput_metrics(category: str, evaluations_per_second: float):
    """Update evaluation throughput metrics"""
    evaluation_throughput.labels(category=category).set(evaluations_per_second)

# Convenience functions for common metrics
def increment_evaluation_counter(category: str, decision: str):
    """Increment evaluation counter"""
    evaluation_total.labels(
        category=category,
        decision=decision,
        confidence_level="unknown",
        requires_review="false",
        sampling_strategy="unknown"
    ).inc()

def observe_evaluation_score(category: str, score: float, decision: str):
    """Observe evaluation score"""
    evaluation_score.labels(
        category=category,
        dimension='overall',
        decision=decision
    ).observe(score)

def observe_evaluation_duration(category: str, duration: float, strategy: str, frames: int):
    """Observe evaluation duration"""
    evaluation_duration.labels(
        category=category,
        sampling_strategy=strategy,
        frames_evaluated=str(frames)
    ).observe(duration)
