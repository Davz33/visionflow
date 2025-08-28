#!/usr/bin/env python3
"""
Integrate evaluation metrics into the existing evaluation service.
This script adds Prometheus metrics to track evaluation performance.
"""

import os
import re
from pathlib import Path
from typing import List, Dict, Any

class EvaluationMetricsIntegrator:
    """Integrate Prometheus metrics into evaluation services"""
    
    def __init__(self):
        self.project_root = Path.cwd()
        self.evaluation_services = [
            "visionflow/services/evaluation/video_evaluation_orchestrator.py",
            "visionflow/services/evaluation/production_orchestrator.py",
            "visionflow/services/evaluation/quality_metrics.py"
        ]
        
    def create_metrics_module(self):
        """Create a dedicated metrics module for evaluation"""
        print("🔧 Creating evaluation metrics module...")
        
        metrics_content = '''"""
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
'''
        
        # Save metrics module
        metrics_file = self.project_root / "visionflow" / "services" / "evaluation" / "metrics.py"
        metrics_file.parent.mkdir(exist_ok=True)
        
        with open(metrics_file, 'w') as f:
            f.write(metrics_content)
        
        print(f"✅ Evaluation metrics module created: {metrics_file}")
        return metrics_file
    
    def integrate_metrics_into_orchestrator(self):
        """Integrate metrics into the video evaluation orchestrator"""
        print("🔧 Integrating metrics into video evaluation orchestrator...")
        
        orchestrator_file = self.project_root / "visionflow" / "services" / "evaluation" / "video_evaluation_orchestrator.py"
        
        if not orchestrator_file.exists():
            print(f"⚠️  Orchestrator file not found: {orchestrator_file}")
            return False
        
        # Read the file
        with open(orchestrator_file, 'r') as f:
            content = f.read()
        
        # Add metrics import
        if "from .metrics import" not in content:
            import_statement = "from .metrics import record_evaluation_metrics, record_evaluation_failure"
            content = content.replace(
                "from ...shared.monitoring import get_logger",
                "from ...shared.monitoring import get_logger\nfrom .metrics import record_evaluation_metrics, record_evaluation_failure"
            )
        
        # Add metrics recording in evaluate_video method
        if "record_evaluation_metrics" not in content:
            # Find the end of evaluate_video method
            method_pattern = r'async def evaluate_video\(self,.*?return result'
            match = re.search(method_pattern, content, re.DOTALL)
            
            if match:
                # Add metrics recording before return
                new_content = content.replace(
                    "return result",
                    """        # Record evaluation metrics
        try:
            record_evaluation_metrics(result, category='video')
        except Exception as e:
            logger.warning(f"Failed to record evaluation metrics: {e}")
        
        return result"""
                )
                content = new_content
        
        # Save updated file
        with open(orchestrator_file, 'w') as f:
            f.write(content)
        
        print(f"✅ Metrics integrated into orchestrator: {orchestrator_file}")
        return True
    
    def create_metrics_configuration(self):
        """Create configuration for metrics collection"""
        print("🔧 Creating metrics configuration...")
        
        config_content = '''# Evaluation Metrics Configuration
# Add this to your environment or config file

# Metrics Server Configuration
EVALUATION_METRICS_ENABLED=true
EVALUATION_METRICS_PORT=9091
EVALUATION_METRICS_HOST=0.0.0.0

# Metrics Collection Settings
EVALUATION_METRICS_INTERVAL=15s
EVALUATION_METRICS_RETENTION=24h

# Prometheus Scrape Configuration
# Add this to your prometheus.yml:

scrape_configs:
  - job_name: 'visionflow-evaluation'
    static_configs:
      - targets: ['localhost:9091']
    metrics_path: '/metrics'
    scrape_interval: 15s
    scrape_timeout: 10s
    
    # Relabeling rules for better metric names
    relabel_configs:
      - source_labels: [__address__]
        target_label: instance
        regex: '([^:]+)(?::\\d+)?'
        replacement: '${1}'
      
      - source_labels: [__name__]
        target_label: metric_name
        regex: 'visionflow_(.+)'
        replacement: '${1}'

# Grafana Dashboard Variables
# These will be available in your evaluation dashboard:
# - $category: Video category filter
# - $decision: Evaluation decision filter  
# - $confidence_level: Confidence level filter
# - $sampling_strategy: Sampling strategy filter
'''
        
        # Save config
        config_file = self.project_root / "evaluation_metrics_config.yml"
        with open(config_file, 'w') as f:
            f.write(config_content)
        
        print(f"✅ Metrics configuration created: {config_file}")
        return config_file
    
    def create_metrics_test_script(self):
        """Create a test script for metrics"""
        print("🔧 Creating metrics test script...")
        
        test_content = '''#!/usr/bin/env python3
"""
Test script for evaluation metrics.
Run this to verify metrics are working correctly.
"""

import time
import random
from visionflow.services.evaluation.metrics import (
    record_evaluation_metrics, start_metrics_server,
    increment_evaluation_counter, observe_evaluation_score
)

def test_evaluation_metrics():
    """Test evaluation metrics functionality"""
    print("🧪 Testing evaluation metrics...")
    
    # Start metrics server
    if start_metrics_server(port=9091):
        print("✅ Metrics server started")
    else:
        print("❌ Failed to start metrics server")
        return
    
    # Simulate some evaluation metrics
    categories = ["nature", "action", "object", "human_activity"]
    decisions = ["auto_approve", "flag_monitoring", "queue_review"]
    
    for i in range(10):
        category = random.choice(categories)
        decision = random.choice(decisions)
        score = random.uniform(0.3, 0.9)
        
        # Record metrics
        increment_evaluation_counter(category, decision)
        observe_evaluation_score(category, score, decision)
        
        print(f"   Recorded: {category} - {decision} - {score:.3f}")
        time.sleep(1)
    
    print("✅ Metrics test completed!")
    print("📊 Check metrics at: http://localhost:9091/metrics")

if __name__ == "__main__":
    test_evaluation_metrics()
'''
        
        # Save test script
        test_file = self.project_root / "test_evaluation_metrics.py"
        with open(test_file, 'w') as f:
            f.write(test_content)
        
        # Make executable
        test_file.chmod(0o755)
        
        print(f"✅ Metrics test script created: {test_file}")
        return test_file
    
    def run_integration(self):
        """Run the complete integration process"""
        print("🚀 Integrating evaluation metrics into VisionFlow...")
        print("=" * 60)
        
        # Step 1: Create metrics module
        metrics_module = self.create_metrics_module()
        
        # Step 2: Integrate into orchestrator
        self.integrate_metrics_into_orchestrator()
        
        # Step 3: Create configuration
        config_file = self.create_metrics_configuration()
        
        # Step 4: Create test script
        test_script = self.create_metrics_test_script()
        
        # Step 5: Integration complete
        print("\n🎉 Evaluation Metrics Integration Complete!")
        print("=" * 60)
        print("\n📊 What was created:")
        print("1. ✅ Evaluation metrics module")
        print("2. ✅ Metrics integration in orchestrator")
        print("3. ✅ Metrics configuration")
        print("4. ✅ Metrics test script")
        
        print("\n🚀 Next steps:")
        print("1. Start your evaluation service")
        print("2. Run the metrics test script")
        print("3. Check metrics at http://localhost:9091/metrics")
        print("4. Import the evaluation dashboard to Grafana")
        print("5. Start collecting real evaluation metrics")
        
        print(f"\n📁 Files created:")
        print(f"   - Metrics module: {metrics_module}")
        print(f"   - Configuration: {config_file}")
        print(f"   - Test script: {test_script}")
        
        print("\n💡 Usage:")
        print("   - Metrics are automatically recorded during evaluation")
        print("   - Access metrics at /metrics endpoint")
        print("   - Prometheus will scrape metrics every 15 seconds")
        print("   - Grafana dashboard will show real-time data")

def main():
    """Main function"""
    integrator = EvaluationMetricsIntegrator()
    integrator.run_integration()

if __name__ == "__main__":
    main()
