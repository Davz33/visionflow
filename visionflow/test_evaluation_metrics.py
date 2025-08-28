#!/usr/bin/env python3
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
