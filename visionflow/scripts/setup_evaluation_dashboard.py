#!/usr/bin/env python3
"""
Setup script for VisionFlow Video Evaluation Dashboard.
This script:
1. Adds evaluation metrics to Prometheus
2. Imports the evaluation dashboard to Grafana
3. Sets up the monitoring infrastructure
"""

import os
import json
import requests
import subprocess
from pathlib import Path
from typing import Dict, Any, Optional
import time

# Configuration
GRAFANA_URL = os.getenv('GRAFANA_URL', 'http://localhost:30300')
GRAFANA_USER = os.getenv('GRAFANA_USER', 'admin')
GRAFANA_PASSWORD = os.getenv('GRAFANA_PASSWORD', 'admin')
DASHBOARD_FILE = Path("evaluation_dashboard.json")

class EvaluationDashboardSetup:
    """Setup class for evaluation dashboard and metrics"""
    
    def __init__(self):
        self.grafana_url = GRAFANA_URL
        self.grafana_user = GRAFANA_USER
        self.grafana_password = GRAFANA_PASSWORD
        self.session = requests.Session()
        
    def setup_prometheus_metrics(self):
        """Add evaluation metrics to Prometheus configuration"""
        print("🔧 Setting up Prometheus metrics for evaluation...")
        
        # Create metrics configuration
        metrics_config = """
# VisionFlow Video Evaluation Metrics
# Add this to your prometheus.yml or prometheus configuration

scrape_configs:
  - job_name: 'visionflow-evaluation'
    static_configs:
      - targets: ['localhost:9091']  # Your metrics endpoint
    metrics_path: '/metrics'
    scrape_interval: 15s
"""
        
        # Save metrics config
        metrics_file = Path("prometheus_evaluation_config.yml")
        with open(metrics_file, 'w') as f:
            f.write(metrics_config)
        
        print(f"✅ Prometheus metrics config saved: {metrics_file}")
        return metrics_file
    
    def add_evaluation_metrics_to_service(self):
        """Add evaluation metrics to the evaluation service"""
        print("🔧 Adding evaluation metrics to evaluation service...")
        
        # Create metrics enhancement file
        metrics_enhancement = '''# Add these metrics to your evaluation service

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
'''
        
        # Save metrics enhancement
        metrics_file = Path("evaluation_metrics_enhancement.py")
        with open(metrics_file, 'w') as f:
            f.write(metrics_enhancement)
        
        print(f"✅ Evaluation metrics enhancement saved: {metrics_file}")
        return metrics_file
    
    def test_grafana_connection(self) -> bool:
        """Test connection to Grafana"""
        try:
            response = self.session.get(
                f"{self.grafana_url}/api/health",
                timeout=10
            )
            if response.status_code == 200:
                print("✅ Grafana connection successful")
                return True
            else:
                print(f"❌ Grafana health check failed: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ Failed to connect to Grafana: {e}")
            return False
    
    def authenticate_grafana(self) -> bool:
        """Authenticate with Grafana"""
        try:
            auth_data = {
                "user": self.grafana_user,
                "password": self.grafana_password
            }
            
            response = self.session.post(
                f"{self.grafana_url}/api/login",
                json=auth_data,
                timeout=10
            )
            
            if response.status_code == 200:
                print("✅ Grafana authentication successful")
                return True
            else:
                print(f"❌ Grafana authentication failed: {response.status_code}")
                return False
                
        except Exception as e:
            print(f"❌ Authentication error: {e}")
            return False
    
    def import_evaluation_dashboard(self) -> bool:
        """Import the evaluation dashboard to Grafana"""
        try:
            if not DASHBOARD_FILE.exists():
                print(f"❌ Dashboard file not found: {DASHBOARD_FILE}")
                return False
            
            # Read dashboard JSON
            with open(DASHBOARD_FILE, 'r') as f:
                dashboard_data = json.load(f)
            
            # Prepare dashboard for import
            dashboard_data['dashboard']['id'] = None
            dashboard_data['dashboard']['uid'] = 'visionflow-evaluation'
            dashboard_data['dashboard']['title'] = 'VisionFlow Video Evaluation Dashboard'
            
            # Import dashboard
            response = self.session.post(
                f"{self.grafana_url}/api/dashboards/db",
                json=dashboard_data,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                dashboard_url = result.get('url', '')
                print(f"✅ Evaluation dashboard imported successfully!")
                print(f"   Dashboard URL: {self.grafana_url}{dashboard_url}")
                return True
            else:
                print(f"❌ Failed to import dashboard: {response.status_code}")
                print(f"   Response: {response.text}")
                return False
                
        except Exception as e:
            print(f"❌ Dashboard import error: {e}")
            return False
    
    def create_dashboard_import_job(self):
        """Create Kubernetes job to import dashboard"""
        print("🔧 Creating dashboard import job...")
        
        job_yaml = f"""apiVersion: batch/v1
kind: Job
metadata:
  name: evaluation-dashboard-import
  namespace: visionflow
spec:
  template:
    spec:
      containers:
      - name: dashboard-importer
        image: curlimages/curl:latest
        command:
        - /bin/sh
        - -c
        - |
          # Wait for Grafana to be ready
          sleep 30
          
          # Import evaluation dashboard
          curl -X POST \\
            -H "Content-Type: application/json" \\
            -d @/dashboard/evaluation_dashboard.json \\
            http://grafana:3000/api/dashboards/db \\
            -u {self.grafana_user}:{self.grafana_password}
          
          echo "Dashboard import completed"
        volumeMounts:
        - name: dashboard-volume
          mountPath: /dashboard
      volumes:
      - name: dashboard-volume
        configMap:
          name: evaluation-dashboard-config
      restartPolicy: Never
  backoffLimit: 3
---
apiVersion: v1
kind: ConfigMap
metadata:
  name: evaluation-dashboard-config
  namespace: visionflow
data:
  evaluation_dashboard.json: |
{chr(10).join('    ' + line for line in DASHBOARD_FILE.read_text().split(chr(10)))}
"""
        
        # Save job YAML
        job_file = Path("k8s/evaluation-dashboard-import-job.yaml")
        job_file.parent.mkdir(exist_ok=True)
        
        with open(job_file, 'w') as f:
            f.write(job_yaml)
        
        print(f"✅ Dashboard import job created: {job_file}")
        return job_file
    
    def setup_complete(self):
        """Complete setup summary"""
        print("\n🎉 VisionFlow Evaluation Dashboard Setup Complete!")
        print("=" * 60)
        print("\n📊 What was created:")
        print("1. ✅ Evaluation dashboard JSON")
        print("2. ✅ Prometheus metrics configuration")
        print("3. ✅ Evaluation metrics enhancement code")
        print("4. ✅ Kubernetes import job")
        
        print("\n🚀 Next steps:")
        print("1. Apply the Prometheus config to your monitoring stack")
        print("2. Add evaluation metrics to your evaluation service")
        print("3. Import the dashboard to Grafana")
        print("4. Start collecting evaluation metrics")
        
        print(f"\n📁 Files created:")
        print(f"   - Dashboard: {DASHBOARD_FILE}")
        print(f"   - Prometheus config: prometheus_evaluation_config.yml")
        print(f"   - Metrics enhancement: evaluation_metrics_enhancement.py")
        print(f"   - K8s job: k8s/evaluation-dashboard-import-job.yaml")
        
        print(f"\n🌐 Access your dashboard:")
        print(f"   Grafana: {self.grafana_url}")
        print(f"   Default credentials: {self.grafana_user}/{self.grafana_password}")
    
    def run_setup(self):
        """Run the complete setup process"""
        print("🚀 Setting up VisionFlow Video Evaluation Dashboard...")
        print("=" * 60)
        
        # Step 1: Setup Prometheus metrics
        self.setup_prometheus_metrics()
        
        # Step 2: Add evaluation metrics to service
        self.add_evaluation_metrics_to_service()
        
        # Step 3: Test Grafana connection
        if not self.test_grafana_connection():
            print("⚠️  Grafana not accessible, creating offline setup...")
            self.create_dashboard_import_job()
        else:
            # Step 4: Authenticate with Grafana
            if self.authenticate_grafana():
                # Step 5: Import dashboard
                self.import_evaluation_dashboard()
            else:
                print("⚠️  Authentication failed, creating offline setup...")
                self.create_dashboard_import_job()
        
        # Step 6: Complete setup
        self.setup_complete()

def main():
    """Main function"""
    setup = EvaluationDashboardSetup()
    setup.run_setup()

if __name__ == "__main__":
    main()
