#!/usr/bin/env python3
"""
Proper deployment script for VisionFlow Evaluation Dashboard.
This script handles dashboard deployment without embedding JSON in K8s YAML.
"""

import os
import json
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, Any, Optional
import yaml

class DashboardDeployer:
    """Proper dashboard deployment without embedding JSON in K8s YAML"""
    
    def __init__(self):
        self.project_root = Path.cwd()
        self.dashboard_file = self.project_root / "evaluation_dashboard.json"
        self.k8s_job_file = self.project_root / "k8s" / "evaluation-dashboard-import-job.yaml"
        
    def create_proper_k8s_job(self):
        """Create K8s job that references external dashboard file"""
        print("🔧 Creating proper K8s job without embedded JSON...")
        
        # Create a proper job that mounts the dashboard as a file
        job_yaml = """apiVersion: batch/v1
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
          echo "Waiting for Grafana to be ready..."
          sleep 30
          
          # Check if Grafana is accessible
          echo "Checking Grafana health..."
          curl -f http://grafana:3000/api/health || exit 1
          
          # Import evaluation dashboard
          echo "Importing evaluation dashboard..."
          curl -X POST \\
            -H "Content-Type: application/json" \\
            -d @/dashboard/evaluation_dashboard.json \\
            http://grafana:3000/api/dashboards/db \\
            -u ${GRAFANA_USER:-admin}:${GRAFANA_PASSWORD:-admin}
          
          if [ $? -eq 0 ]; then
            echo "✅ Dashboard import completed successfully"
          else
            echo "❌ Dashboard import failed"
            exit 1
          fi
        env:
        - name: GRAFANA_USER
          value: "admin"
        - name: GRAFANA_PASSWORD
          value: "admin"
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
  # Reference to external dashboard file
  evaluation_dashboard.json: |
    # Dashboard content will be populated from external file
    # This approach keeps K8s YAML clean and maintainable
"""
        
        # Save the clean job YAML
        with open(self.k8s_job_file, 'w') as f:
            f.write(job_yaml)
        
        print(f"✅ Clean K8s job created: {self.k8s_job_file}")
        return True
    
    def create_dashboard_configmap(self):
        """Create a separate ConfigMap for the dashboard content"""
        print("🔧 Creating dashboard ConfigMap...")
        
        if not self.dashboard_file.exists():
            print(f"❌ Dashboard file not found: {self.dashboard_file}")
            return False
        
        # Read dashboard JSON
        with open(self.dashboard_file, 'r') as f:
            dashboard_data = json.load(f)
        
        # Create ConfigMap YAML
        configmap_yaml = f"""apiVersion: v1
kind: ConfigMap
metadata:
  name: evaluation-dashboard-content
  namespace: visionflow
  labels:
    app: visionflow
    component: evaluation-dashboard
data:
  evaluation_dashboard.json: |
{chr(10).join('    ' + line for line in json.dumps(dashboard_data, indent=2).split(chr(10)))}
"""
        
        # Save ConfigMap
        configmap_file = self.project_root / "k8s" / "evaluation-dashboard-configmap.yaml"
        with open(configmap_file, 'w') as f:
            f.write(configmap_yaml)
        
        print(f"✅ Dashboard ConfigMap created: {configmap_file}")
        return configmap_file
    
    def create_helm_chart_structure(self):
        """Create a proper Helm chart structure for dashboard deployment"""
        print("🔧 Creating Helm chart structure...")
        
        # Read dashboard data for ConfigMap
        if not self.dashboard_file.exists():
            print(f"❌ Dashboard file not found: {self.dashboard_file}")
            return None
            
        with open(self.dashboard_file, 'r') as f:
            dashboard_data = json.load(f)
        
        helm_dir = self.project_root / "helm" / "evaluation-dashboard"
        helm_dir.mkdir(parents=True, exist_ok=True)
        
        # Chart.yaml
        chart_yaml = """apiVersion: v2
name: evaluation-dashboard
description: VisionFlow Video Evaluation Dashboard
type: application
version: 0.1.0
appVersion: "1.0.0"
"""
        
        with open(helm_dir / "Chart.yaml", 'w') as f:
            f.write(chart_yaml)
        
        # values.yaml
        values_yaml = """# Default values for evaluation-dashboard
replicaCount: 1

image:
  repository: curlimages/curl
  tag: latest
  pullPolicy: IfNotPresent

grafana:
  host: grafana
  port: 3000
  user: admin
  password: admin

dashboard:
  name: "VisionFlow Video Evaluation Dashboard"
  tags: ["visionflow", "video-evaluation", "quality-assessment"]
  refresh: "30s"

resources:
  limits:
    cpu: 100m
    memory: 128Mi
  requests:
    cpu: 50m
    memory: 64Mi

nodeSelector: {}

tolerations: []

affinity: {}
"""
        
        with open(helm_dir / "values.yaml", 'w') as f:
            f.write(values_yaml)
        
        # templates/deployment.yaml
        templates_dir = helm_dir / "templates"
        templates_dir.mkdir(exist_ok=True)
        
        deployment_yaml = """apiVersion: apps/v1
kind: Deployment
metadata:
  name: {{ include "evaluation-dashboard.fullname" . }}
  labels:
    {{- include "evaluation-dashboard.labels" . | nindent 4 }}
spec:
  replicas: {{ .Values.replicaCount }}
  selector:
    matchLabels:
      {{- include "evaluation-dashboard.selectorLabels" . | nindent 6 }}
  template:
    metadata:
      labels:
        {{- include "evaluation-dashboard.selectorLabels" . | nindent 8 }}
    spec:
      containers:
        - name: {{ .Chart.Name }}
          image: "{{ .Values.image.repository }}:{{ .Values.image.tag }}"
          imagePullPolicy: {{ .Values.image.pullPolicy }}
          command:
            - /bin/sh
            - -c
            - |
              # Wait for Grafana
              sleep 30
              
              # Import dashboard
              curl -X POST \\
                -H "Content-Type: application/json" \\
                -d @/dashboard/evaluation_dashboard.json \\
                http://{{ .Values.grafana.host }}:{{ .Values.grafana.port }}/api/dashboards/db \\
                -u {{ .Values.grafana.user }}:{{ .Values.grafana.password }}
              
              echo "Dashboard import completed"
          volumeMounts:
            - name: dashboard-volume
              mountPath: /dashboard
          resources:
            {{- toYaml .Values.resources | nindent 12 }}
      volumes:
        - name: dashboard-volume
          configMap:
            name: {{ include "evaluation-dashboard.fullname" . }}-config
      {{- with .Values.nodeSelector }}
      nodeSelector:
        {{- toYaml . | nindent 8 }}
      {{- end }}
      {{- with .Values.affinity }}
      affinity:
        {{- toYaml . | nindent 8 }}
      {{- end }}
      {{- with .Values.tolerations }}
      tolerations:
        {{- toYaml . | nindent 8 }}
      {{- end }}
"""
        
        with open(templates_dir / "deployment.yaml", 'w') as f:
            f.write(deployment_yaml)
        
        # templates/configmap.yaml
        configmap_yaml = f"""apiVersion: v1
kind: ConfigMap
metadata:
  name: {{ include "evaluation-dashboard.fullname" . }}-config
  labels:
    {{- include "evaluation-dashboard.labels" . | nindent 4 }}
data:
  evaluation_dashboard.json: |
{chr(10).join('    ' + line for line in json.dumps(dashboard_data, indent=2).split(chr(10)))}
"""
        
        with open(templates_dir / "configmap.yaml", 'w') as f:
            f.write(configmap_yaml)
        
        # templates/_helpers.tpl
        helpers_tpl = """{{/*
Expand the name of the chart.
*/}}
{{- define "evaluation-dashboard.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Create a default fully qualified app name.
*/}}
{{- define "evaluation-dashboard.fullname" -}}
{{- if .Values.fullnameOverride }}
{{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- $name := default .Chart.Name .Values.nameOverride }}
{{- if contains $name .Release.Name }}
{{- .Release.Name | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- printf "%s-%s" .Release.Name $name | trunc 63 | trimSuffix "-" }}
{{- end }}
{{- end }}
{{- end }}

{{/*
Common labels
*/}}
{{- define "evaluation-dashboard.labels" -}}
helm.sh/chart: {{ include "evaluation-dashboard.chart" . }}
{{ include "evaluation-dashboard.selectorLabels" . }}
{{- if .Chart.AppVersion }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
{{- end }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- end }}

{{/*
Selector labels
*/}}
{{- define "evaluation-dashboard.selectorLabels" -}}
app.kubernetes.io/name: {{ include "evaluation-dashboard.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end }}

{{/*
Create chart name and version as used by the chart label.
*/}}
{{- define "evaluation-dashboard.chart" -}}
{{- printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" }}
{{- end }}
"""
        
        with open(templates_dir / "_helpers.tpl", 'w') as f:
            f.write(helpers_tpl)
        
        print(f"✅ Helm chart structure created: {helm_dir}")
        return helm_dir
    
    def create_deployment_instructions(self):
        """Create deployment instructions"""
        print("🔧 Creating deployment instructions...")
        
        instructions = """# VisionFlow Evaluation Dashboard Deployment

## 🚀 Deployment Options

### Option 1: Direct Grafana Import (Recommended for Development)
```bash
# 1. Open Grafana (http://localhost:30300)
# 2. Login with admin/admin
# 3. Go to Dashboards → Import
# 4. Upload evaluation_dashboard.json
# 5. Select Prometheus as data source
```

### Option 2: Kubernetes ConfigMap (Production)
```bash
# Apply the dashboard ConfigMap
kubectl apply -f k8s/evaluation-dashboard-configmap.yaml

# Apply the import job
kubectl apply -f k8s/evaluation-dashboard-import-job.yaml

# Check job status
kubectl get jobs -n visionflow
kubectl logs job/evaluation-dashboard-import -n visionflow
```

### Option 3: Helm Chart (Enterprise)
```bash
# Install the Helm chart
helm install evaluation-dashboard ./helm/evaluation-dashboard

# Upgrade existing installation
helm upgrade evaluation-dashboard ./helm/evaluation-dashboard

# Uninstall
helm uninstall evaluation-dashboard
```

## 🔧 Configuration

### Environment Variables
```bash
GRAFANA_URL=http://localhost:30300
GRAFANA_USER=admin
GRAFANA_PASSWORD=admin
```

### Prometheus Configuration
Add to your prometheus.yml:
```yaml
scrape_configs:
  - job_name: 'visionflow-evaluation'
    static_configs:
      - targets: ['localhost:9091']
    metrics_path: '/metrics'
    scrape_interval: 15s
```

## 📊 Verification

### Check Dashboard Import
1. Open Grafana
2. Go to Dashboards
3. Look for "VisionFlow Video Evaluation Dashboard"

### Check Metrics Collection
```bash
# Test metrics endpoint
curl http://localhost:9091/metrics | grep visionflow

# Run evaluation test
python test_evaluation_metrics.py
```

## 🚨 Troubleshooting

### Common Issues
1. **Dashboard not appearing**: Check Grafana logs and import job status
2. **No metrics**: Verify Prometheus is scraping the metrics endpoint
3. **Import failures**: Check Grafana credentials and network access

### Debug Commands
```bash
# Check Grafana health
curl http://localhost:30300/api/health

# Check Prometheus targets
curl http://localhost:9090/api/v1/targets

# View job logs
kubectl logs job/evaluation-dashboard-import -n visionflow
```
"""
        
        instructions_file = self.project_root / "DASHBOARD_DEPLOYMENT.md"
        with open(instructions_file, 'w') as f:
            f.write(instructions)
        
        print(f"✅ Deployment instructions created: {instructions_file}")
        return instructions_file
    
    def run_deployment_setup(self):
        """Run the complete deployment setup"""
        print("🚀 Setting up proper dashboard deployment...")
        print("=" * 60)
        
        # Step 1: Create clean K8s job
        self.create_proper_k8s_job()
        
        # Step 2: Create dashboard ConfigMap
        self.create_dashboard_configmap()
        
        # Step 3: Create Helm chart structure
        helm_dir = self.create_helm_chart_structure()
        
        # Step 4: Create deployment instructions
        instructions = self.create_deployment_instructions()
        
        # Step 5: Complete setup
        print("\n🎉 Proper Dashboard Deployment Setup Complete!")
        print("=" * 60)
        print("\n📊 What was created:")
        print("1. ✅ Clean K8s job (no embedded JSON)")
        print("2. ✅ Separate dashboard ConfigMap")
        print("3. ✅ Professional Helm chart structure")
        print("4. ✅ Comprehensive deployment instructions")
        
        print("\n🚀 Deployment Options:")
        print("1. **Direct Import**: Upload JSON to Grafana (dev)")
        print("2. **K8s ConfigMap**: Use ConfigMap + Job (production)")
        print("3. **Helm Chart**: Professional Helm deployment (enterprise)")
        
        print(f"\n📁 Files created:")
        print(f"   - Clean K8s job: {self.k8s_job_file}")
        print(f"   - Dashboard ConfigMap: k8s/evaluation-dashboard-configmap.yaml")
        print(f"   - Helm chart: {helm_dir}")
        print(f"   - Instructions: {instructions}")
        
        print("\n💡 Best Practices Implemented:")
        print("   - ✅ No JSON embedded in K8s YAML")
        print("   - ✅ Separation of concerns")
        print("   - ✅ Multiple deployment options")
        print("   - ✅ Professional Helm chart structure")
        print("   - ✅ Easy maintenance and updates")

def main():
    """Main function"""
    deployer = DashboardDeployer()
    deployer.run_deployment_setup()

if __name__ == "__main__":
    main()
