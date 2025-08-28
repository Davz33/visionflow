#!/bin/bash

# VisionFlow Dashboards Deployment Script
# This script builds and deploys the Flutter web application to Kubernetes

set -e

echo "🚀 Starting VisionFlow Dashboards deployment..."

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "❌ Docker is not running. Please start Docker and try again."
    exit 1
fi

# Check if kubectl is available
if ! command -v kubectl &> /dev/null; then
    echo "❌ kubectl is not installed. Please install kubectl and try again."
    exit 1
fi

# Build the Docker image
echo "🔨 Building Docker image..."
docker build -t visionflow-dashboards:latest .

if [ $? -eq 0 ]; then
    echo "✅ Docker image built successfully"
else
    echo "❌ Docker build failed"
    exit 1
fi

# Deploy to Kubernetes
echo "🚀 Deploying to Kubernetes..."

# Apply the deployment
kubectl apply -f k8s/dashboards-deployment.yaml

if [ $? -eq 0 ]; then
    echo "✅ Kubernetes deployment applied successfully"
else
    echo "❌ Kubernetes deployment failed"
    exit 1
fi

# Wait for deployment to be ready
echo "⏳ Waiting for deployment to be ready..."
kubectl wait --for=condition=available --timeout=300s deployment/visionflow-dashboards -n visionflow-local

if [ $? -eq 0 ]; then
    echo "✅ Deployment is ready"
else
    echo "❌ Deployment failed to become ready"
    exit 1
fi

# Get service information
echo "📊 Service Information:"
kubectl get service visionflow-dashboards-service -n visionflow-local

echo ""
echo "🎉 Deployment completed successfully!"
echo ""
echo "🌐 Access your dashboards:"
echo "   - Local: http://localhost:30500"
echo "   - Ingress: http://dashboards.visionflow.local (if configured)"
echo ""
echo "📋 Useful commands:"
echo "   - View logs: kubectl logs -f deployment/visionflow-dashboards -n visionflow-local"
echo "   - Check status: kubectl get pods -n visionflow-local | grep dashboards"
echo "   - Scale: kubectl scale deployment visionflow-dashboards --replicas=3 -n visionflow-local"
