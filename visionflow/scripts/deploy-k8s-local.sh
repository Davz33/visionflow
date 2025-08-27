#!/bin/bash

# Deploy VisionFlow to local Kubernetes cluster

set -e

echo "🚀 Deploying VisionFlow to local Kubernetes..."

# Set up paths
SCRIPT_DIR="$(dirname "$0")"
PROJECT_DIR="$SCRIPT_DIR/.."
K8S_DIR="$PROJECT_DIR/k8s"

# Check if kubectl is available
if ! command -v kubectl &> /dev/null; then
    echo "❌ kubectl is not installed. Please install kubectl first."
    exit 1
fi

# Check if local cluster is running
if ! kubectl cluster-info &> /dev/null; then
    echo "❌ No Kubernetes cluster is running."
    echo "💡 For local development, you can use:"
    echo "   - kind: kind create cluster --config k8s/local/kind-config.yaml"
    echo "   - minikube: minikube start"
    echo "   - Docker Desktop Kubernetes: Enable in settings"
    exit 1
fi

# Build images first
echo "🔨 Building Docker images..."
bash "$SCRIPT_DIR/build-images.sh"

# Check if using kind and load images
if kubectl config current-context | grep -q "kind"; then
    echo "🐋 Loading images into kind cluster..."
    kind load docker-image visionflow-api:local
    kind load docker-image visionflow-generation:local
    kind load docker-image visionflow-orchestrator:local
fi

# Apply Kubernetes manifests
echo "📦 Applying Kubernetes manifests..."
kubectl apply -f "$K8S_DIR/local/complete-local-deployment.yaml"

# Wait for deployments to be ready
echo "⏳ Waiting for deployments to be ready..."
kubectl wait --for=condition=available --timeout=300s deployment --all -n visionflow-local

# Show status
echo "✅ Deployment complete!"
echo ""
echo "📊 Deployment status:"
kubectl get pods -n visionflow-local

echo ""
echo "🌐 Service endpoints:"
echo "   API Gateway:    http://localhost:30000"
echo "   Generation:     http://localhost:30002"  
echo "   Orchestration:  http://localhost:30001"
echo "   Prometheus:     http://localhost:30090"
echo "   Grafana:        http://localhost:30300 (admin/admin)"
echo "   MinIO Console:  http://localhost:30901"

echo ""
echo "📝 Useful commands:"
echo "   View logs: kubectl logs -f deployment/api-gateway -n visionflow-local"
echo "   Scale:     kubectl scale deployment api-gateway --replicas=2 -n visionflow-local" 
echo "   Delete:    kubectl delete namespace visionflow-local"