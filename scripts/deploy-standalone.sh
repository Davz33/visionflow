#!/bin/bash

# Deploy standalone VisionFlow services to local Kubernetes cluster

set -e

echo "🚀 Deploying VisionFlow Standalone Services to local Kubernetes..."

# Set up paths
SCRIPT_DIR="$(dirname "$0")"
PROJECT_DIR="$SCRIPT_DIR/.."
K8S_DIR="$PROJECT_DIR/k8s/local/standalone"

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
echo "🔨 Building Docker image..."
bash "$SCRIPT_DIR/build-standalone-images.sh"

# Check if using kind and load image
if kubectl config current-context | grep -q "kind"; then
    echo "🐋 Loading image into kind cluster..."
    kind load docker-image visionflow-generation:local --name visionflow
fi

# Deploy generation service
echo "📦 Deploying Generation Service..."
kubectl apply -f "$K8S_DIR/generation-service.yaml"

# Wait for deployment to be ready
echo "⏳ Waiting for generation service to be ready..."
kubectl wait --for=condition=available --timeout=300s deployment/generation-service -n visionflow-generation

# Show status
echo "✅ Generation service deployment complete!"
echo ""
echo "📊 Generation service status:"
kubectl get pods -n visionflow-generation

echo ""
echo "🌐 Generation service endpoint:"
echo "   http://localhost:30002/health"

echo ""
echo "📝 Useful commands:"
echo "   View logs: kubectl logs -f deployment/generation-service -n visionflow-generation"
echo "   Test health: curl http://localhost:30002/health"
echo "   Delete: kubectl delete namespace visionflow-generation"
