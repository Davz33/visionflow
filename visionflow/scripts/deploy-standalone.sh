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
echo "🔨 Building Docker images..."
bash "$SCRIPT_DIR/build-standalone-images.sh"

# Check if using kind and load images
if kubectl config current-context | grep -q "kind"; then
    echo "🐋 Loading images into kind cluster..."
    kind load docker-image visionflow-generation-health:local --name visionflow
    kind load docker-image visionflow-generation:local --name visionflow
fi

# Deploy health service first (recommended for testing)
echo "📦 Deploying Generation Health Service..."
kubectl apply -f "$K8S_DIR/generation-health-service.yaml"

# Wait for health service to be ready
echo "⏳ Waiting for health service to be ready..."
kubectl wait --for=condition=available --timeout=300s deployment/generation-health-service -n visionflow-generation-health

# Show status
echo "✅ Health service deployment complete!"
echo ""
echo "📊 Health service status:"
kubectl get pods -n visionflow-generation-health

echo ""
echo "🌐 Health service endpoint:"
echo "   Generation Health: http://localhost:30002/health"

echo ""
echo "💡 To deploy the full generation service (with ML):"
echo "   kubectl apply -f k8s/local/standalone/generation-service.yaml"

echo ""
echo "📝 Useful commands:"
echo "   View logs: kubectl logs -f deployment/generation-health-service -n visionflow-generation-health"
echo "   Test health: curl http://localhost:30002/health"
echo "   Delete: kubectl delete namespace visionflow-generation-health"
