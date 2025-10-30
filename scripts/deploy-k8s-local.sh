#!/bin/bash
# Deploy VisionFlow to local Kubernetes kind cluster
# run via: cd visionflow && source scripts/deploy-k8s-local.sh

echo "🚀 Deploying VisionFlow to local kind cluster"

# Check if Docker engine is running
if ! docker info > /dev/null 2>&1; then
    echo "❌ Docker engine is not running. Starting Docker engine..."

    if [[ "$OSTYPE" == "darwin"* ]]; then
        open -a Docker --background
    elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
        sudo systemctl start docker
    elif [[ "$OSTYPE" == "cygwin"* ]] || [[ "$OSTYPE" == "msys"* ]] || [[ "$OSTYPE" == "win32"* ]]; then
        cmd.exe /C "start \"\" \"C:\Program Files\Docker\Docker\Docker Desktop.exe\""
    else
        echo "❌ Unsupported operating system. Please start Docker engine manually."
        exit 1
    fi
fi

# Check if kind is installed
if ! command -v kind &> /dev/null; then
    echo "❌ kind is not installed. Please install kind first."
    echo "💡 To install kind, please refer to the documentation: https://kind.sigs.k8s.io/docs/user/quick-start/"
    exit 1
fi
# Check if kubectl is available
if ! command -v kubectl &> /dev/null; then
    echo "❌ kubectl is not installed. Please install kubectl first."
    echo "💡 To install kubectl, please refer to the documentation: https://kubernetes.io/docs/tasks/tools/install-kubectl/"
    exit 1
fi


# Check if local cluster is running
if ! kubectl cluster-info &> /dev/null; then
    echo "❌ No Kubernetes cluster is running."
    echo "Starting kind cluster..."
    kind create cluster --name visionflow --config k8s/local/kind-config.yaml &
    
    # Wait for cluster to be ready
    echo "⏳ Waiting for cluster to be ready..."
    kubectl wait --for=condition=Ready nodes --all --timeout=10s
fi

# Build images first, if not already built
if ! docker images | grep -q "visionflow-generation:local"; then
    echo "🔨 Building Docker images..."
    docker build -t visionflow-generation:local -f docker/Dockerfile.generation.local .
fi

# Check if using kind and load images
if kubectl config current-context | grep -q "kind"; then
    echo "🐋 Loading images into kind cluster..."
    kind load docker-image visionflow-generation:local --name visionflow
fi

# Apply Kubernetes manifests
echo "📦 Applying Kubernetes manifests..."
kubectl apply -f k8s/local/standalone/generation-service.yaml

# Wait for deployments to be ready
echo "⏳ Waiting for deployments to be ready..."
echo "To monitor the deployment, you can use the following command:"
echo "kubectl get pods -n visionflow-generation"

echo "📝 Useful commands:"
echo "   View logs: kubectl logs -n visionflow-generation -l app=generation-service -f"
echo "   Run command in container: kubectl exec -it -n visionflow-generation <pod-name>-- bash -c '...'"
