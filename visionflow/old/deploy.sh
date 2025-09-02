#!/bin/bash

# Deployment script for VisionFlow on GKE
# This script builds and deploys the application

set -e

# Configuration
PROJECT_ID=${PROJECT_ID:-"visionflow-gcp-project"}
REGION=${REGION:-"us-central1"}
ZONE=${ZONE:-"us-central1-a"}
CLUSTER_NAME=${CLUSTER_NAME:-"visionflow-cluster"}
NAMESPACE=${NAMESPACE:-"visionflow"}

# Image names
IMAGE_API="gcr.io/$PROJECT_ID/visionflow-api"
IMAGE_ORCHESTRATOR="gcr.io/$PROJECT_ID/visionflow-orchestrator"
IMAGE_GENERATION="gcr.io/$PROJECT_ID/visionflow-generation"

# Get git commit hash for tagging
GIT_COMMIT=$(git rev-parse --short HEAD)
TIMESTAMP=$(date +%Y%m%d%H%M%S)
TAG="${GIT_COMMIT}-${TIMESTAMP}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_step() {
    echo -e "${BLUE}[STEP]${NC} $1"
}

# Check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."
    
    # Check if we're in the right directory
    if [[ ! -f "pyproject.toml" ]]; then
        log_error "Must be run from project root directory"
        exit 1
    fi
    
    # Check if gcloud is authenticated
    if ! gcloud auth list --filter=status:ACTIVE --format="value(account)" | head -n 1 > /dev/null; then
        log_error "Please authenticate with gcloud: gcloud auth login"
        exit 1
    fi
    
    # Check if kubectl is configured
    if ! kubectl cluster-info &> /dev/null; then
        log_error "kubectl is not configured. Run: gcloud container clusters get-credentials $CLUSTER_NAME --zone=$ZONE"
        exit 1
    fi
    
    log_info "Prerequisites check passed"
}

# Build Docker images
build_images() {
    log_step "Building Docker images..."
    
    # Configure Docker for GCR
    gcloud auth configure-docker
    
    # Build API image
    log_info "Building API image..."
    docker build \
        -t "$IMAGE_API:$TAG" \
        -t "$IMAGE_API:latest" \
        -f docker/Dockerfile.api \
        .
    
    # Build Orchestrator image
    log_info "Building Orchestrator image..."
    docker build \
        -t "$IMAGE_ORCHESTRATOR:$TAG" \
        -t "$IMAGE_ORCHESTRATOR:latest" \
        -f docker/Dockerfile.orchestrator \
        .
    
    # Build Generation image
    log_info "Building Generation image..."
    docker build \
        -t "$IMAGE_GENERATION:$TAG" \
        -t "$IMAGE_GENERATION:latest" \
        -f docker/Dockerfile.generation \
        .
    
    log_info "Images built successfully"
}

# Push Docker images
push_images() {
    log_step "Pushing Docker images to GCR..."
    
    # Push API image
    docker push "$IMAGE_API:$TAG"
    docker push "$IMAGE_API:latest"
    
    # Push Orchestrator image
    docker push "$IMAGE_ORCHESTRATOR:$TAG"
    docker push "$IMAGE_ORCHESTRATOR:latest"
    
    # Push Generation image
    docker push "$IMAGE_GENERATION:$TAG"
    docker push "$IMAGE_GENERATION:latest"
    
    log_info "Images pushed successfully"
}

# Deploy to Kubernetes
deploy_k8s() {
    log_step "Deploying to Kubernetes..."
    
    # Apply manifests in order
    log_info "Applying namespace and RBAC..."
    kubectl apply -f k8s/namespace.yaml
    kubectl apply -f k8s/rbac.yaml
    
    log_info "Applying configuration and secrets..."
    kubectl apply -f k8s/configmap.yaml
    kubectl apply -f k8s/secrets.yaml
    
    log_info "Deploying databases..."
    kubectl apply -f k8s/postgres-deployment.yaml
    kubectl apply -f k8s/redis-deployment.yaml
    
    # Wait for databases to be ready
    log_info "Waiting for databases to be ready..."
    kubectl wait --for=condition=ready pod -l app=postgres -n $NAMESPACE --timeout=300s
    kubectl wait --for=condition=ready pod -l app=redis -n $NAMESPACE --timeout=300s
    
    log_info "Deploying application services..."
    kubectl apply -f k8s/api-deployment.yaml
    kubectl apply -f k8s/orchestration-deployment.yaml
    kubectl apply -f k8s/gpu-worker-deployment.yaml
    
    # Update image tags to latest
    log_info "Updating image tags..."
    kubectl set image deployment/visionflow-api api="$IMAGE_API:$TAG" -n $NAMESPACE
    kubectl set image deployment/visionflow-orchestrator orchestrator="$IMAGE_ORCHESTRATOR:$TAG" -n $NAMESPACE
    kubectl set image deployment/visionflow-gpu-worker gpu-worker="$IMAGE_GENERATION:$TAG" -n $NAMESPACE
    
    log_info "Applying ingress and monitoring..."
    kubectl apply -f k8s/ingress.yaml
    kubectl apply -f k8s/monitoring.yaml
    
    log_info "Kubernetes deployment completed"
}

# Wait for deployment
wait_for_deployment() {
    log_step "Waiting for deployments to be ready..."
    
    # Wait for API deployment
    log_info "Waiting for API deployment..."
    kubectl rollout status deployment/visionflow-api -n $NAMESPACE --timeout=600s
    
    # Wait for Orchestrator deployment
    log_info "Waiting for Orchestrator deployment..."
    kubectl rollout status deployment/visionflow-orchestrator -n $NAMESPACE --timeout=600s
    
    # Wait for GPU worker deployment
    log_info "Waiting for GPU worker deployment..."
    kubectl rollout status deployment/visionflow-gpu-worker -n $NAMESPACE --timeout=600s
    
    log_info "All deployments are ready"
}

# Run health checks
health_check() {
    log_step "Running health checks..."
    
    # Get service endpoints
    API_IP=$(kubectl get service visionflow-api-service -n $NAMESPACE -o jsonpath='{.status.loadBalancer.ingress[0].ip}' 2>/dev/null || echo "")
    ORCH_IP=$(kubectl get service visionflow-orchestrator-service -n $NAMESPACE -o jsonpath='{.status.loadBalancer.ingress[0].ip}' 2>/dev/null || echo "")
    
    # If LoadBalancer IPs not available, use port-forward for testing
    if [[ -z "$API_IP" ]]; then
        log_warn "LoadBalancer IP not available, using port-forward for health check"
        kubectl port-forward service/visionflow-api-service 8080:80 -n $NAMESPACE &
        PORT_FORWARD_PID=$!
        sleep 5
        
        if curl -f "http://localhost:8080/health" &> /dev/null; then
            log_info "API health check passed"
        else
            log_warn "API health check failed"
        fi
        
        kill $PORT_FORWARD_PID 2>/dev/null || true
    else
        if curl -f "http://$API_IP/health" &> /dev/null; then
            log_info "API health check passed"
        else
            log_warn "API health check failed"
        fi
    fi
    
    log_info "Health checks completed"
}

# Show deployment info
show_info() {
    log_step "Deployment Information"
    
    echo ""
    echo "=============================="
    echo "VisionFlow Deployment Complete"
    echo "=============================="
    echo ""
    echo "Project: $PROJECT_ID"
    echo "Cluster: $CLUSTER_NAME"
    echo "Namespace: $NAMESPACE"
    echo "Tag: $TAG"
    echo ""
    
    echo "Services:"
    kubectl get services -n $NAMESPACE
    echo ""
    
    echo "Pods:"
    kubectl get pods -n $NAMESPACE
    echo ""
    
    echo "Ingress:"
    kubectl get ingress -n $NAMESPACE
    echo ""
    
    echo "To access the application:"
    echo "  kubectl port-forward service/visionflow-api-service 8080:80 -n $NAMESPACE"
    echo "  curl http://localhost:8080/health"
    echo ""
    
    echo "To view logs:"
    echo "  kubectl logs -f deployment/visionflow-api -n $NAMESPACE"
    echo "  kubectl logs -f deployment/visionflow-orchestrator -n $NAMESPACE"
    echo ""
    
    echo "To scale the deployment:"
    echo "  kubectl scale deployment visionflow-api --replicas=5 -n $NAMESPACE"
    echo ""
}

# Cleanup function
cleanup() {
    if [[ -n "$PORT_FORWARD_PID" ]]; then
        kill $PORT_FORWARD_PID 2>/dev/null || true
    fi
}

# Main execution
main() {
    log_info "Starting VisionFlow deployment..."
    
    # Set up cleanup trap
    trap cleanup EXIT
    
    check_prerequisites
    build_images
    push_images
    deploy_k8s
    wait_for_deployment
    health_check
    show_info
    
    log_info "Deployment completed successfully!"
}

# Script execution
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi
