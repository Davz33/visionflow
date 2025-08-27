#!/bin/bash

# Test VisionFlow Monitoring Dashboard with Local k8s Deployment
# This script deploys everything locally and tests real-time monitoring

set -e

echo "🎬 Testing VisionFlow Monitoring Dashboard with Local k8s Deployment"
echo "=================================================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
NAMESPACE="visionflow-local"
SCRIPT_DIR="$(dirname "$0")"
PROJECT_DIR="$SCRIPT_DIR/.."

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to wait for service to be ready
wait_for_service() {
    local service_name=$1
    local port=$2
    local max_attempts=30
    local attempt=1
    
    print_status "Waiting for $service_name to be ready on port $port..."
    
    while [ $attempt -le $max_attempts ]; do
        if curl -s -f "http://localhost:$port/health" > /dev/null 2>&1; then
            print_success "$service_name is ready!"
            return 0
        fi
        
        echo -n "."
        sleep 2
        attempt=$((attempt + 1))
    done
    
    print_error "$service_name failed to start after $max_attempts attempts"
    return 1
}

# Function to check if kind cluster exists
check_kind_cluster() {
    if ! kind get clusters | grep -q "visionflow-local"; then
        print_status "Creating kind cluster..."
        kind create cluster --name visionflow-local --config "$PROJECT_DIR/k8s/local/kind-config.yaml"
        print_success "Kind cluster created!"
    else
        print_status "Kind cluster already exists"
    fi
}

# Function to deploy the monitoring stack
deploy_monitoring() {
    print_status "Deploying monitoring stack..."
    
    # Apply the monitoring configuration
    kubectl apply -f "$PROJECT_DIR/k8s/monitoring.yaml" -n $NAMESPACE
    
    # Wait for monitoring services to be ready
    kubectl wait --for=condition=available --timeout=300s deployment/prometheus -n $NAMESPACE
    kubectl wait --for=condition=available --timeout=300s deployment/grafana -n $NAMESPACE
    
    print_success "Monitoring stack deployed!"
}

# Function to test video generation API
test_video_generation() {
    print_status "Testing video generation API..."
    
    # Test the API health
    if ! curl -s -f "http://localhost:30000/health" > /dev/null; then
        print_error "API is not responding. Check if it's running."
        return 1
    fi
    
    # Generate a test video
    print_status "Sending video generation request..."
    
    local response=$(curl -s -X POST "http://localhost:30000/generate/video" \
        -H "Content-Type: application/json" \
        -d '{
            "prompt": "A beautiful sunset over the ocean for testing monitoring",
            "duration": 3,
            "quality": "medium",
            "fps": 24,
            "resolution": "512x512",
            "seed": 42,
            "guidance_scale": 7.5,
            "num_inference_steps": 20
        }')
    
    if echo "$response" | grep -q "error"; then
        print_warning "Video generation request failed: $response"
        return 1
    else
        print_success "Video generation request sent successfully!"
        echo "Response: $response"
        return 0
    fi
}

# Function to check monitoring metrics
check_monitoring_metrics() {
    print_status "Checking monitoring metrics..."
    
    # Wait a bit for metrics to be collected
    sleep 10
    
    # Check Prometheus metrics
    print_status "Checking Prometheus metrics..."
    local metrics=$(curl -s "http://localhost:30090/metrics")
    
    if echo "$metrics" | grep -q "visionflow_video_generation"; then
        print_success "Video generation metrics found in Prometheus!"
        echo "Available metrics:"
        echo "$metrics" | grep "visionflow_video_generation" | head -10
    else
        print_warning "No video generation metrics found in Prometheus"
    fi
    
    # Check Grafana dashboard
    print_status "Checking Grafana dashboard..."
    if curl -s -f "http://localhost:30300" > /dev/null; then
        print_success "Grafana is accessible!"
        print_status "Grafana URL: http://localhost:30300 (admin/admin123)"
    else
        print_warning "Grafana is not accessible"
    fi
}

# Function to show real-time monitoring
show_real_time_monitoring() {
    print_status "Setting up real-time monitoring..."
    
    # Start port forwarding for monitoring services
    print_status "Starting port forwarding for monitoring services..."
    
    # Prometheus
    kubectl port-forward svc/prometheus-service 30090:9090 -n $NAMESPACE &
    PROMETHEUS_PID=$!
    
    # Grafana
    kubectl port-forward svc/grafana-service 30300:3000 -n $NAMESPACE &
    GRAFANA_PID=$!
    
    # Wait for services to be accessible
    sleep 5
    
    print_success "Port forwarding started!"
    print_status "Prometheus: http://localhost:30090"
    print_status "Grafana: http://localhost:30300 (admin/admin123)"
    
    # Show current metrics
    print_status "Current video generation metrics:"
    curl -s "http://localhost:30090/api/v1/query?query=visionflow_video_generation_jobs_active" | jq '.' 2>/dev/null || echo "Metrics not available yet"
    
    # Instructions for testing
    echo ""
    print_status "🎯 MONITORING DASHBOARD TEST INSTRUCTIONS:"
    echo "1. Open Grafana: http://localhost:30300 (admin/admin123)"
    echo "2. Navigate to Dashboards > VisionFlow Video Generation Dashboard"
    echo "3. Send video generation requests via API or web monitor"
    echo "4. Watch real-time updates in the dashboard"
    echo ""
    print_status "📱 WEB MONITOR:"
    echo "Open: $PROJECT_DIR/video_generation_monitor.html in your browser"
    echo "Update API_BASE_URL to: http://localhost:30000"
    echo ""
    print_status "🔍 API TESTING:"
    echo "Health check: curl http://localhost:30000/health"
    echo "Generate video: curl -X POST http://localhost:30000/generate/video -H 'Content-Type: application/json' -d '{\"prompt\":\"test\",\"duration\":3,\"quality\":\"medium\",\"fps\":24,\"resolution\":\"512x512\"}'"
    
    # Keep the script running to maintain port forwarding
    print_status "Press Ctrl+C to stop monitoring and clean up..."
    
    # Trap to clean up background processes
    trap 'cleanup' INT TERM
    
    # Wait for user to stop
    wait
}

# Function to cleanup
cleanup() {
    print_status "Cleaning up..."
    
    # Kill background processes
    if [ ! -z "$PROMETHEUS_PID" ]; then
        kill $PROMETHEUS_PID 2>/dev/null || true
    fi
    
    if [ ! -z "$GRAFANA_PID" ]; then
        kill $GRAFANA_PID 2>/dev/null || true
    fi
    
    print_success "Cleanup complete!"
    exit 0
}

# Main execution
main() {
    print_status "Starting VisionFlow Monitoring Dashboard Test..."
    
    # Check prerequisites
    if ! command -v kubectl &> /dev/null; then
        print_error "kubectl is not installed. Please install kubectl first."
        exit 1
    fi
    
    if ! command -v kind &> /dev/null; then
        print_error "kind is not installed. Please install kind first."
        exit 1
    fi
    
    if ! command -v curl &> /dev/null; then
        print_error "curl is not installed. Please install curl first."
        exit 1
    fi
    
    # Check if kind cluster exists and create if needed
    check_kind_cluster
    
    # Deploy the complete stack
    print_status "Deploying complete VisionFlow stack..."
    bash "$SCRIPT_DIR/deploy-k8s-local.sh"
    
    # Wait for all services to be ready
    print_status "Waiting for all services to be ready..."
    kubectl wait --for=condition=available --timeout=600s deployment --all -n $NAMESPACE
    
    # Deploy monitoring stack
    deploy_monitoring
    
    # Start port forwarding for main services
    print_status "Starting port forwarding for main services..."
    kubectl port-forward svc/api-gateway 30000:80 -n $NAMESPACE &
    API_PID=$!
    
    # Wait for API to be ready
    wait_for_service "API Gateway" 30000
    
    # Test video generation
    test_video_generation
    
    # Check monitoring metrics
    check_monitoring_metrics
    
    # Show real-time monitoring
    show_real_time_monitoring
}

# Run main function
main "$@"
