#!/bin/bash
# VisionFlow Development Environment Manager
# Comprehensive tool for managing local development environment

set -e

# Configuration
NAMESPACE="visionflow-local"
CLUSTER_NAME="visionflow-local"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Service definitions (compatible with bash 3.2)
SERVICES_NAMES=("api" "grafana" "prometheus" "generation" "orchestration")
SERVICES_INFO=(
    "api-gateway-service:8000:30000"
    "grafana-service:3000:30300"
    "prometheus-service:9090:30090"
    "generation-service:8002:30002"
    "orchestration-service:8001:30001"
)

# Function to print colored output
print_status() {
    local status=$1
    local message=$2
    case $status in
        "info") echo -e "${BLUE}ℹ️  $message${NC}" ;;
        "success") echo -e "${GREEN}✅ $message${NC}" ;;
        "warning") echo -e "${YELLOW}⚠️  $message${NC}" ;;
        "error") echo -e "${RED}❌ $message${NC}" ;;
        "header") echo -e "${PURPLE}🎯 $message${NC}" ;;
        "subheader") echo -e "${CYAN}📋 $message${NC}" ;;
    esac
}

# Function to check if kubectl is available
check_kubectl() {
    if ! command -v kubectl &> /dev/null; then
        print_status "error" "kubectl is not installed or not in PATH"
        exit 1
    fi
}

# Function to check cluster status
check_cluster() {
    print_status "info" "Checking cluster status..."
    
    if ! kubectl cluster-info &> /dev/null; then
        print_status "error" "Cannot connect to Kubernetes cluster"
        exit 1
    fi
    
    if ! kubectl get namespace $NAMESPACE &> /dev/null; then
        print_status "error" "Namespace '$NAMESPACE' not found. Run './dev-env-manager.sh setup' first"
        exit 1
    fi
    
    print_status "success" "Connected to cluster: $(kubectl config current-context)"
}

# Function to check if port is in use
check_port() {
    local port=$1
    if lsof -Pi :$port -sTCP:LISTEN -t >/dev/null 2>&1; then
        return 0  # Port is in use
    else
        return 1  # Port is free
    fi
}

# Function to kill existing port forwarding
kill_port_forwards() {
    print_status "info" "Stopping existing port forwarding..."
    pkill -f "kubectl port-forward" || true
    sleep 2
}

# Function to get service info by name
get_service_info() {
    local service_name=$1
    for i in "${!SERVICES_NAMES[@]}"; do
        if [[ "${SERVICES_NAMES[$i]}" == "$service_name" ]]; then
            echo "${SERVICES_INFO[$i]}"
            return 0
        fi
    done
    return 1
}

# Function to start port forwarding for a service
start_port_forward() {
    local service_name=$1
    local service_info=$(get_service_info "$service_name")
    
    if [[ -z "$service_info" ]]; then
        print_status "error" "Unknown service: $service_name"
        return 1
    fi
    
    IFS=':' read -r k8s_service service_port local_port <<< "$service_info"
    
    if check_port $local_port; then
        print_status "warning" "Port $local_port is already in use"
        return 1
    fi
    
    print_status "info" "Starting port forward for $service_name ($service_port -> $local_port)"
    
    nohup kubectl port-forward svc/$k8s_service $local_port:$service_port -n $NAMESPACE > /dev/null 2>&1 &
    local pid=$!
    
    # Wait and verify
    sleep 3
    if check_port $local_port; then
        print_status "success" "$service_name: $service_port -> localhost:$local_port (PID: $pid)"
        return 0
    else
        print_status "error" "Failed to start port forward for $service_name"
        return 1
    fi
}

# Function to health check a service
health_check_service() {
    local service_name=$1
    local service_info=$(get_service_info "$service_name")
    IFS=':' read -r k8s_service service_port local_port <<< "$service_info"
    
    case $service_name in
        "api")
            if curl -s http://localhost:$local_port/health > /dev/null 2>&1; then
                return 0
            fi
            ;;
        "grafana")
            if curl -s http://localhost:$local_port > /dev/null 2>&1; then
                return 0
            fi
            ;;
        "prometheus")
            if curl -s http://localhost:$local_port/api/v1/query?query=up > /dev/null 2>&1; then
                return 0
            fi
            ;;
        *)
            if check_port $local_port; then
                return 0
            fi
            ;;
    esac
    
    return 1
}

# Function to show service status
show_service_status() {
    print_status "subheader" "Service Status"
    echo "=================================="
    
    for service_name in "${SERVICES_NAMES[@]}"; do
        local service_info=$(get_service_info "$service_name")
        IFS=':' read -r k8s_service service_port local_port <<< "$service_info"
        
        if check_port $local_port; then
            if health_check_service "$service_name"; then
                print_status "success" "$service_name: localhost:$local_port"
            else
                print_status "warning" "$service_name: localhost:$local_port (port open but unhealthy)"
            fi
        else
            print_status "error" "$service_name: localhost:$local_port (not accessible)"
        fi
    done
}

# Function to show access URLs
show_access_urls() {
    print_status "subheader" "Service Access URLs"
    echo "=================================="
    echo -e "${GREEN}🚀 API Gateway:${NC} http://localhost:30000"
    echo -e "${GREEN}📊 Grafana:${NC} http://localhost:30300 (admin/admin123)"
    echo -e "${GREEN}📈 Prometheus:${NC} http://localhost:30090"
    echo -e "${GREEN}🎬 Generation Service:${NC} http://localhost:30002"
    echo -e "${GREEN}🎭 Orchestration Service:${NC} http://localhost:30001"
    echo ""
    print_status "info" "Alternative via kubectl proxy: http://localhost:8001/api/v1/namespaces/$NAMESPACE/services/"
}

# Function to start all services
start_all_services() {
    print_status "header" "Starting VisionFlow Development Environment"
    echo "=================================================="
    
    check_cluster
    kill_port_forwards
    
    local failed_services=()
    
    for service_name in "${SERVICES_NAMES[@]}"; do
        if ! start_port_forward "$service_name"; then
            failed_services+=("$service_name")
        fi
    done
    
    if [[ ${#failed_services[@]} -eq 0 ]]; then
        print_status "success" "All services started successfully!"
        show_access_urls
    else
        print_status "warning" "Some services failed to start: ${failed_services[*]}"
        show_service_status
    fi
}

# Function to stop all services
stop_all_services() {
    print_status "header" "Stopping VisionFlow Development Environment"
    echo "=================================================="
    
    kill_port_forwards
    print_status "success" "All port forwarding stopped"
}

# Function to restart all services
restart_all_services() {
    print_status "header" "Restarting VisionFlow Development Environment"
    echo "=================================================="
    
    stop_all_services
    sleep 2
    start_all_services
}

# Function to monitor services
monitor_services() {
    print_status "header" "Starting Service Monitoring"
    echo "=========================================="
    print_status "info" "Press Ctrl+C to stop monitoring"
    
    while true; do
        local all_healthy=true
        
        for service_name in "${SERVICES_NAMES[@]}"; do
            if ! health_check_service "$service_name"; then
                all_healthy=false
                print_status "warning" "Service $service_name is unhealthy, attempting restart..."
                
                # Kill existing forward for this service
                local service_info=$(get_service_info "$service_name")
                IFS=':' read -r k8s_service service_port local_port <<< "$service_info"
                pkill -f "kubectl port-forward.*$k8s_service" || true
                sleep 2
                
                # Restart it
                if start_port_forward "$service_name"; then
                    print_status "success" "Successfully restarted $service_name"
                else
                    print_status "error" "Failed to restart $service_name"
                fi
            fi
        done
        
        if $all_healthy; then
            print_status "success" "All services are healthy"
        fi
        
        sleep 30
    done
}

# Function to setup development environment
setup_dev_env() {
    print_status "header" "Setting Up VisionFlow Development Environment"
    echo "=========================================================="
    
    check_kubectl
    
    # Check if Kind cluster exists
    if ! kind get clusters | grep -q "$CLUSTER_NAME"; then
        print_status "info" "Creating Kind cluster '$CLUSTER_NAME'..."
        kind create cluster --name "$CLUSTER_NAME" --config "$PROJECT_ROOT/k8s/local/kind-config.yaml"
    else
        print_status "info" "Kind cluster '$CLUSTER_NAME' already exists"
    fi
    
    # Deploy services
    print_status "info" "Deploying VisionFlow services..."
    kubectl apply -f "$PROJECT_ROOT/k8s/local/complete-local-deployment.yaml"
    
    # Wait for services to be ready
    print_status "info" "Waiting for services to be ready..."
    kubectl wait --for=condition=ready pod -l app=api-gateway -n $NAMESPACE --timeout=300s
    
    print_status "success" "Development environment setup complete!"
    print_status "info" "Run './dev-env-manager.sh start' to start port forwarding"
}

# Function to cleanup development environment
cleanup_dev_env() {
    print_status "header" "Cleaning Up VisionFlow Development Environment"
    echo "=========================================================="
    
    print_status "warning" "This will delete the entire local cluster and all data!"
    read -p "Are you sure? (y/N): " -n 1 -r
    echo
    
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        stop_all_services
        kind delete cluster --name "$CLUSTER_NAME"
        print_status "success" "Development environment cleaned up"
    else
        print_status "info" "Cleanup cancelled"
    fi
}

# Function to show logs
show_logs() {
    local service_name=$1
    
    if [[ -z "$service_name" ]]; then
        print_status "error" "Please specify a service name"
        echo "Available services: ${!SERVICES[*]}"
        exit 1
    fi
    
    local service_info=$(get_service_info "$service_name")
    IFS=':' read -r k8s_service service_port local_port <<< "$service_info"
    
    print_status "info" "Showing logs for $service_name..."
    kubectl logs -f deployment/$k8s_service -n $NAMESPACE
}

# Function to show help
show_help() {
    cat << EOF
VisionFlow Development Environment Manager

Usage: $0 [COMMAND] [OPTIONS]

Commands:
  start           Start all services with port forwarding
  stop            Stop all port forwarding
  restart         Restart all services
  status          Show current service status
  urls            Show service access URLs
  monitor         Start auto-monitoring with auto-recovery
  setup           Setup development environment (create cluster, deploy services)
  cleanup         Clean up development environment (delete cluster)
  logs <service>  Show logs for a specific service
  help            Show this help message

Services:
  api             API Gateway (localhost:30000)
  grafana         Grafana Dashboard (localhost:30300)
  prometheus      Prometheus Metrics (localhost:30090)
  generation      Generation Service (localhost:30002)
  orchestration   Orchestration Service (localhost:30001)

Examples:
  $0 setup        # First time setup
  $0 start        # Start all services
  $0 monitor      # Start monitoring with auto-recovery
  $0 logs api     # Show API gateway logs
  $0 status       # Check service health
  $0 cleanup      # Clean up everything

EOF
}

# Main function
main() {
    case "${1:-help}" in
        "start")
            start_all_services
            ;;
        "stop")
            stop_all_services
            ;;
        "restart")
            restart_all_services
            ;;
        "status")
            check_cluster
            show_service_status
            ;;
        "urls")
            show_access_urls
            ;;
        "monitor")
            check_cluster
            monitor_services
            ;;
        "setup")
            setup_dev_env
            ;;
        "cleanup")
            cleanup_dev_env
            ;;
        "logs")
            check_cluster
            show_logs "$2"
            ;;
        "help"|*)
            show_help
            ;;
    esac
}

# Trap Ctrl+C to clean up
trap 'echo -e "\n${YELLOW}🛑 Received interrupt signal${NC}"; exit 0' INT

# Run main function
main "$@"
