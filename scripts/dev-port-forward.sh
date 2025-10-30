#!/bin/bash
# Development Port Forwarding with Auto-Recovery
# This script automatically manages port forwarding for local development

set -e

# Configuration
NAMESPACE="visionflow-local"
SERVICES=(
    "api-gateway-service:8000:30000"
    "grafana-service:3000:30300"
    "prometheus-service:9090:30090"
    "generation-service:8002:30002"
    "orchestration-service:8001:30001"
)

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to check if a port is in use
check_port() {
    local port=$1
    if lsof -Pi :$port -sTCP:LISTEN -t >/dev/null 2>&1; then
        return 0  # Port is in use
    else
        return 1  # Port is free
    fi
}

# Function to kill existing port forwarding
kill_existing_forwards() {
    echo -e "${YELLOW}🔌 Killing existing port forwarding processes...${NC}"
    pkill -f "kubectl port-forward" || true
    sleep 2
}

# Function to start port forwarding for a service
start_port_forward() {
    local service_info=$1
    IFS=':' read -r service_name service_port local_port <<< "$service_info"
    
    if check_port $local_port; then
        echo -e "${YELLOW}⚠️  Port $local_port is already in use${NC}"
        return 1
    fi
    
    echo -e "${BLUE}🚀 Starting port forward for $service_name (${service_port} -> $local_port)${NC}"
    
    # Start port forwarding in background with nohup
    nohup kubectl port-forward svc/$service_name $local_port:$service_port -n $NAMESPACE > /dev/null 2>&1 &
    local pid=$!
    
    # Wait a moment and check if it's working
    sleep 3
    if check_port $local_port; then
        echo -e "${GREEN}✅ $service_name: $service_port -> localhost:$local_port (PID: $pid)${NC}"
        return 0
    else
        echo -e "${RED}❌ Failed to start port forward for $service_name${NC}"
        return 1
    fi
}

# Function to health check a service
health_check() {
    local service_info=$1
    IFS=':' read -r service_name service_port local_port <<< "$service_info"
    
    case $service_name in
        "api-gateway-service")
            if curl -s http://localhost:$local_port/health > /dev/null 2>&1; then
                echo -e "${GREEN}✅ $service_name health check passed${NC}"
                return 0
            fi
            ;;
        "grafana-service")
            if curl -s http://localhost:$local_port > /dev/null 2>&1; then
                echo -e "${GREEN}✅ $service_name health check passed${NC}"
                return 0
            fi
            ;;
        "prometheus-service")
            if curl -s http://localhost:$local_port/api/v1/query?query=up > /dev/null 2>&1; then
                echo -e "${GREEN}✅ $service_name health check passed${NC}"
                return 0
            fi
            ;;
        *)
            # For other services, just check if port is listening
            if check_port $local_port; then
                echo -e "${GREEN}✅ $service_name port check passed${NC}"
                return 0
            fi
            ;;
    esac
    
    echo -e "${RED}❌ $service_name health check failed${NC}"
    return 1
}

# Function to monitor and auto-recover port forwarding
monitor_services() {
    echo -e "${BLUE}🔍 Starting service monitoring...${NC}"
    
    while true; do
        local all_healthy=true
        
        for service_info in "${SERVICES[@]}"; do
            if ! health_check "$service_info"; then
                all_healthy=false
                echo -e "${YELLOW}🔄 Attempting to restart $service_info...${NC}"
                
                # Kill existing forward for this service
                IFS=':' read -r service_name service_port local_port <<< "$service_info"
                pkill -f "kubectl port-forward.*$service_name" || true
                sleep 2
                
                # Restart it
                if start_port_forward "$service_info"; then
                    echo -e "${GREEN}✅ Successfully restarted $service_name${NC}"
                else
                    echo -e "${RED}❌ Failed to restart $service_name${NC}"
                fi
            fi
        done
        
        if $all_healthy; then
            echo -e "${GREEN}🎯 All services are healthy${NC}"
        fi
        
        # Wait before next check
        sleep 30
    done
}

# Function to show service status
show_status() {
    echo -e "${BLUE}📊 Current Service Status:${NC}"
    echo "=================================="
    
    for service_info in "${SERVICES[@]}"; do
        IFS=':' read -r service_name service_port local_port <<< "$service_info"
        
        if check_port $local_port; then
            if health_check "$service_info" > /dev/null 2>&1; then
                echo -e "${GREEN}✅ $service_name: localhost:$local_port${NC}"
            else
                echo -e "${YELLOW}⚠️  $service_name: localhost:$local_port (port open but unhealthy)${NC}"
            fi
        else
            echo -e "${RED}❌ $service_name: localhost:$local_port (not accessible)${NC}"
        fi
    done
}

# Function to show access URLs
show_urls() {
    echo -e "${BLUE}🌐 Service Access URLs:${NC}"
    echo "=================================="
    echo -e "${GREEN}🚀 API Gateway:${NC} http://localhost:30000"
    echo -e "${GREEN}📊 Grafana:${NC} http://localhost:30300 (admin/admin123)"
    echo -e "${GREEN}📈 Prometheus:${NC} http://localhost:30090"
    echo -e "${GREEN}🎬 Generation Service:${NC} http://localhost:30002"
    echo -e "${GREEN}🎭 Orchestration Service:${NC} http://localhost:30001"
    echo ""
    echo -e "${BLUE}💡 Alternative via kubectl proxy:${NC}"
    echo "http://localhost:8001/api/v1/namespaces/$NAMESPACE/services/api-gateway-service:8000/proxy/"
}

# Main function
main() {
    echo -e "${BLUE}🚀 VisionFlow Development Port Forwarding${NC}"
    echo "=============================================="
    
    # Check if kubectl is available
    if ! command -v kubectl &> /dev/null; then
        echo -e "${RED}❌ kubectl is not installed or not in PATH${NC}"
        exit 1
    fi
    
    # Check if we're connected to the right cluster
    if ! kubectl get namespace $NAMESPACE > /dev/null 2>&1; then
        echo -e "${RED}❌ Cannot access namespace '$NAMESPACE'. Make sure:${NC}"
        echo "   1. You're connected to the right cluster"
        echo "   2. The namespace exists"
        echo "   3. You have proper permissions"
        exit 1
    fi
    
    case "${1:-start}" in
        "start")
            echo -e "${BLUE}🚀 Starting port forwarding for all services...${NC}"
            kill_existing_forwards
            
            for service_info in "${SERVICES[@]}"; do
                start_port_forward "$service_info"
            done
            
            echo -e "${GREEN}🎉 All port forwarding started!${NC}"
            show_urls
            echo ""
            echo -e "${BLUE}💡 Use '${NC}./dev-port-forward.sh monitor${NC}' to start auto-monitoring"
            echo -e "${BLUE}💡 Use '${NC}./dev-port-forward.sh status${NC}' to check service health"
            ;;
        "stop")
            echo -e "${YELLOW}🛑 Stopping all port forwarding...${NC}"
            kill_existing_forwards
            echo -e "${GREEN}✅ All port forwarding stopped${NC}"
            ;;
        "restart")
            echo -e "${BLUE}🔄 Restarting all port forwarding...${NC}"
            kill_existing_forwards
            sleep 2
            
            for service_info in "${SERVICES[@]}"; do
                start_port_forward "$service_info"
            done
            
            echo -e "${GREEN}🎉 All port forwarding restarted!${NC}"
            show_urls
            ;;
        "status")
            show_status
            ;;
        "urls")
            show_urls
            ;;
        "monitor")
            echo -e "${BLUE}🔍 Starting auto-monitoring mode...${NC}"
            echo -e "${YELLOW}Press Ctrl+C to stop monitoring${NC}"
            monitor_services
            ;;
        *)
            echo -e "${BLUE}Usage: $0 [start|stop|restart|status|urls|monitor]${NC}"
            echo ""
            echo "Commands:"
            echo "  start   - Start port forwarding for all services"
            echo "  stop    - Stop all port forwarding"
            echo "  restart - Restart all port forwarding"
            echo "  status  - Show current service status"
            echo "  urls    - Show service access URLs"
            echo "  monitor - Start auto-monitoring with auto-recovery"
            echo ""
            echo "Examples:"
            echo "  $0 start    # Start all services"
            echo "  $0 monitor  # Start monitoring with auto-recovery"
            echo "  $0 status   # Check current status"
            exit 1
            ;;
    esac
}

# Trap Ctrl+C to clean up
trap 'echo -e "\n${YELLOW}🛑 Received interrupt signal${NC}"; exit 0' INT

# Run main function
main "$@"
