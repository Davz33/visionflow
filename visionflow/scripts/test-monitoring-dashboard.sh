#!/bin/bash
# Test VisionFlow Monitoring Dashboard with Local k8s Deployment
# This script tests the complete monitoring infrastructure

set -e

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_status() {
    local status=$1
    local message=$2
    case $status in
        "info") echo -e "${BLUE}ℹ️  $message${NC}" ;;
        "success") echo -e "${GREEN}✅ $message${NC}" ;;
        "warning") echo -e "${YELLOW}⚠️  $message${NC}" ;;
        "error") echo -e "${RED}❌ $message${NC}" ;;
    esac
}

# Function to check if services are accessible
check_services() {
    print_status "info" "Checking service accessibility..."
    
    # Check API Gateway
    if curl -s http://localhost:30000/health > /dev/null; then
        print_status "success" "API Gateway is accessible"
    else
        print_status "error" "API Gateway is not accessible"
        return 1
    fi
    
    # Check Grafana
    if curl -s http://localhost:30300 > /dev/null; then
        print_status "success" "Grafana is accessible"
    else
        print_status "error" "Grafana is not accessible"
        return 1
    fi
    
    # Check Prometheus
    if curl -s http://localhost:30090/api/v1/query?query=up > /dev/null; then
        print_status "success" "Prometheus is accessible"
    else
        print_status "error" "Prometheus is not accessible"
        return 1
    fi
}

# Function to test video generation API
test_video_generation() {
    print_status "info" "Testing video generation API..."
    
    # Send a test request
    local response=$(curl -s -X POST "http://localhost:30000/generate/video" \
        -H "Content-Type: application/json" \
        -d '{"prompt": "A beautiful sunset for monitoring test", "duration": 3, "quality": "medium", "resolution": "512x512"}' \
        --max-time 10 || echo "TIMEOUT")
    
    if [[ "$response" == "TIMEOUT" ]]; then
        print_status "warning" "Video generation request timed out (expected for long-running ML tasks)"
        print_status "info" "Check the API logs to verify the request was processed"
    else
        print_status "success" "Video generation request sent successfully"
        echo "Response: $response"
    fi
}

# Function to check monitoring metrics
check_monitoring_metrics() {
    print_status "info" "Checking monitoring metrics..."
    
    # Check if Prometheus is collecting metrics from the API
    local metrics=$(curl -s http://localhost:30000/metrics)
    
    if echo "$metrics" | grep -q "visionflow"; then
        print_status "success" "VisionFlow metrics are being exposed"
        echo "$metrics" | grep "visionflow" | head -5
    else
        print_status "warning" "No VisionFlow metrics found yet (they appear when services are used)"
    fi
    
    # Check Prometheus targets
    local targets=$(curl -s "http://localhost:30090/api/v1/query?query=up")
    if echo "$targets" | grep -q "api-gateway-service"; then
        print_status "success" "Prometheus is scraping the API Gateway"
    else
        print_status "error" "Prometheus is not scraping the API Gateway"
    fi
}

# Function to show monitoring access instructions
show_monitoring_instructions() {
    print_status "info" "Monitoring Dashboard Access Instructions"
    echo "=================================================="
    echo ""
    echo "🌐 Grafana Dashboard:"
    echo "   URL: http://localhost:30300"
    echo "   Username: admin"
    echo "   Password: admin123"
    echo "   Look for: VisionFlow Video Generation Dashboard"
    echo ""
    echo "📈 Prometheus Metrics:"
    echo "   URL: http://localhost:30090"
    echo "   Check: Targets, queries, and raw metrics"
    echo ""
    echo "🚀 API Gateway:"
    echo "   URL: http://localhost:30000"
    echo "   Health: http://localhost:30000/health"
    echo "   Metrics: http://localhost:30000/metrics"
    echo ""
    echo "📱 Web Monitor:"
    echo "   File: $PROJECT_ROOT/video_generation_monitor.html"
    echo "   Open in browser for quick monitoring"
}

# Function to show real-time monitoring
show_real_time_monitoring() {
    print_status "info" "Real-Time Monitoring Status"
    echo "====================================="
    
    # Show current metrics
    echo "Current API Metrics:"
    curl -s http://localhost:30000/metrics | grep -E "(visionflow|python_gc)" | head -10
    
    echo ""
    echo "Current Prometheus Targets:"
    curl -s "http://localhost:30090/api/v1/query?query=up" | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    print('Target Status:')
    for result in data['data']['result']:
        instance = result['metric']['instance']
        status = 'UP' if result['value'][1] == '1' else 'DOWN'
        print(f'  {instance}: {status}')
except:
    print('Could not parse Prometheus response')
"
}

# Main function
main() {
    print_status "info" "🎬 Testing VisionFlow Monitoring Dashboard"
    echo "=================================================="
    
    # Check if development environment manager is available
    if [[ ! -f "$SCRIPT_DIR/dev-env-manager.sh" ]]; then
        print_status "error" "Development environment manager not found"
        exit 1
    fi
    
    # Check if services are running
    print_status "info" "Checking if services are running..."
    if ! bash "$SCRIPT_DIR/dev-env-manager.sh" status > /dev/null 2>&1; then
        print_status "warning" "Services are not running. Starting them..."
        bash "$SCRIPT_DIR/dev-env-manager.sh" start
    fi
    
    # Wait for services to be ready
    print_status "info" "Waiting for services to be ready..."
    sleep 10
    
    # Check service accessibility
    if ! check_services; then
        print_status "error" "Some services are not accessible. Check the deployment."
        exit 1
    fi
    
    # Test video generation
    test_video_generation
    
    # Check monitoring metrics
    check_monitoring_metrics
    
    # Show monitoring instructions
    show_monitoring_instructions
    
    # Show real-time status
    show_real_time_monitoring
    
    print_status "success" "🎉 Monitoring dashboard test completed!"
    echo ""
    print_status "info" "Next steps:"
    echo "1. Open Grafana: http://localhost:30300"
    echo "2. Send video generation requests via the API"
    echo "3. Watch real-time metrics update in the dashboard"
    echo "4. Use './scripts/dev-env-manager.sh monitor' for auto-recovery"
}

# Run main function
main "$@"
