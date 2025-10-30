#!/bin/bash

# VisionFlow Setup Script
# This script sets up the VisionFlow development environment

set -e  # Exit on any error

echo "🚀 Setting up VisionFlow development environment..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

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

# Check if we're on macOS
if [[ "$OSTYPE" != "darwin"* ]]; then
    print_warning "This script is optimized for macOS. Some steps may need adjustment for other platforms."
fi

# Check for required tools
print_status "Checking prerequisites..."

# Check Docker
if ! command -v docker &> /dev/null; then
    print_error "Docker is required but not installed. Please install Docker Desktop from https://www.docker.com/products/docker-desktop"
    exit 1
fi

# Check Docker Compose
if ! command -v docker-compose &> /dev/null; then
    print_error "Docker Compose is required but not installed. Please install Docker Compose."
    exit 1
fi

print_success "Prerequisites check passed"

# Create necessary directories
print_status "Creating project directories..."
mkdir -p models
mkdir -p generated
mkdir -p logs
mkdir -p data/postgres
mkdir -p data/redis
mkdir -p data/minio

print_success "Directories created"

# Create .env file if it doesn't exist
if [[ ! -f .env ]]; then
    print_status "Creating .env file..."
    cat > .env << EOF
# Environment
ENVIRONMENT=development
DEBUG=true

# Database
DB_HOST=localhost
DB_PORT=5432
DB_NAME=visionflow
DB_USER=visionflow
DB_PASSWORD=visionflow

# Redis
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_DB=0
REDIS_PASSWORD=

# Storage
STORAGE_ENDPOINT=localhost:9000
STORAGE_ACCESS_KEY=minio
STORAGE_SECRET_KEY=minio123
STORAGE_BUCKET=visionflow
STORAGE_SECURE=false

# API
API_HOST=0.0.0.0
API_PORT=8000
API_WORKERS=1
API_RELOAD=true
LOG_LEVEL=info

# Model
WAN_MODEL_PATH=multimodalart/wan2-1-fast
MODEL_CACHE_DIR=./models
MODEL_DEVICE=auto
DEFAULT_DURATION=5
MAX_DURATION=30

# Rate Limiting
RATE_LIMIT_REQUESTS=100
RATE_LIMIT_WINDOW=3600

# Circuit Breaker
CIRCUIT_BREAKER_FAILURE_THRESHOLD=5
CIRCUIT_BREAKER_RECOVERY_TIMEOUT=30

# Monitoring
PROMETHEUS_PORT=9090
GRAFANA_PORT=3000
ENABLE_METRICS=true

# Concurrency
MAX_CONCURRENT_GENERATIONS=2
TASK_TIMEOUT=300
EOF
    print_success ".env file created"
else
    print_warning ".env file already exists, skipping creation"
fi

# Note: All Python dependencies are handled within Docker containers
print_status "Python dependencies will be managed within Docker containers"

# Pull required Docker images
print_status "Pulling Docker images..."
docker-compose pull

print_success "Docker images pulled"

# Docker Compose will create network automatically
print_status "Docker Compose will handle network creation automatically"

# Start infrastructure services (database, redis, minio)
print_status "Starting infrastructure services..."
docker-compose up -d postgres redis minio

# Wait for services to be ready
print_status "Waiting for services to be ready..."
sleep 10

# Check if services are healthy
for service in postgres redis minio; do
    if docker-compose ps $service | grep -q "healthy\|Up"; then
        print_success "$service is ready"
    else
        print_warning "$service may not be ready yet"
    fi
done

# Create MinIO bucket
print_status "Setting up MinIO bucket..."
docker-compose exec -T minio mc alias set local http://localhost:9000 minio minio123
docker-compose exec -T minio mc mb local/visionflow --ignore-existing
print_success "MinIO bucket created"

# Database tables will be created when the API service starts
print_status "Database tables will be created automatically when services start"

# Create example API test script
print_status "Creating test scripts..."
cat > test_api.py << 'EOF'
#!/usr/bin/env python3
"""Simple API test script for VisionFlow."""

import requests
import time
import json

def test_health():
    """Test health endpoint."""
    response = requests.get("http://localhost:8000/health")
    print(f"Health check: {response.status_code}")
    if response.status_code == 200:
        health_data = response.json()
        print(f"Status: {health_data['status']}")
        print(f"Database: {'✓' if health_data['database_connected'] else '✗'}")
        print(f"Redis: {'✓' if health_data['redis_connected'] else '✗'}")
        print(f"Storage: {'✓' if health_data['storage_connected'] else '✗'}")

def test_generate_video():
    """Test video generation."""
    payload = {
        "prompt": "A serene sunset over mountains with gentle clouds",
        "duration": 5,
        "quality": "medium",
        "fps": 24,
        "resolution": "512x512"
    }
    
    print("\nTesting video generation...")
    response = requests.post("http://localhost:8000/api/v1/generate", json=payload)
    
    if response.status_code == 200:
        job_data = response.json()
        job_id = job_data["job_id"]
        print(f"Job created: {job_id}")
        
        # Poll for completion
        for i in range(30):  # Wait up to 5 minutes
            status_response = requests.get(f"http://localhost:8000/api/v1/jobs/{job_id}/status")
            if status_response.status_code == 200:
                status_data = status_response.json()
                print(f"Status: {status_data['status']} ({status_data['progress']:.1%})")
                
                if status_data['status'] == 'completed':
                    print("✓ Video generation completed!")
                    break
                elif status_data['status'] == 'failed':
                    print(f"✗ Video generation failed: {status_data.get('error_message')}")
                    break
            
            time.sleep(10)
    else:
        print(f"Failed to create job: {response.status_code} - {response.text}")

if __name__ == "__main__":
    print("VisionFlow API Test")
    print("==================")
    
    try:
        test_health()
        test_generate_video()
    except requests.exceptions.ConnectionError:
        print("❌ Could not connect to API. Make sure the service is running.")
    except Exception as e:
        print(f"❌ Test failed: {e}")
EOF

chmod +x test_api.py
print_success "Test script created"

# Create development helper script
cat > dev.sh << 'EOF'
#!/bin/bash

# VisionFlow Development Helper Script - Docker Edition

# Colors for output
BLUE='\033[0;34m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

# Function to run commands in development container
run_in_dev_container() {
    docker-compose -f docker-compose.yml -f docker-compose.dev.yml run --rm --no-deps dev "$@"
}

# Function to check if services are running
check_services() {
    if ! docker-compose ps | grep -q "Up"; then
        print_warning "Services don't appear to be running. Starting them first..."
        docker-compose up -d postgres redis minio
        sleep 5
    fi
}

case "$1" in
    "start")
        echo "🚀 Starting VisionFlow services in development mode..."
        docker-compose -f docker-compose.yml -f docker-compose.dev.yml up -d
        echo ""
        print_success "Services started!"
        echo "  📡 API & Docs: http://localhost:8000 (with hot reload)"
        echo "  📊 Grafana: http://localhost:3000 (admin/admin)"
        echo "  📈 Prometheus: http://localhost:9090"
        echo "  💾 MinIO: http://localhost:9001 (minio/minio123)"
        echo "  🗄️  PostgreSQL: localhost:5432 (for external tools)"
        echo "  📦 Redis: localhost:6379 (for external tools)"
        ;;
    "stop")
        echo "🛑 Stopping VisionFlow services..."
        docker-compose -f docker-compose.yml -f docker-compose.dev.yml down
        print_success "Services stopped"
        ;;
    "restart")
        echo "🔄 Restarting VisionFlow services..."
        docker-compose -f docker-compose.yml -f docker-compose.dev.yml down
        docker-compose -f docker-compose.yml -f docker-compose.dev.yml up -d
        print_success "Services restarted"
        ;;
    "build")
        echo "🔨 Building Docker images..."
        docker-compose -f docker-compose.yml -f docker-compose.dev.yml build
        print_success "Images built"
        ;;
    "rebuild")
        echo "🔨 Rebuilding Docker images from scratch..."
        docker-compose -f docker-compose.yml -f docker-compose.dev.yml build --no-cache
        print_success "Images rebuilt"
        ;;
    "logs")
        if [ -z "$2" ]; then
            docker-compose logs -f
        else
            docker-compose logs -f "$2"
        fi
        ;;
    "test")
        echo "🧪 Running API tests in container..."
        check_services
        run_in_dev_container python test_api.py
        ;;
    "pytest")
        echo "🧪 Running pytest in container..."
        shift  # Remove first argument
        run_in_dev_container pytest tests/ "$@"
        ;;
    "lint")
        echo "🔍 Running code linting in container..."
        print_info "Checking with black..."
        run_in_dev_container black . --check
        echo ""
        print_info "Checking with isort..."
        run_in_dev_container isort . --check-only
        echo ""
        print_info "Type checking with mypy..."
        run_in_dev_container mypy . || true
        ;;
    "format")
        echo "✨ Formatting code in container..."
        print_info "Running black..."
        run_in_dev_container black .
        print_info "Running isort..."
        run_in_dev_container isort .
        print_success "Code formatted"
        ;;
    "shell")
        echo "🐚 Opening shell in development container..."
        docker-compose -f docker-compose.yml -f docker-compose.dev.yml run --rm dev /bin/bash
        ;;
    "python")
        echo "🐍 Starting Python shell in container..."
        run_in_dev_container python
        ;;
    "install")
        echo "📦 Installing/updating dependencies in container..."
        run_in_dev_container pip install -e .[dev,monitoring]
        print_success "Dependencies updated"
        ;;
    "clean")
        echo "🧹 Cleaning up Docker resources..."
        docker-compose -f docker-compose.yml -f docker-compose.dev.yml down -v --remove-orphans
        docker system prune -f
        echo "🧹 Removing unused images..."
        docker image prune -f
        print_success "Cleanup completed"
        ;;
    "reset")
        echo "🔄 Resetting entire environment..."
        docker-compose -f docker-compose.yml -f docker-compose.dev.yml down -v --remove-orphans
        docker-compose -f docker-compose.yml -f docker-compose.dev.yml build --no-cache
        docker-compose -f docker-compose.yml -f docker-compose.dev.yml up -d
        print_success "Environment reset"
        ;;
    "docs")
        echo "📚 Opening API documentation..."
        if command -v open &> /dev/null; then
            open http://localhost:8000/docs
        elif command -v xdg-open &> /dev/null; then
            xdg-open http://localhost:8000/docs
        else
            echo "Please open http://localhost:8000/docs in your browser"
        fi
        ;;
    "status")
        echo "📊 Service Status:"
        docker-compose -f docker-compose.yml -f docker-compose.dev.yml ps
        ;;
    "exec")
        if [ -z "$2" ]; then
            echo "Usage: ./dev.sh exec <service> [command]"
            echo "Example: ./dev.sh exec api-gateway bash"
        else
            service="$2"
            shift 2
            docker-compose exec "$service" "$@"
        fi
        ;;
    *)
        echo "VisionFlow Development Helper - Docker Edition"
        echo ""
        echo "Usage: ./dev.sh [command]"
        echo ""
        echo "🐳 Docker Service Management:"
        echo "  start     Start all services"
        echo "  stop      Stop all services"
        echo "  restart   Restart all services"
        echo "  build     Build Docker images"
        echo "  rebuild   Rebuild images from scratch"
        echo "  status    Show service status"
        echo "  logs      Show logs (optionally for specific service)"
        echo "  clean     Clean up Docker resources"
        echo "  reset     Reset entire environment"
        echo ""
        echo "🔧 Development Commands (run in containers):"
        echo "  test      Run API tests"
        echo "  pytest    Run pytest with optional arguments"
        echo "  lint      Check code style"
        echo "  format    Format code"
        echo "  shell     Open bash shell in dev container"
        echo "  python    Start Python shell"
        echo "  install   Install/update dependencies"
        echo ""
        echo "🛠️  Utility Commands:"
        echo "  docs      Open API documentation"
        echo "  exec      Execute command in specific service"
        echo ""
        echo "Examples:"
        echo "  ./dev.sh start"
        echo "  ./dev.sh logs api-gateway"
        echo "  ./dev.sh test"
        echo "  ./dev.sh pytest -v"
        echo "  ./dev.sh shell"
        echo "  ./dev.sh exec postgres psql -U visionflow"
        ;;
esac
EOF

chmod +x dev.sh
print_success "Development helper script created"

# Final instructions
echo ""
echo "🎉 VisionFlow setup completed successfully!"
echo ""
echo "📋 Next steps:"
echo "   1. Start all services: ./dev.sh start"
echo "   2. Test the API: ./dev.sh test"
echo "   3. View documentation: ./dev.sh docs"
echo ""
echo "🔗 Service URLs:"
echo "   • API & Docs: http://localhost:8000"
echo "   • Grafana: http://localhost:3000 (admin/admin)"
echo "   • Prometheus: http://localhost:9090"
echo "   • MinIO: http://localhost:9001 (minio/minio123)"
echo ""
echo "💡 Helpful commands:"
echo "   • ./dev.sh start|stop|restart|logs|test|clean"
echo "   • ./dev.sh shell     # Access development container"
echo "   • ./dev.sh lint      # Run code linting in container"
echo "   • ./dev.sh format    # Format code in container"
echo "   • ./dev.sh rebuild   # Rebuild containers from scratch"
echo ""
print_success "Setup complete! Happy coding! 🚀"
