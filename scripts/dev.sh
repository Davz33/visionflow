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
