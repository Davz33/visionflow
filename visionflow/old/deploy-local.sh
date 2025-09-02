#!/bin/bash

# VisionFlow Local Deployment Script
# Deploys VisionFlow with GPU support using Docker Compose

set -e

echo "🚀 VisionFlow Local Deployment"
echo "=============================="

# Check prerequisites
check_prerequisites() {
    echo "📋 Checking prerequisites..."
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        echo "❌ Docker is not installed. Please install Docker first."
        exit 1
    fi
    
    # Check Docker Compose
    if ! docker compose version &> /dev/null; then
        echo "❌ Docker Compose is not available. Please install Docker Compose."
        exit 1
    fi
    
    # Check NVIDIA Docker (optional but recommended)
    if command -v nvidia-docker &> /dev/null || docker run --rm --gpus all nvidia/cuda:11.8-base nvidia-smi &> /dev/null; then
        echo "✅ NVIDIA Docker support detected"
        GPU_SUPPORT=true
    else
        echo "⚠️  No GPU support detected. Will run in CPU mode."
        GPU_SUPPORT=false
    fi
    
    echo "✅ Prerequisites check completed"
}

# Setup environment
load_environment() {
    echo "🔧 Loading environment configuration..."
    
    # Create required directories
    mkdir -p models generated logs
    
    # Load .env file if it exists
    if [ -f .env ]; then
        echo "📄 Loading variables from .env file..."
        set -a  # automatically export all variables
        source .env
        set +a  # stop automatically exporting
        echo "✅ Environment variables loaded from .env"
    else
        echo "📝 Creating .env file template..."
        cat > .env << EOF
# VisionFlow Local Environment Configuration

# HuggingFace Token (required for model downloads)
HUGGINGFACE_TOKEN=your_huggingface_token_here

# GCP Project (for Vertex AI integration)
VERTEX_AI_PROJECT=visionflow-gcp-project

# LangChain API Key (optional)
LANGCHAIN_API_KEY=your_langchain_api_key_here

# Environment
ENVIRONMENT=local
LOG_LEVEL=debug

# Model Configuration
WAN_MODEL_PATH=multimodalart/wan2-1-fast
MODEL_DEVICE=auto
MAX_CONCURRENT_GENERATIONS=1
EOF
        echo "⚠️  Created .env template. Please edit it with your actual tokens:"
        echo "   Required: HUGGINGFACE_TOKEN"
        echo "   Optional: VERTEX_AI_PROJECT, LANGCHAIN_API_KEY"
        echo ""
        echo "   Then run the script again: ./scripts/deploy-local.sh"
        exit 1
    fi
    
    # Validate required environment variables
    if [ -z "$HUGGINGFACE_TOKEN" ] || [ "$HUGGINGFACE_TOKEN" = "your_huggingface_token_here" ]; then
        echo "❌ HUGGINGFACE_TOKEN is required in .env file"
        echo "   Get your token from: https://huggingface.co/settings/tokens"
        exit 1
    fi
    
    # Set defaults for optional variables
    export VERTEX_AI_PROJECT=${VERTEX_AI_PROJECT:-"visionflow-local"}
    export LANGCHAIN_API_KEY=${LANGCHAIN_API_KEY:-""}
    export ENVIRONMENT=${ENVIRONMENT:-"local"}
    export LOG_LEVEL=${LOG_LEVEL:-"debug"}
    export WAN_MODEL_PATH=${WAN_MODEL_PATH:-"multimodalart/wan2-1-fast"}
    export MODEL_DEVICE=${MODEL_DEVICE:-"auto"}
    export MAX_CONCURRENT_GENERATIONS=${MAX_CONCURRENT_GENERATIONS:-"1"}
    
    echo "🔑 Using environment variables:"
    echo "   HUGGINGFACE_TOKEN: ${HUGGINGFACE_TOKEN:0:10}..." # Only show first 10 chars
    echo "   VERTEX_AI_PROJECT: $VERTEX_AI_PROJECT"
    echo "   WAN_MODEL_PATH: $WAN_MODEL_PATH"
    echo "   MODEL_DEVICE: $MODEL_DEVICE"
    
    # Create GCP service account directory
    mkdir -p ~/.gcp
    
    if [ ! -f ~/.gcp/visionflow-service-account.json ]; then
        echo "📝 Creating placeholder GCP service account file..."
        cat > ~/.gcp/visionflow-service-account.json << EOF
{
  "type": "service_account",
  "project_id": "${VERTEX_AI_PROJECT}",
  "private_key_id": "placeholder",
  "private_key": "-----BEGIN PRIVATE KEY-----\nPLACEHOLDER\n-----END PRIVATE KEY-----\n",
  "client_email": "visionflow@${VERTEX_AI_PROJECT}.iam.gserviceaccount.com",
  "client_id": "placeholder",
  "auth_uri": "https://accounts.google.com/o/oauth2/auth",
  "token_uri": "https://oauth2.googleapis.com/token"
}
EOF
        echo "⚠️  Please replace ~/.gcp/visionflow-service-account.json with your actual GCP service account key"
        echo "   This is only needed if you want to use Vertex AI integration"
    fi
    
    echo "✅ Environment setup completed"
}

# Build and start services
deploy_services() {
    echo "🏗️  Building and starting services..."
    
    # Choose the appropriate docker-compose file
    if [ "$GPU_SUPPORT" = true ]; then
        echo "🎮 Using GPU-accelerated configuration"
        COMPOSE_FILE="docker-compose.local.yml"
    else
        echo "💻 Using CPU-only configuration"
        # Use the standard docker-compose.yml which is already CPU-only
        COMPOSE_FILE="docker-compose.yml"
    fi
    
    # Stop any existing services
    echo "🛑 Stopping existing services..."
    docker compose -f $COMPOSE_FILE down -v 2>/dev/null || true
    
    # Build images
    echo "🔨 Building Docker images..."
    docker compose -f $COMPOSE_FILE build --parallel
    
    # Start services
    echo "🚀 Starting services..."
    docker compose -f $COMPOSE_FILE up -d
    
    echo "✅ Services deployment completed"
}

# Wait for services to be ready
wait_for_services() {
    echo "⏳ Waiting for services to be ready..."
    
    # Wait for API
    echo "   🔗 Waiting for API service..."
    timeout 300 bash -c 'until curl -f http://localhost:8000/health &>/dev/null; do sleep 5; done' || {
        echo "❌ API service failed to start"
        exit 1
    }
    
    # Wait for Generation service (if GPU support)
    if [ "$GPU_SUPPORT" = true ]; then
        echo "   🎮 Waiting for Generation service..."
        timeout 600 bash -c 'until curl -f http://localhost:8002/health &>/dev/null; do sleep 10; done' || {
            echo "❌ Generation service failed to start"
            exit 1
        }
    fi
    
    echo "✅ All services are ready"
}

# Display status and URLs
show_status() {
    echo ""
    echo "🎉 VisionFlow Deployment Complete!"
    echo "=================================="
    echo ""
    echo "📊 Service Status:"
    echo "   API Gateway:     http://localhost:8000"
    echo "   API Docs:        http://localhost:8000/docs"
    echo "   Health Check:    http://localhost:8000/health"
    
    if [ "$GPU_SUPPORT" = true ]; then
        echo "   Generation:      http://localhost:8002"
        echo "   Generation Docs: http://localhost:8002/docs"
    fi
    
    echo "   Grafana:         http://localhost:3000 (admin/admin)"
    echo "   Prometheus:      http://localhost:9090"
    echo "   MinIO Console:   http://localhost:9001 (minio/minio123)"
    echo ""
    echo "📁 Data Directories:"
    echo "   Models:          ./models"
    echo "   Generated:       ./generated"
    echo "   Logs:           ./logs"
    echo ""
    echo "🧪 Test the API:"
    echo "   curl -X POST http://localhost:8000/api/v1/generate \\"
    echo "        -H 'Content-Type: application/json' \\"
    echo "        -d '{\"prompt\": \"A beautiful sunset over mountains\", \"duration\": 5}'"
    echo ""
    echo "📝 View logs:"
    echo "   docker compose -f $COMPOSE_FILE logs -f"
    echo ""
    echo "🛑 Stop services:"
    echo "   docker compose -f $COMPOSE_FILE down"
}

# Main execution
main() {
    check_prerequisites
    load_environment
    deploy_services
    wait_for_services
    show_status
}

# Help message
if [ "$1" = "--help" ] || [ "$1" = "-h" ]; then
    echo "VisionFlow Local Deployment Script"
    echo ""
    echo "Usage: $0 [options]"
    echo ""
    echo "Options:"
    echo "  --help, -h     Show this help message"
    echo "  --stop        Stop all services"
    echo "  --status      Show service status"
    echo ""
    echo "Environment variables:"
    echo "  HUGGINGFACE_TOKEN    Required for model downloads"
    echo "  VERTEX_AI_PROJECT    GCP project for Vertex AI (optional)"
    echo "  LANGCHAIN_API_KEY    LangChain API key (optional)"
    echo ""
    exit 0
fi

# Stop services
if [ "$1" = "--stop" ]; then
    echo "🛑 Stopping VisionFlow services..."
    docker compose -f docker-compose.local.yml down -v
    echo "✅ Services stopped"
    exit 0
fi

# Show status
if [ "$1" = "--status" ]; then
    echo "📊 VisionFlow Service Status:"
    docker compose -f docker-compose.local.yml ps
    exit 0
fi

# Run main deployment
main
