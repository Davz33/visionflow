#!/bin/bash

# VisionFlow macOS Deployment Script
# Optimized for macOS with Apple Silicon support

set -e

echo "🍎 VisionFlow macOS Deployment"
echo "=============================="

# Check macOS and Apple Silicon
check_macos() {
    if [[ "$OSTYPE" != "darwin"* ]]; then
        echo "❌ This script is designed for macOS"
        exit 1
    fi
    
    # Check for Apple Silicon
    ARCH=$(uname -m)
    if [[ "$ARCH" == "arm64" ]]; then
        echo "✅ Apple Silicon detected (M1/M2/M3)"
        USE_MPS=true
    else
        echo "✅ Intel Mac detected"
        USE_MPS=false
    fi
}

# Load environment for macOS
load_macos_environment() {
    echo "🔧 Loading macOS environment configuration..."
    
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
        echo "📝 Creating macOS-optimized .env file template..."
        cat > .env << EOF
# VisionFlow macOS Environment

# HuggingFace Token (required)
HUGGINGFACE_TOKEN=your_huggingface_token_here

# Device optimization for macOS
MODEL_DEVICE=$([ "$USE_MPS" = true ] && echo "mps" || echo "cpu")
MAX_CONCURRENT_GENERATIONS=1

# Model configuration
WAN_MODEL_PATH=multimodalart/wan2-1-fast
MODEL_CACHE_DIR=/app/models

# Optional cloud integration
VERTEX_AI_PROJECT=visionflow-gcp-project
LANGCHAIN_API_KEY=your_langchain_api_key_here

# Environment
ENVIRONMENT=local
LOG_LEVEL=info

# Performance settings for macOS
PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0
EOF
        echo "⚠️  Created .env template. Please edit it with your actual tokens:"
        echo "   Required: HUGGINGFACE_TOKEN"
        echo "   Get your token from: https://huggingface.co/settings/tokens"
        echo ""
        echo "   Then run the script again: ./scripts/deploy-macos.sh"
        exit 1
    fi
    
    # Validate required environment variables
    if [ -z "$HUGGINGFACE_TOKEN" ] || [ "$HUGGINGFACE_TOKEN" = "your_huggingface_token_here" ]; then
        echo "❌ HUGGINGFACE_TOKEN is required in .env file"
        echo "   Get your token from: https://huggingface.co/settings/tokens"
        exit 1
    fi
    
    # Set macOS-optimized defaults
    export MODEL_DEVICE=${MODEL_DEVICE:-$([ "$USE_MPS" = true ] && echo "mps" || echo "cpu")}
    export MAX_CONCURRENT_GENERATIONS=${MAX_CONCURRENT_GENERATIONS:-"1"}
    export WAN_MODEL_PATH=${WAN_MODEL_PATH:-"multimodalart/wan2-1-fast"}
    export MODEL_CACHE_DIR=${MODEL_CACHE_DIR:-"/app/models"}
    export VERTEX_AI_PROJECT=${VERTEX_AI_PROJECT:-"visionflow-gcp-project"}
    export LANGCHAIN_API_KEY=${LANGCHAIN_API_KEY:-""}
    export ENVIRONMENT=${ENVIRONMENT:-"local"}
    export LOG_LEVEL=${LOG_LEVEL:-"info"}
    export PYTORCH_MPS_HIGH_WATERMARK_RATIO=${PYTORCH_MPS_HIGH_WATERMARK_RATIO:-"0.0"}
    
    echo "🔑 Using environment variables:"
    echo "   HUGGINGFACE_TOKEN: ${HUGGINGFACE_TOKEN:0:10}..." # Only show first 10 chars
    echo "   MODEL_DEVICE: $MODEL_DEVICE"
    echo "   WAN_MODEL_PATH: $WAN_MODEL_PATH"
    echo "   VERTEX_AI_PROJECT: $VERTEX_AI_PROJECT"
    
    echo "✅ macOS environment setup completed"
}

# Deploy services optimized for macOS
deploy_macos_services() {
    echo "🏗️  Building and starting macOS-optimized services..."
    
    # Create macOS-specific docker-compose override
    cat > docker-compose.macos.yml << EOF
version: '3.8'

services:
  generation-service:
    environment:
      - MODEL_DEVICE=$([ "$USE_MPS" = true ] && echo "mps" || echo "cpu")
      - MAX_CONCURRENT_GENERATIONS=1
      - PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0
    volumes:
      - ./models:/app/models
      - ./generated:/app/generated
      - ./logs:/app/logs
    deploy:
      resources:
        limits:
          memory: 8G
        reservations:
          memory: 4G

  api-gateway:
    environment:
      - MODEL_DEVICE=$([ "$USE_MPS" = true ] && echo "mps" || echo "cpu")
EOF
    
    # Stop any existing services
    echo "🛑 Stopping existing services..."
    docker compose down -v 2>/dev/null || true
    
    # Build images
    echo "🔨 Building Docker images for macOS..."
    docker compose -f docker-compose.yml -f docker-compose.macos.yml build --parallel
    
    # Start services
    echo "🚀 Starting macOS-optimized services..."
    docker compose -f docker-compose.yml -f docker-compose.macos.yml up -d
    
    echo "✅ macOS services deployment completed"
}

# Wait for services with macOS-specific timeouts
wait_for_macos_services() {
    echo "⏳ Waiting for services to be ready..."
    
    # Wait for API (longer timeout for Apple Silicon compilation)
    echo "   🔗 Waiting for API service..."
    timeout 600 bash -c 'until curl -f http://localhost:8000/health &>/dev/null; do sleep 5; done' || {
        echo "❌ API service failed to start"
        echo "💡 Try checking logs: docker compose logs api-gateway"
        exit 1
    }
    
    # Wait for Generation service (even longer for model loading)
    echo "   🎮 Waiting for Generation service..."
    timeout 900 bash -c 'until curl -f http://localhost:8002/health &>/dev/null; do sleep 10; done' || {
        echo "❌ Generation service failed to start"
        echo "💡 Try checking logs: docker compose logs generation-service"
        echo "💡 Model download can take 10+ minutes on first run"
        exit 1
    }
    
    echo "✅ All services are ready"
}

# Show macOS-specific status
show_macos_status() {
    echo ""
    echo "🎉 VisionFlow macOS Deployment Complete!"
    echo "======================================="
    echo ""
    echo "🖥️  System: macOS $(sw_vers -productVersion) ($ARCH)"
    echo "🧠 Compute: $([ "$USE_MPS" = true ] && echo "Apple Silicon MPS" || echo "Intel CPU")"
    echo ""
    echo "📊 Service URLs:"
    echo "   API Gateway:     http://localhost:8000"
    echo "   API Docs:        http://localhost:8000/docs"
    echo "   Generation:      http://localhost:8002"
    echo "   Generation Docs: http://localhost:8002/docs"
    echo "   Grafana:         http://localhost:3000 (admin/admin)"
    echo "   MinIO Console:   http://localhost:9001 (minio/minio123)"
    echo ""
    echo "🧪 Test video generation:"
    echo "   curl -X POST http://localhost:8000/api/v1/generate \\"
    echo "        -H 'Content-Type: application/json' \\"
    echo "        -d '{\"prompt\": \"A peaceful mountain lake\", \"duration\": 3}'"
    echo ""
    echo "⚠️  Note: First video generation will take longer due to model download"
    echo "    (~2-5GB download + compilation time)"
    echo ""
    echo "🛠️  Management:"
    echo "   View logs:    docker compose logs -f"
    echo "   Stop:         docker compose down"
    echo "   Monitor CPU:  top -pid \$(docker inspect -f '{{.State.Pid}}' visionflow-generation)"
}

# Main execution
main() {
    check_macos
    load_macos_environment
    deploy_macos_services
    wait_for_macos_services
    show_macos_status
}

# Help
if [ "$1" = "--help" ] || [ "$1" = "-h" ]; then
    echo "VisionFlow macOS Deployment Script"
    echo ""
    echo "Optimized for macOS with Apple Silicon (MPS) support"
    echo ""
    echo "Usage: $0 [options]"
    echo ""
    echo "Options:"
    echo "  --help, -h     Show this help message"
    echo "  --stop         Stop all services"
    echo "  --status       Show service status"
    echo ""
    echo "Requirements:"
    echo "  - Docker Desktop for Mac"
    echo "  - HuggingFace account and token"
    echo ""
    exit 0
fi

# Stop services
if [ "$1" = "--stop" ]; then
    echo "🛑 Stopping VisionFlow services..."
    docker compose down -v
    rm -f docker-compose.macos.yml
    echo "✅ Services stopped"
    exit 0
fi

# Show status
if [ "$1" = "--status" ]; then
    echo "📊 VisionFlow macOS Status:"
    docker compose ps
    echo ""
    echo "Device status:"
    curl -s http://localhost:8002/models/status 2>/dev/null | jq '.device' || echo "Service not ready"
    exit 0
fi

# Run main deployment
main
