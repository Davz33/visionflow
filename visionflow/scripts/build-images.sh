#!/bin/bash

# Build optimized Docker images for local Kubernetes deployment

set -e

echo "🔨 Building VisionFlow Docker Images for Kubernetes..."

# Set up build context
cd "$(dirname "$0")/.."

# Build API service (minimal health service)
echo "📦 Building API health service..."
docker build -t visionflow-api:local -f docker/Dockerfile.api.minimal .

# Build Generation service (health service)
echo "🎬 Building Generation health service..."
docker build -t visionflow-generation:local -f docker/Dockerfile.generation.health .

# Build Orchestration service (health service)
echo "🎭 Building Orchestration health service..."
docker build -t visionflow-orchestrator:local -f docker/Dockerfile.orchestration.health .

echo "✅ All images built successfully!"

# Show image sizes
echo "📊 Image sizes:"
docker images | grep "visionflow-.*:local" | awk '{printf "%-30s %10s\n", $1":"$2, $7}'

echo ""
echo "🚀 Images are ready for Kubernetes deployment!"
echo "   Run: kubectl apply -f k8s/local/complete-local-deployment.yaml"