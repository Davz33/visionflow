#!/bin/bash

# Build standalone Docker images for individual service testing

set -e

echo "🔨 Building VisionFlow Standalone Docker Images..."

# Set up build context
cd "$(dirname "$0")/.."

# Build Generation health service (minimal, no ML dependencies)
echo "🎬 Building Generation health service..."
docker build -t visionflow-generation-health:local -f docker/Dockerfile.generation.health .

# Build Generation service (full ML service)
echo "🎬 Building Generation service (full ML)..."
docker build -t visionflow-generation:local -f docker/Dockerfile.generation.local .

echo "✅ Standalone images built successfully!"

# Show image sizes
echo "📊 Image sizes:"
docker images | grep "visionflow-generation.*:local" | awk '{printf "%-30s %10s\n", $1":"$2, $7}'

echo ""
echo "🚀 Images are ready for standalone deployment!"
echo "   Health service: kubectl apply -f k8s/local/standalone/generation-health-service.yaml"
echo "   Full service:   kubectl apply -f k8s/local/standalone/generation-service.yaml"
