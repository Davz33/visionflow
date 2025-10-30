#!/bin/bash

# Build standalone Docker images for individual service testing

set -e

echo "🔨 Building VisionFlow Standalone Docker Images..."

# Set up build context
cd "$(dirname "$0")/.."

# Build Generation service (full ML service)
echo "🎬 Building Generation service (full ML)..."
docker build -t visionflow-generation:local -f docker/Dockerfile.generation.local .

echo "✅ Standalone image built successfully!"

echo "📊 Image size:"
docker images | grep "visionflow-generation:local" | awk '{printf "%-30s %10s\n", $1":"$2, $7}'

echo ""
echo "🚀 Image is ready for standalone deployment!"
echo "   Deploy: kubectl apply -f k8s/local/standalone/generation-service.yaml"
