#!/bin/bash

# Run VisionFlow generation service with volume mounting for local development

set -e

echo "🚀 Building Docker image for volume mounting..."

# Build the image
docker build -f docker/Dockerfile.generation.local -t visionflow-generation:local .

echo "🐋 Running container with volume mounting..."

# Run the container with volume mounting
docker run -it --rm \
  --name visionflow-generation-dev \
  -p 8002:8002 \
  -v "$(pwd):/app" \
  -v "$(pwd)/models:/app/models" \
  -v "$(pwd)/generated:/app/generated" \
  -v "$(pwd)/logs:/app/logs" \
  -e PYTHONPATH=/app/src \
  visionflow-generation:local

echo "✅ Container stopped"
