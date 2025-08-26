#!/bin/bash
# RunPod Setup Script - Run this on the RunPod pod instead of uploading zip

echo "🚀 Setting up WAN 2.1 on RunPod..."

# Create workspace
mkdir -p /workspace/wan21
cd /workspace/wan21

# Create Dockerfile.runpod
cat > Dockerfile.runpod << 'EOF'
# Optimized Dockerfile for RunPod PyTorch Image
FROM runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04

WORKDIR /workspace
COPY . /workspace/

RUN pip install --no-cache-dir \
    diffusers>=0.21.0 \
    transformers>=4.25.0 \
    accelerate>=0.24.0 \
    huggingface_hub>=0.19.0 \
    opencv-python-headless \
    pillow \
    runpod \
    fastapi \
    uvicorn \
    pydantic

RUN pip install -e .

ENV CUDA_VISIBLE_DEVICES=0
ENV PYTHONPATH=/workspace
ENV HUGGINGFACE_HUB_CACHE=/workspace/models
ENV PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
ENV TOKENIZERS_PARALLELISM=false

RUN mkdir -p /workspace/models
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=30s --start-period=60s --retries=3 \
    CMD python -c "import torch; print('CUDA available:', torch.cuda.is_available())" || exit 1

CMD ["python", "/workspace/scripts/runpod_handler.py"]
EOF

# Create basic requirements.txt
cat > requirements.txt << 'EOF'
torch>=2.0.0
diffusers>=0.21.0
transformers>=4.25.0
accelerate>=0.24.0
huggingface_hub>=0.19.0
opencv-python-headless
pillow
runpod
fastapi
uvicorn
pydantic
EOF

# Create pyproject.toml
cat > pyproject.toml << 'EOF'
[build-system]
requires = ["setuptools>=45", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "visionflow"
version = "0.1.0"
description = "Vision Flow - T2V Generation and Evaluation"
dependencies = [
    "torch>=2.0.0",
    "diffusers>=0.21.0", 
    "transformers>=4.25.0",
    "accelerate>=0.24.0",
    "huggingface_hub>=0.19.0",
    "opencv-python-headless",
    "pillow",
    "runpod",
    "fastapi",
    "uvicorn",
    "pydantic"
]

[project.optional-dependencies]
dev = ["pytest", "black", "flake8"]
EOF

echo "✅ Essential files created!"
echo "Next steps:"
echo "1. Copy your visionflow/ source code to this directory"
echo "2. Run: docker build -f Dockerfile.runpod -t wan21-service ."
echo "3. Run: docker run -d --gpus all -p 8000:8000 wan21-service"
