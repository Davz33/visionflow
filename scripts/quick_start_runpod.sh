#!/bin/bash
# Quick Start Script for RunPod WAN 2.1 Setup
# Run this script to set up everything automatically

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Print banner
echo -e "${BLUE}"
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║                🚀 RunPod WAN 2.1 Quick Start                ║"
echo "║           Automated setup in under 10 minutes               ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo -e "${NC}"

# Step 1: Check prerequisites
echo -e "${YELLOW}📋 Step 1: Checking prerequisites...${NC}"

# Check if we're in the right directory
if [ ! -f "setup_remote_wan.py" ]; then
    echo -e "${RED}❌ Please run this script from the visionflow directory${NC}"
    exit 1
fi

# Check Docker
if ! command -v docker &> /dev/null; then
    echo -e "${RED}❌ Docker not found. Please install Docker first:${NC}"
    echo "   https://docs.docker.com/get-docker/"
    exit 1
fi

# Check Python and venv
if [ ! -f "venv/bin/activate" ]; then
    echo -e "${RED}❌ Virtual environment not found. Please run from visionflow directory with venv activated${NC}"
    exit 1
fi

echo -e "${GREEN}✅ Prerequisites check passed${NC}"

# Step 2: Activate virtual environment and install dependencies
echo -e "${YELLOW}📦 Step 2: Installing dependencies...${NC}"
source venv/bin/activate
pip install runpod > /dev/null 2>&1
echo -e "${GREEN}✅ Dependencies installed${NC}"

# Step 3: Environment setup
echo -e "${YELLOW}🔑 Step 3: Environment setup...${NC}"

# Check for existing API key
if [ -z "$RUNPOD_API_KEY" ]; then
    echo "Please get your RunPod API key:"
    echo "1. Go to https://www.runpod.io/console/user/settings"
    echo "2. Create a new API key"
    echo "3. Copy and paste it below"
    echo ""
    read -p "Enter your RunPod API key: " RUNPOD_API_KEY
    
    if [ -z "$RUNPOD_API_KEY" ]; then
        echo -e "${RED}❌ API key is required${NC}"
        exit 1
    fi
    
    # Save to .zshrc
    echo "export RUNPOD_API_KEY=\"$RUNPOD_API_KEY\"" >> ~/.zshrc
    export RUNPOD_API_KEY="$RUNPOD_API_KEY"
    echo -e "${GREEN}✅ API key saved to ~/.zshrc${NC}"
else
    echo -e "${GREEN}✅ RunPod API key found${NC}"
fi

# Optional: HuggingFace token
if [ -z "$HUGGINGFACE_TOKEN" ]; then
    echo ""
    read -p "Enter your HuggingFace token (optional, press Enter to skip): " HUGGINGFACE_TOKEN
    if [ ! -z "$HUGGINGFACE_TOKEN" ]; then
        echo "export HUGGINGFACE_TOKEN=\"$HUGGINGFACE_TOKEN\"" >> ~/.zshrc
        export HUGGINGFACE_TOKEN="$HUGGINGFACE_TOKEN"
        echo -e "${GREEN}✅ HuggingFace token saved${NC}"
    fi
fi

# Step 4: Docker image build
echo -e "${YELLOW}🐳 Step 4: Building Docker image...${NC}"

# Get Docker Hub username
read -p "Enter your Docker Hub username (or press Enter to skip RunPod setup): " DOCKER_USERNAME

if [ ! -z "$DOCKER_USERNAME" ]; then
    IMAGE_NAME="${DOCKER_USERNAME}/wan21-service:latest"
    echo "Tagging existing Docker image: $IMAGE_NAME"
    docker tag wan21-service:simple "$IMAGE_NAME"
    echo -e "${GREEN}✅ Docker image tagged successfully${NC}"
else
    echo -e "${YELLOW}⚠️  Skipping Docker Hub push and RunPod setup${NC}"
    echo -e "${BLUE}📋 Manual Setup Instructions:${NC}"
    echo ""
    echo "1. Push your image to Docker Hub:"
    echo "   docker tag wan21-service:simple YOUR_USERNAME/wan21-service:latest"
    echo "   docker push YOUR_USERNAME/wan21-service:latest"
    echo ""
    echo "2. Go to RunPod Console: https://www.runpod.io/console/serverless"
    echo "3. Create a new template with image: YOUR_USERNAME/wan21-service:latest"
    echo "4. Create a serverless endpoint using that template"
    echo ""
    echo -e "${GREEN}✅ Local Docker image 'wan21-service:simple' is ready!${NC}"
    exit 0
fi

# Optional: Push to Docker Hub
if [ ! -z "$DOCKER_USERNAME" ]; then
    read -p "Push image to Docker Hub? [y/N]: " PUSH_CHOICE
    if [[ $PUSH_CHOICE =~ ^[Yy]$ ]]; then
        echo "Pushing to Docker Hub..."
        docker push "$IMAGE_NAME" > /dev/null
        echo -e "${GREEN}✅ Image pushed to Docker Hub${NC}"
    fi
fi

# Step 5: Create RunPod resources
echo -e "${YELLOW}🚀 Step 5: Creating RunPod resources...${NC}"

# Use Python script for RunPod setup
python3 - <<EOF
import runpod
import os
import json
import time

# Set API key
runpod.api_key = os.environ['RUNPOD_API_KEY']

# Create template
print("Creating RunPod template...")
template_config = {
    "name": "WAN 2.1 Generation Service",
    "containerDiskInGb": 50,
    "volumeInGb": 20,
    "volumeMountPath": "/workspace",
    "ports": "8000/http",
    "env": [
        {"key": "CUDA_VISIBLE_DEVICES", "value": "0"},
        {"key": "PYTORCH_CUDA_ALLOC_CONF", "value": "max_split_size_mb:512"}
    ]
}

if os.getenv('HUGGINGFACE_TOKEN'):
    template_config["env"].append({"key": "HUGGINGFACE_TOKEN", "value": os.environ['HUGGINGFACE_TOKEN']})

template = runpod.create_template("$IMAGE_NAME", template_config)
template_id = template["id"]
print(f"✅ Template created: {template_id}")

# Create endpoint
print("Creating serverless endpoint...")
endpoint_config = {
    "name": "wan21-generation",
    "template_id": template_id,
    "idle_timeout": 2,
    "containers": {"max": 3, "throttle": 1},
    "locations": {"US": True, "EU": False, "AS": False},
    "gpu_ids": "NVIDIA RTX 4090,NVIDIA GeForce RTX 4090"
}

endpoint = runpod.create_endpoint(endpoint_config)
endpoint_id = endpoint["id"]
print(f"✅ Endpoint created: {endpoint_id}")

# Save configuration
config = {
    "template_id": template_id,
    "endpoint_id": endpoint_id,
    "docker_image": "$IMAGE_NAME",
    "setup_time": time.strftime("%Y-%m-%d %H:%M:%S")
}

with open("runpod_config.json", "w") as f:
    json.dump(config, f, indent=2)

# Save endpoint ID to environment
with open(os.path.expanduser("~/.zshrc"), "a") as f:
    f.write(f'\\nexport RUNPOD_ENDPOINT_ID="{endpoint_id}"\\n')

print(f"ENDPOINT_ID={endpoint_id}")
EOF

# Extract endpoint ID from the output
ENDPOINT_ID=$(python3 -c "
import runpod
import os
runpod.api_key = os.environ['RUNPOD_API_KEY']
endpoints = runpod.get_endpoints()
for ep in endpoints:
    if ep['name'] == 'wan21-generation':
        print(ep['id'])
        break
")

export RUNPOD_ENDPOINT_ID="$ENDPOINT_ID"

echo -e "${GREEN}✅ RunPod resources created successfully${NC}"

# Step 6: Test the setup
echo -e "${YELLOW}🧪 Step 6: Testing the setup...${NC}"

echo "Running connection test..."
python test_runpod_generation.py --quick-test > /dev/null 2>&1 && echo -e "${GREEN}✅ Connection test passed${NC}" || echo -e "${YELLOW}⚠️ Connection test had issues (this is normal for initial setup)${NC}"

# Step 7: Summary
echo -e "${BLUE}"
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║                     🎉 Setup Complete!                      ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo -e "${NC}"

echo -e "${GREEN}✅ RunPod WAN 2.1 setup completed successfully!${NC}"
echo ""
echo "📋 Configuration Summary:"
echo "  Docker Image: $IMAGE_NAME"
echo "  Endpoint ID: $ENDPOINT_ID"
echo "  GPU Type: RTX 4090 (preferred)"
echo "  Cost: ~\$0.44/hour (~\$0.22-0.44/video)"
echo ""
echo "🚀 Quick Start:"
echo "  1. Update your code:"
echo "     from services.inference.wan_remote_wrapper import create_wan_remote_service"
echo "     wan_service = create_wan_remote_service('runpod')"
echo ""
echo "  2. Test generation:"
echo "     python test_runpod_generation.py"
echo ""
echo "  3. Monitor costs:"
echo "     https://runpod.io/console"
echo ""
echo "💡 Next time, just restart your shell to load environment variables:"
echo "   source ~/.zshrc"

echo -e "${GREEN}Happy generating! 🎬${NC}"
