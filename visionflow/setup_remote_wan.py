#!/usr/bin/env python3
"""
Setup script for WAN 2.1 remote execution
Demonstrates how to migrate existing code with minimal changes
"""

import asyncio
import os
import sys
from pathlib import Path
import yaml
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def demo_local_vs_remote():
    """Demo showing identical code running locally vs remotely"""
    
    print("🎬 WAN 2.1 Remote Execution Demo")
    print("=" * 50)
    
    # Your existing VideoGenerationRequest (simulated)
    class MockVideoGenerationRequest:
        def __init__(self):
            self.prompt = "A cat dancing in the rain, cinematic, 4K"
            self.duration = 5
            self.fps = 24
            self.resolution = "512x512" 
            self.quality = "medium"
            self.guidance_scale = 7.5
            self.num_inference_steps = 25
            self.seed = 42
    
    request = MockVideoGenerationRequest()
    
    # Load remote execution wrapper
    try:
        from services.inference.wan_remote_wrapper import DistributedWANService, create_wan_remote_service
        
        print("✅ Remote WAN wrapper loaded successfully")
        
        # Option 1: Drop-in replacement (auto-selects best backend)
        print("\n🔄 Testing auto-selection...")
        distributed_wan = DistributedWANService(prefer_remote=False, fallback_local=True)
        
        # Option 2: Explicit backend selection
        print("\n🔄 Testing explicit backend selection...")
        
        backends_to_test = ["local"]  # Start with local
        
        # Add available remote backends
        try:
            import ray
            backends_to_test.append("ray")
            print("✅ Ray available")
        except ImportError:
            print("❌ Ray not available (install with: pip install ray)")
        
        try:
            import modal
            backends_to_test.append("modal") 
            print("✅ Modal available")
        except ImportError:
            print("❌ Modal not available (install with: pip install modal)")
        
        for backend in backends_to_test:
            print(f"\n🎯 Testing {backend.upper()} backend...")
            
            try:
                wan_service = create_wan_remote_service(backend)
                print(f"✅ {backend} service created")
                
                # This would be your exact existing call:
                # result = await wan_service.generate_video_remote(request)
                # print(f"✅ Generation completed with {backend}")
                
            except Exception as e:
                print(f"❌ {backend} failed: {e}")
        
        print("\n🎉 Demo completed!")
        
    except ImportError as e:
        print(f"❌ Could not import WAN remote wrapper: {e}")
        return False
    
    return True

def setup_ray_cluster():
    """Setup instructions for Ray cluster"""
    print("\n🔧 Ray Cluster Setup")
    print("=" * 30)
    print("""
To use Ray for distributed WAN 2.1 execution:

1. Install Ray:
   pip install ray[default]

2. Start Ray cluster:
   # On head node (CPU machine):
   ray start --head --port=6379
   
   # On GPU workers:
   ray start --address='head-node-ip:6379'

3. Set environment variable:
   export RAY_ADDRESS="ray://head-node-ip:10001"

4. Your code stays the same:
   wan_service = create_wan_remote_service("ray")
   result = await wan_service.generate_video_remote(request)
""")

def setup_modal():
    """Setup instructions for Modal"""
    print("\n☁️  Modal Setup")
    print("=" * 20)
    print("""
To use Modal for serverless WAN 2.1 execution:

1. Install Modal:
   pip install modal

2. Create account and authenticate:
   modal token new

3. Deploy your WAN service:
   modal deploy services/inference/wan_remote_wrapper.py

4. Your code stays the same:
   wan_service = create_wan_remote_service("modal")
   result = await wan_service.generate_video_remote(request)

Costs: ~$1-3 per video (A100 GPU time)
""")

def setup_runpod():
    """Setup instructions for RunPod"""
    print("\n🚀 RunPod Setup")
    print("=" * 20)
    print("""
To use RunPod for GPU rental:

1. Create RunPod account: https://www.runpod.io/

2. Install RunPod SDK:
   pip install runpod

3. Create endpoint with your WAN Docker image:
   - Use our provided Dockerfile
   - Deploy as serverless endpoint
   - Get endpoint ID

4. Set environment variables:
   export RUNPOD_API_KEY="your-api-key"
   export RUNPOD_ENDPOINT_ID="your-endpoint-id"

5. Your code stays the same:
   wan_service = create_wan_remote_service("runpod")
   result = await wan_service.generate_video_remote(request)

Costs: ~$0.50-2.00 per video depending on GPU
""")

def create_docker_image():
    """Create Dockerfile for containerizing WAN service"""
    dockerfile_content = """# Dockerfile for WAN 2.1 Remote Execution
FROM nvidia/cuda:11.8-devel-ubuntu22.04

# Install Python and system dependencies
RUN apt-get update && apt-get install -y \\
    python3.10 \\
    python3-pip \\
    git \\
    wget \\
    curl \\
    && rm -rf /var/lib/apt/lists/*

# Set Python version
RUN ln -s /usr/bin/python3.10 /usr/bin/python

# Install PyTorch and dependencies
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install WAN 2.1 dependencies  
RUN pip install \\
    diffusers>=0.21.0 \\
    transformers>=4.25.0 \\
    accelerate>=0.24.0 \\
    huggingface_hub>=0.19.0 \\
    opencv-python \\
    pillow \\
    numpy \\
    scipy

# Copy your WAN service code
COPY . /app
WORKDIR /app

# Install your package
RUN pip install -e .

# Set environment variables
ENV CUDA_VISIBLE_DEVICES=0
ENV HUGGINGFACE_HUB_CACHE=/app/models

# Expose port for API
EXPOSE 8000

# Run your WAN service
CMD ["python", "-m", "visionflow.services.generation.wan_video_service"]
"""
    
    with open("Dockerfile.wan", "w") as f:
        f.write(dockerfile_content)
    
    print("✅ Created Dockerfile.wan for containerized deployment")

def show_migration_examples():
    """Show exact migration examples"""
    print("\n📝 Migration Examples")
    print("=" * 25)
    print("""
BEFORE (Your existing code):
──────────────────────────
from visionflow.services.generation.wan_video_service import WanVideoGenerationService

wan_service = WanVideoGenerationService()
result = await wan_service.generate_video(request)

AFTER (Distributed execution):
─────────────────────────────
from services.inference.wan_remote_wrapper import DistributedWANService

# Option 1: Auto-select best backend (local fallback)
wan_service = DistributedWANService(prefer_remote=True)
result = await wan_service.generate_video(request)  # Same interface!

# Option 2: Explicit backend selection
from services.inference.wan_remote_wrapper import create_wan_remote_service

wan_service = create_wan_remote_service("ray")  # or "modal", "runpod"
result = await wan_service.generate_video_remote(request)

BENEFITS:
────────
✅ Keep your exact WAN 2.1 model and code
✅ Zero changes to your VideoGenerationRequest
✅ Same return format and error handling  
✅ Automatic fallback to local execution
✅ Cost optimization and monitoring
✅ Easy switching between backends
""")

def load_config():
    """Load remote execution configuration"""
    config_path = Path("config/remote_execution_config.yaml")
    
    if not config_path.exists():
        print(f"❌ Config file not found: {config_path}")
        return None
    
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        print(f"✅ Loaded config from {config_path}")
        return config
    except Exception as e:
        print(f"❌ Error loading config: {e}")
        return None

async def main():
    """Main setup and demo function"""
    print("🎬 WAN 2.1 Remote Execution Setup")
    print("=" * 40)
    
    # Load configuration
    config = load_config()
    if config:
        backend = config['execution']['preferred_backend']
        print(f"📋 Preferred backend: {backend}")
    
    # Run demo
    success = await demo_local_vs_remote()
    
    if success:
        print("\n🎉 Setup successful!")
    else:
        print("\n❌ Setup had issues")
    
    # Show setup instructions
    print("\n📚 Setup Instructions:")
    setup_ray_cluster()
    setup_modal()
    setup_runpod()
    
    # Show migration examples
    show_migration_examples()
    
    # Create Docker image
    create_docker_image()
    
    print("\n🚀 Next Steps:")
    print("1. Choose your preferred backend (Ray, Modal, or RunPod)")
    print("2. Follow the setup instructions above")
    print("3. Replace your WAN service import with DistributedWANService")
    print("4. Your existing code will work unchanged!")
    print("\n💡 Pro tip: Start with 'local' backend to test, then switch to remote")

if __name__ == "__main__":
    asyncio.run(main())
