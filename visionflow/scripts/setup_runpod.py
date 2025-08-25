#!/usr/bin/env python3
"""
Automated RunPod Setup Script for WAN 2.1
Handles the complete setup process with guided steps
"""

import os
import sys
import json
import time
import asyncio
import subprocess
from pathlib import Path
from typing import Dict, Any, Optional
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RunPodSetup:
    """Automated RunPod setup for WAN 2.1 generation"""
    
    def __init__(self):
        self.api_key = None
        self.template_id = None
        self.endpoint_id = None
        self.docker_image = None
        self.config = {}
        
    def print_banner(self):
        """Print setup banner"""
        print("""
╔══════════════════════════════════════════════════════════════╗
║                 🚀 RunPod WAN 2.1 Setup                     ║
║         Automated setup for remote WAN generation           ║
╚══════════════════════════════════════════════════════════════╝
        """)
    
    def check_prerequisites(self) -> bool:
        """Check if prerequisites are installed"""
        print("📋 Checking prerequisites...")
        
        # Check Docker
        try:
            subprocess.run(["docker", "--version"], capture_output=True, check=True)
            print("✅ Docker installed")
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("❌ Docker not found. Please install Docker first.")
            print("   https://docs.docker.com/get-docker/")
            return False
        
        # Check Python packages
        try:
            import runpod
            print("✅ RunPod SDK available")
        except ImportError:
            print("⚠️  RunPod SDK not found. Installing...")
            subprocess.run([sys.executable, "-m", "pip", "install", "runpod"], check=True)
            print("✅ RunPod SDK installed")
        
        return True
    
    def get_api_key(self) -> str:
        """Get or prompt for RunPod API key"""
        # Check environment variable first
        api_key = os.getenv('RUNPOD_API_KEY')
        
        if not api_key:
            print("\n🔑 RunPod API Key Setup")
            print("1. Go to https://www.runpod.io/console/user/settings")
            print("2. Create a new API key")
            print("3. Copy the key and paste it below")
            
            api_key = input("\nEnter your RunPod API key: ").strip()
            
            if not api_key:
                raise ValueError("API key is required")
            
            # Save to environment
            with open(os.path.expanduser("~/.zshrc"), "a") as f:
                f.write(f'\nexport RUNPOD_API_KEY="{api_key}"\n')
            
            print("✅ API key saved to ~/.zshrc")
        
        self.api_key = api_key
        os.environ['RUNPOD_API_KEY'] = api_key
        return api_key
    
    def get_huggingface_token(self) -> str:
        """Get or prompt for HuggingFace token"""
        hf_token = os.getenv('HUGGINGFACE_TOKEN')
        
        if not hf_token:
            print("\n🤗 HuggingFace Token Setup")
            print("1. Go to https://huggingface.co/settings/tokens")
            print("2. Create a token with 'Read' access")
            print("3. Copy the token and paste it below")
            
            hf_token = input("\nEnter your HuggingFace token (or press Enter to skip): ").strip()
            
            if hf_token:
                with open(os.path.expanduser("~/.zshrc"), "a") as f:
                    f.write(f'\nexport HUGGINGFACE_TOKEN="{hf_token}"\n')
                print("✅ HuggingFace token saved")
            else:
                print("⚠️  Skipping HuggingFace token (some models may not work)")
        
        return hf_token or ""
    
    def build_docker_image(self) -> str:
        """Build and optionally push Docker image"""
        print("\n🐳 Building Docker Image...")
        
        # Get Docker Hub username
        docker_username = input("Enter your Docker Hub username (or press Enter to skip push): ").strip()
        
        if docker_username:
            image_name = f"{docker_username}/wan21-service:latest"
        else:
            image_name = "wan21-service:latest"
        
        # Build image
        try:
            print(f"Building image: {image_name}")
            result = subprocess.run([
                "docker", "build", 
                "-f", "Dockerfile.wan",
                "-t", image_name,
                "."
            ], check=True, capture_output=True, text=True)
            
            print("✅ Docker image built successfully")
            
            # Push to Docker Hub if username provided
            if docker_username:
                push_choice = input("Push to Docker Hub? [y/N]: ").strip().lower()
                if push_choice in ['y', 'yes']:
                    print("Pushing image to Docker Hub...")
                    subprocess.run(["docker", "push", image_name], check=True)
                    print("✅ Image pushed to Docker Hub")
            
            self.docker_image = image_name
            return image_name
            
        except subprocess.CalledProcessError as e:
            print(f"❌ Docker build failed: {e}")
            print("stderr:", e.stderr)
            raise
    
    def create_template(self) -> str:
        """Create RunPod template"""
        print("\n📋 Creating RunPod Template...")
        
        import runpod
        
        hf_token = self.get_huggingface_token()
        
        template_config = {
            "name": "WAN 2.1 Generation Service",
            "imageName": self.docker_image,
            "dockerArgs": "",
            "containerDiskInGb": 50,
            "volumeInGb": 20,
            "volumeMountPath": "/workspace",
            "ports": "8000/http,22/tcp",
            "env": [
                {"key": "CUDA_VISIBLE_DEVICES", "value": "0"},
                {"key": "PYTORCH_CUDA_ALLOC_CONF", "value": "max_split_size_mb:512"}
            ]
        }
        
        if hf_token:
            template_config["env"].append({"key": "HUGGINGFACE_TOKEN", "value": hf_token})
        
        try:
            template = runpod.create_template(template_config)
            template_id = template["id"]
            
            print(f"✅ Template created: {template_id}")
            self.template_id = template_id
            return template_id
            
        except Exception as e:
            print(f"❌ Template creation failed: {e}")
            raise
    
    def create_endpoint(self) -> str:
        """Create RunPod serverless endpoint"""
        print("\n🚀 Creating Serverless Endpoint...")
        
        import runpod
        
        # Get GPU preference
        print("\nSelect GPU type:")
        print("1. RTX 4090 (Recommended - $0.44/hr)")
        print("2. RTX A6000 ($0.79/hr)")
        print("3. A100 80GB ($1.19/hr)")
        
        gpu_choice = input("Enter choice [1]: ").strip() or "1"
        
        gpu_options = {
            "1": "NVIDIA RTX 4090,NVIDIA GeForce RTX 4090",
            "2": "NVIDIA RTX A6000",
            "3": "NVIDIA A100 80GB PCIe,NVIDIA A100-SXM4-80GB"
        }
        
        endpoint_config = {
            "name": "wan21-generation",
            "template_id": self.template_id,
            "network_volume_id": None,
            "locations": {
                "US": True,
                "EU": False,
                "AS": False
            },
            "idle_timeout": 2,  # Minutes before auto-shutdown
            "containers": {
                "max": 3,  # Maximum concurrent containers
                "throttle": 1  # Requests before scaling
            },
            "gpu_ids": gpu_options.get(gpu_choice, gpu_options["1"])
        }
        
        try:
            endpoint = runpod.create_endpoint(endpoint_config)
            endpoint_id = endpoint["id"]
            
            print(f"✅ Endpoint created: {endpoint_id}")
            print(f"🌐 Endpoint URL: {endpoint.get('url', 'N/A')}")
            
            self.endpoint_id = endpoint_id
            
            # Save to environment
            with open(os.path.expanduser("~/.zshrc"), "a") as f:
                f.write(f'\nexport RUNPOD_ENDPOINT_ID="{endpoint_id}"\n')
            
            return endpoint_id
            
        except Exception as e:
            print(f"❌ Endpoint creation failed: {e}")
            raise
    
    def create_config_file(self):
        """Create configuration file"""
        print("\n📝 Creating configuration...")
        
        config = {
            "runpod": {
                "api_key": self.api_key,
                "endpoint_id": self.endpoint_id,
                "template_id": self.template_id,
                "docker_image": self.docker_image
            },
            "setup_date": time.strftime("%Y-%m-%d %H:%M:%S"),
            "status": "configured"
        }
        
        config_path = Path("config/runpod_config.json")
        config_path.parent.mkdir(exist_ok=True)
        
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)
        
        print(f"✅ Configuration saved to {config_path}")
        self.config = config
    
    def test_setup(self) -> bool:
        """Test the complete setup"""
        print("\n🧪 Testing Setup...")
        
        try:
            # Import your wrapper
            sys.path.append(str(Path.cwd()))
            from services.inference.wan_remote_wrapper import create_wan_remote_service
            
            print("✅ WAN remote wrapper imported")
            
            # Create service
            wan_service = create_wan_remote_service("runpod")
            print("✅ RunPod service created")
            
            print("\n🎬 Ready to generate videos!")
            print("Use this code in your application:")
            print("""
from services.inference.wan_remote_wrapper import create_wan_remote_service

wan_service = create_wan_remote_service("runpod")
result = await wan_service.generate_video_remote(request)
            """)
            
            return True
            
        except Exception as e:
            print(f"❌ Test failed: {e}")
            return False
    
    def show_cost_estimate(self):
        """Show cost estimates"""
        print("\n💰 Cost Estimates:")
        print("─" * 50)
        print("GPU Type        | Cost/Hour | Est. Cost/Video")
        print("─" * 50)
        print("RTX 4090        | $0.44     | $0.22-0.44")
        print("RTX A6000       | $0.79     | $0.40-0.79") 
        print("A100 80GB       | $1.19     | $0.60-1.19")
        print("─" * 50)
        print("* Estimates based on 3-8 minute generation time")
        print("* Idle timeout set to 2 minutes to minimize costs")
        print("* Monitor usage at https://runpod.io/console")
    
    def print_next_steps(self):
        """Print next steps"""
        print(f"""
🎉 RunPod Setup Complete!

Next Steps:
───────────
1. Update your code to use RunPod:
   
   # Change this:
   from visionflow.services.generation.wan_video_service import WanVideoGenerationService
   wan_service = WanVideoGenerationService()
   
   # To this:
   from services.inference.wan_remote_wrapper import create_wan_remote_service
   wan_service = create_wan_remote_service("runpod")

2. Test with a generation:
   python test_runpod_generation.py

3. Monitor costs:
   https://runpod.io/console

4. Scale endpoint:
   runpod endpoints update {self.endpoint_id} --max-workers 5

Environment Variables Set:
─────────────────────────
RUNPOD_API_KEY={self.api_key[:8]}...
RUNPOD_ENDPOINT_ID={self.endpoint_id}

Configuration saved to: config/runpod_config.json
        """)
    
    async def run_setup(self):
        """Run the complete setup process"""
        try:
            self.print_banner()
            
            if not self.check_prerequisites():
                return False
            
            self.get_api_key()
            self.build_docker_image()
            self.create_template()
            self.create_endpoint()
            self.create_config_file()
            
            if self.test_setup():
                self.show_cost_estimate()
                self.print_next_steps()
                return True
            else:
                print("❌ Setup completed but tests failed")
                return False
                
        except KeyboardInterrupt:
            print("\n⚠️  Setup interrupted by user")
            return False
        except Exception as e:
            print(f"\n❌ Setup failed: {e}")
            logger.exception("Setup error")
            return False

def main():
    """Main setup function"""
    setup = RunPodSetup()
    success = asyncio.run(setup.run_setup())
    
    if success:
        print("\n🎉 Setup completed successfully!")
        exit(0)
    else:
        print("\n❌ Setup failed. Check the logs above.")
        exit(1)

if __name__ == "__main__":
    main()
