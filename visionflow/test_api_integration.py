#!/usr/bin/env python3
"""
Integration test script for the VisionFlow API.
This script starts the FastAPI application and tests all endpoints with proper service initialization.
"""

import asyncio
import sys
import uvicorn
from pathlib import Path
import requests
import time
import json

# Add the visionflow package to the path
sys.path.insert(0, str(Path(__file__).parent))

def test_api_integration():
    """Test the API with a running server."""
    print("🚀 Starting VisionFlow API server for integration testing...")
    
    # Start the server in a separate process
    import subprocess
    import signal
    import os
    
    # Start the server
    server_process = subprocess.Popen([
        sys.executable, "-m", "uvicorn", 
        "visionflow.api.main:app", 
        "--host", "0.0.0.0", 
        "--port", "8001"  # Use different port to avoid conflicts
    ], cwd=Path(__file__).parent)
    
    try:
        # Wait for server to start
        print("⏳ Waiting for server to start...")
        time.sleep(5)
        
        base_url = "http://localhost:8001"
        
        print("🧪 Testing API endpoints...")
        
        # Test root endpoint
        print("\n1. Testing root endpoint...")
        response = requests.get(f"{base_url}/")
        print(f"   Status: {response.status_code}")
        print(f"   Response: {json.dumps(response.json(), indent=2)}")
        
        # Test health endpoint
        print("\n2. Testing health endpoint...")
        response = requests.get(f"{base_url}/health")
        print(f"   Status: {response.status_code}")
        print(f"   Response: {json.dumps(response.json(), indent=2)}")
        
        # Test video generation endpoint
        print("\n3. Testing video generation endpoint...")
        test_request = {
            "prompt": "A beautiful sunset over the ocean",
            "duration": 5,
            "quality": "medium",
            "fps": 24,
            "resolution": "512x512",
            "seed": 42,
            "guidance_scale": 7.5,
            "num_inference_steps": 20
        }
        
        response = requests.post(f"{base_url}/generate/video", json=test_request)
        print(f"   Status: {response.status_code}")
        print(f"   Response: {json.dumps(response.json(), indent=2)}")
        
        # Test generation status endpoint
        print("\n4. Testing generation status endpoint...")
        response = requests.get(f"{base_url}/generate/video/status/test-123")
        print(f"   Status: {response.status_code}")
        print(f"   Response: {json.dumps(response.json(), indent=2)}")
        
        # Test generation history endpoint
        print("\n5. Testing generation history endpoint...")
        response = requests.get(f"{base_url}/generate/video/history?limit=5&offset=0")
        print(f"   Status: {response.status_code}")
        print(f"   Response: {json.dumps(response.json(), indent=2)}")
        
        print("\n✅ Integration tests completed!")
        
    finally:
        # Clean up - stop the server
        print("\n🛑 Stopping server...")
        server_process.terminate()
        server_process.wait()
        print("✅ Server stopped")

if __name__ == "__main__":
    test_api_integration()
