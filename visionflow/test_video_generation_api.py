#!/usr/bin/env python3
"""
Test script for the new video generation API endpoints.
This script tests the FastAPI application without starting the full server.
"""

import asyncio
import sys
from pathlib import Path

# Add the visionflow package to the path
sys.path.insert(0, str(Path(__file__).parent))

from visionflow.api.main import app
from visionflow.shared.models import VideoGenerationRequest, VideoQuality
from fastapi.testclient import TestClient

def test_api_endpoints():
    """Test the API endpoints using FastAPI's TestClient."""
    client = TestClient(app)
    
    print("🧪 Testing VisionFlow API endpoints...")
    
    # Test root endpoint
    print("\n1. Testing root endpoint...")
    response = client.get("/")
    print(f"   Status: {response.status_code}")
    print(f"   Response: {response.json()}")
    
    # Test health endpoint
    print("\n2. Testing health endpoint...")
    response = client.get("/health")
    print(f"   Status: {response.status_code}")
    print(f"   Response: {response.json()}")
    
    # Test video generation endpoint
    print("\n3. Testing video generation endpoint...")
    test_request = VideoGenerationRequest(
        prompt="A beautiful sunset over the ocean",
        duration=5,
        quality=VideoQuality.MEDIUM,
        fps=24,
        resolution="512x512",
        seed=42,
        guidance_scale=7.5,
        num_inference_steps=20
    )
    
    # Use model_dump() instead of deprecated dict()
    response = client.post("/generate/video", json=test_request.model_dump())
    print(f"   Status: {response.status_code}")
    print(f"   Response: {response.json()}")
    
    # Test generation status endpoint
    print("\n4. Testing generation status endpoint...")
    response = client.get("/generate/video/status/test-123")
    print(f"   Status: {response.status_code}")
    print(f"   Response: {response.json()}")
    
    # Test generation history endpoint
    print("\n5. Testing generation history endpoint...")
    response = client.get("/generate/video/history?limit=5&offset=0")
    print(f"   Status: {response.status_code}")
    print(f"   Response: {response.json()}")
    
    print("\n✅ API endpoint tests completed!")

if __name__ == "__main__":
    test_api_endpoints()
