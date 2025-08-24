#!/usr/bin/env python3
"""Simple API test script for VisionFlow."""

import requests
import time
import json

def test_health():
    """Test health endpoint."""
    response = requests.get("http://localhost:8000/health")
    print(f"Health check: {response.status_code}")
    if response.status_code == 200:
        health_data = response.json()
        print(f"Status: {health_data['status']}")
        print(f"Database: {'✓' if health_data['database_connected'] else '✗'}")
        print(f"Redis: {'✓' if health_data['redis_connected'] else '✗'}")
        print(f"Storage: {'✓' if health_data['storage_connected'] else '✗'}")

def test_generate_video():
    """Test video generation."""
    payload = {
        "prompt": "A serene sunset over mountains with gentle clouds",
        "duration": 5,
        "quality": "medium",
        "fps": 24,
        "resolution": "512x512"
    }
    
    print("\nTesting video generation...")
    response = requests.post("http://localhost:8000/api/v1/generate", json=payload)
    
    if response.status_code == 200:
        job_data = response.json()
        job_id = job_data["job_id"]
        print(f"Job created: {job_id}")
        
        # Poll for completion
        for i in range(30):  # Wait up to 5 minutes
            status_response = requests.get(f"http://localhost:8000/api/v1/jobs/{job_id}/status")
            if status_response.status_code == 200:
                status_data = status_response.json()
                print(f"Status: {status_data['status']} ({status_data['progress']:.1%})")
                
                if status_data['status'] == 'completed':
                    print("✓ Video generation completed!")
                    break
                elif status_data['status'] == 'failed':
                    print(f"✗ Video generation failed: {status_data.get('error_message')}")
                    break
            
            time.sleep(10)
    else:
        print(f"Failed to create job: {response.status_code} - {response.text}")

if __name__ == "__main__":
    print("VisionFlow API Test")
    print("==================")
    
    try:
        test_health()
        test_generate_video()
    except requests.exceptions.ConnectionError:
        print("❌ Could not connect to API. Make sure the service is running.")
    except Exception as e:
        print(f"❌ Test failed: {e}")
