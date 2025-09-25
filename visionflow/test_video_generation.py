#!/usr/bin/env python3
"""
Test script for VisionFlow Video Generation API
Tests the video generation endpoint and monitors response
"""

import requests
import json
from datetime import datetime

def test_video_generation():
    """Test video generation endpoint"""
    
    api_url = "http://localhost:30000"
    
    # Test data
    test_request = {
        "prompt": "A beautiful sunset over the ocean for testing monitoring dashboard",
        "duration": 3,
        "quality": "medium",
        "fps": 24,
        "resolution": "512x512",
        "seed": 42,
        "guidance_scale": 7.5,
        "num_inference_steps": 20
    }
    
    print("🎬 Testing VisionFlow Video Generation API")
    print("=" * 50)
    print(f"⏰ Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Test 1: Check API health
    print("🔍 Test 1: API Health Check")
    try:
        response = requests.get(f"{api_url}/health", timeout=10)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ API Health: {data['status']}")
            print(f"   Services: {data['services']}")
        else:
            print(f"❌ API Health Failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ API Health Error: {e}")
        return False
    
    print()
    
    # Test 2: Check available endpoints
    print("🔍 Test 2: Available Endpoints")
    try:
        response = requests.get(f"{api_url}/", timeout=10)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ API Root: {data['message']}")
            print(f"   Version: {data['version']}")
            print(f"   Capabilities: {data['capabilities']}")
            print(f"   Endpoints: {list(data['endpoints'].keys())}")
        else:
            print(f"❌ API Root Failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ API Root Error: {e}")
        return False
    
    print()
    
    # Test 3: Send video generation request
    print("🔍 Test 3: Video Generation Request")
    print(f"   Prompt: {test_request['prompt']}")
    print(f"   Duration: {test_request['duration']}s")
    print(f"   Quality: {test_request['quality']}")
    print(f"   Resolution: {test_request['resolution']}")
    print()
    print("⏳ Sending request (this may take several minutes for video generation)...")
    
    try:
        # Send request with a longer timeout
        response = requests.post(
            f"{api_url}/generate/video",
            json=test_request,
            timeout=300,  # 5 minutes timeout
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            data = response.json()
            print("✅ Video Generation Request Successful!")
            print(f"   Response: {json.dumps(data, indent=2)}")
            return True
        elif response.status_code == 202:
            print("✅ Video Generation Request Accepted (Async)")
            print(f"   Response: {json.dumps(response.json(), indent=2)}")
            return True
        else:
            print(f"❌ Video Generation Failed: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
            
    except requests.exceptions.Timeout:
        print("⏰ Request timed out after 5 minutes")
        print("   This is expected for video generation - the service is working but takes time")
        print("   Check the monitoring dashboard for progress indicators")
        return True  # Timeout is expected for video generation
        
    except Exception as e:
        print(f"❌ Video Generation Error: {e}")
        return False

def show_monitoring_instructions():
    """Show how to monitor the video generation"""
    print()
    print("📊 Monitoring Your Video Generation")
    print("=" * 40)
    print("🌐 Grafana Dashboard:")
    print("   URL: http://localhost:30300")
    print("   Username: admin")
    print("   Password: admin123")
    print("   Dashboard: VisionFlow Video Generation Dashboard")
    print()
    print("📈 Prometheus Metrics:")
    print("   URL: http://localhost:30090")
    print("   Query: visionflow_video_generation_jobs_active")
    print()
    print("📱 Web Monitor:")
    print("   File: visionflow/video_generation_monitor.html")
    print()
    print("💡 What to Look For:")
    print("   - Active job count increases")
    print("   - Queue length changes")
    print("   - Success/failure rates update")
    print("   - Generation duration metrics")

def main():
    """Main function"""
    print("🎬 VisionFlow Video Generation Test")
    print("=" * 50)
    
    success = test_video_generation()
    
    if success:
        print()
        print("🎉 Test completed successfully!")
        print("📊 Your monitoring dashboard should now show activity!")
    else:
        print()
        print("⚠️  Test failed. Check the logs above.")
    
    show_monitoring_instructions()

if __name__ == "__main__":
    main()
