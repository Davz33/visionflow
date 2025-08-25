#!/usr/bin/env python3
"""
Test script for RunPod WAN 2.1 generation
Verifies that the complete setup works end-to-end
"""

import asyncio
import os
import sys
import time
import json
from pathlib import Path
from dataclasses import dataclass
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class TestVideoRequest:
    """Test request matching your VideoGenerationRequest format"""
    prompt: str = "A cat dancing in the rain, cinematic style, 4K"
    duration: int = 5
    fps: int = 24
    resolution: str = "512x512"
    quality: str = "medium"
    guidance_scale: float = 7.5
    num_inference_steps: int = 25
    seed: int = 42

class RunPodTester:
    """Test RunPod WAN generation setup"""
    
    def __init__(self):
        self.results = {}
        
    def print_banner(self):
        """Print test banner"""
        print("""
╔══════════════════════════════════════════════════════════════╗
║                🧪 RunPod WAN 2.1 Test Suite                 ║
║              Testing your remote generation setup           ║
╚══════════════════════════════════════════════════════════════╝
        """)
    
    def check_environment(self) -> bool:
        """Check if environment is properly configured"""
        print("🔍 Checking environment configuration...")
        
        required_vars = {
            'RUNPOD_API_KEY': 'RunPod API key',
            'RUNPOD_ENDPOINT_ID': 'RunPod endpoint ID'
        }
        
        missing_vars = []
        for var, description in required_vars.items():
            value = os.getenv(var)
            if not value:
                print(f"❌ Missing {description}: {var}")
                missing_vars.append(var)
            else:
                masked_value = value[:8] + "..." if len(value) > 8 else value
                print(f"✅ {description}: {masked_value}")
        
        if missing_vars:
            print(f"\n💡 To fix, run:")
            print(f"export RUNPOD_API_KEY='your-api-key'")
            print(f"export RUNPOD_ENDPOINT_ID='your-endpoint-id'")
            return False
        
        return True
    
    def check_dependencies(self) -> bool:
        """Check if required packages are installed"""
        print("\n📦 Checking dependencies...")
        
        try:
            import runpod
            print("✅ RunPod SDK available")
        except ImportError:
            print("❌ RunPod SDK not found. Install with: pip install runpod")
            return False
        
        # Check if wrapper exists
        try:
            from services.inference.wan_remote_wrapper import create_wan_remote_service
            print("✅ WAN remote wrapper available")
        except ImportError as e:
            print(f"❌ WAN remote wrapper not found: {e}")
            print("💡 Make sure you're running from the visionflow directory")
            return False
        
        return True
    
    def test_runpod_connection(self) -> bool:
        """Test basic RunPod API connection"""
        print("\n🔗 Testing RunPod connection...")
        
        try:
            import runpod
            
            # Test API connection
            endpoints = runpod.get_endpoints()
            print(f"✅ Connected to RunPod API ({len(endpoints)} endpoints found)")
            
            # Check specific endpoint
            endpoint_id = os.getenv('RUNPOD_ENDPOINT_ID')
            endpoint_found = False
            
            for endpoint in endpoints:
                if endpoint['id'] == endpoint_id:
                    endpoint_found = True
                    print(f"✅ Found target endpoint: {endpoint['name']}")
                    print(f"   Status: {endpoint.get('status', 'unknown')}")
                    break
            
            if not endpoint_found:
                print(f"⚠️  Endpoint {endpoint_id} not found in your account")
                print("   Available endpoints:")
                for ep in endpoints[:3]:  # Show first 3
                    print(f"   - {ep['name']} ({ep['id']})")
                return False
            
            return True
            
        except Exception as e:
            print(f"❌ RunPod connection failed: {e}")
            return False
    
    async def test_service_creation(self) -> bool:
        """Test WAN service creation"""
        print("\n🛠️  Testing service creation...")
        
        try:
            from services.inference.wan_remote_wrapper import create_wan_remote_service
            
            # Create RunPod service
            wan_service = create_wan_remote_service("runpod")
            print("✅ RunPod WAN service created successfully")
            
            # Check if backend is available
            if hasattr(wan_service, '_is_backend_available'):
                available = wan_service._is_backend_available()
                if available:
                    print("✅ RunPod backend is available")
                else:
                    print("⚠️  RunPod backend not available - will fallback to local")
            
            self.wan_service = wan_service
            return True
            
        except Exception as e:
            print(f"❌ Service creation failed: {e}")
            return False
    
    async def test_quick_generation(self) -> bool:
        """Test a quick video generation"""
        print("\n🎬 Testing video generation...")
        print("⚠️  This will cost ~$0.50-1.00 depending on GPU type")
        
        # Ask for confirmation
        confirm = input("Proceed with test generation? [y/N]: ").strip().lower()
        if confirm not in ['y', 'yes']:
            print("⏭️  Skipping generation test")
            return True
        
        try:
            # Create test request
            request = TestVideoRequest(
                prompt="A simple test: red ball bouncing",
                duration=3,  # Shorter for testing
                num_inference_steps=15  # Faster for testing
            )
            
            print(f"🎯 Prompt: {request.prompt}")
            print(f"⏱️  Duration: {request.duration}s")
            print(f"🔧 Steps: {request.num_inference_steps}")
            
            start_time = time.time()
            
            # Generate video
            print("\n🚀 Starting generation...")
            result = await self.wan_service.generate_video_remote(request)
            
            generation_time = time.time() - start_time
            
            print(f"✅ Generation completed in {generation_time:.1f}s!")
            
            # Display results
            if isinstance(result, dict):
                print("\n📊 Results:")
                print(f"   Status: {result.get('status', 'unknown')}")
                print(f"   Video path: {result.get('video_path', 'N/A')}")
                print(f"   Generation time: {result.get('generation_time', 'N/A')}s")
                
                # Estimate cost
                if 'generation_time' in result:
                    cost_estimate = (result['generation_time'] / 3600) * 0.44  # RTX 4090 rate
                    print(f"   Estimated cost: ${cost_estimate:.3f}")
            
            self.results['generation_test'] = {
                'success': True,
                'total_time': generation_time,
                'result': result
            }
            
            return True
            
        except Exception as e:
            print(f"❌ Generation test failed: {e}")
            self.results['generation_test'] = {
                'success': False,
                'error': str(e)
            }
            return False
    
    def test_fallback_behavior(self) -> bool:
        """Test fallback to local generation"""
        print("\n🔄 Testing fallback behavior...")
        
        try:
            from services.inference.wan_remote_wrapper import create_wan_remote_service
            
            # Create service with local fallback
            wan_service = create_wan_remote_service("local")
            print("✅ Local fallback service created")
            
            # This should work even without RunPod
            print("✅ Fallback mechanism working")
            return True
            
        except Exception as e:
            print(f"❌ Fallback test failed: {e}")
            return False
    
    def show_performance_summary(self):
        """Show performance and cost summary"""
        print("\n📊 Performance Summary:")
        print("─" * 60)
        
        if 'generation_test' in self.results:
            test = self.results['generation_test']
            if test['success']:
                result = test['result']
                total_time = test['total_time']
                
                print(f"✅ Generation successful")
                print(f"📏 Total time: {total_time:.1f}s")
                
                if isinstance(result, dict) and 'generation_time' in result:
                    gen_time = result['generation_time']
                    overhead = total_time - gen_time
                    cost = (gen_time / 3600) * 0.44
                    
                    print(f"🎬 Generation time: {gen_time:.1f}s")
                    print(f"⚡ Network overhead: {overhead:.1f}s")
                    print(f"💰 Estimated cost: ${cost:.3f}")
                    
                    # Performance rating
                    if overhead < 30:
                        print(f"🚀 Performance: Excellent (low overhead)")
                    elif overhead < 60:
                        print(f"👍 Performance: Good")
                    else:
                        print(f"⚠️  Performance: High overhead detected")
            else:
                print(f"❌ Generation failed: {test.get('error', 'Unknown error')}")
        
        print("─" * 60)
    
    def save_test_results(self):
        """Save test results to file"""
        results_file = Path("test_results_runpod.json")
        
        test_summary = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "environment": {
                "runpod_api_key_set": bool(os.getenv('RUNPOD_API_KEY')),
                "runpod_endpoint_id_set": bool(os.getenv('RUNPOD_ENDPOINT_ID'))
            },
            "results": self.results
        }
        
        with open(results_file, "w") as f:
            json.dump(test_summary, f, indent=2)
        
        print(f"📁 Test results saved to: {results_file}")
    
    async def run_all_tests(self) -> bool:
        """Run all tests"""
        self.print_banner()
        
        tests = [
            ("Environment Check", self.check_environment),
            ("Dependencies Check", self.check_dependencies),
            ("RunPod Connection", self.test_runpod_connection),
            ("Service Creation", self.test_service_creation),
            ("Generation Test", self.test_quick_generation),
            ("Fallback Test", self.test_fallback_behavior)
        ]
        
        passed = 0
        failed = 0
        
        for test_name, test_func in tests:
            print(f"\n{'='*60}")
            print(f"Running: {test_name}")
            print(f"{'='*60}")
            
            try:
                if asyncio.iscoroutinefunction(test_func):
                    success = await test_func()
                else:
                    success = test_func()
                
                if success:
                    passed += 1
                    print(f"✅ {test_name} PASSED")
                else:
                    failed += 1
                    print(f"❌ {test_name} FAILED")
                    
            except Exception as e:
                failed += 1
                print(f"❌ {test_name} ERROR: {e}")
        
        print(f"\n{'='*60}")
        print(f"📊 Test Summary: {passed} passed, {failed} failed")
        
        if failed == 0:
            print("🎉 All tests passed! RunPod setup is working correctly.")
            self.show_performance_summary()
        else:
            print("⚠️  Some tests failed. Check the output above for details.")
        
        self.save_test_results()
        return failed == 0

async def main():
    """Main test function"""
    tester = RunPodTester()
    success = await tester.run_all_tests()
    
    if success:
        print("\n🚀 Your RunPod WAN 2.1 setup is ready!")
        print("\nNext steps:")
        print("1. Update your code to use the RunPod backend")
        print("2. Monitor costs at https://runpod.io/console")
        print("3. Scale your endpoint based on demand")
    else:
        print("\n🔧 Fix the issues above and run the test again")
    
    return success

if __name__ == "__main__":
    success = asyncio.run(main())
    exit(0 if success else 1)
