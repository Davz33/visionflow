#!/usr/bin/env python3
"""
Test Grafana Login Credentials
"""

import requests
import json

def test_grafana_login():
    """Test Grafana login with different credential combinations"""
    
    grafana_url = "http://localhost:30300"
    
    # Test different credential combinations
    test_credentials = [
        ("admin", "admin"),
        ("admin", "admin123"),
        ("admin", "password"),
        ("admin", ""),
    ]
    
    print("🔐 Testing Grafana Login Credentials")
    print("=" * 40)
    
    for username, password in test_credentials:
        print(f"\n🧪 Testing: {username}/{password}")
        
        try:
            # Try to access Grafana
            response = requests.get(f"{grafana_url}/api/health", timeout=5)
            
            if response.status_code == 200:
                print(f"✅ Health check passed: {response.status_code}")
                
                # Try to login
                login_data = {
                    "user": username,
                    "password": password
                }
                
                login_response = requests.post(
                    f"{grafana_url}/login",
                    data=login_data,
                    timeout=5
                )
                
                print(f"📝 Login response: {login_response.status_code}")
                
                if login_response.status_code == 200:
                    print("🎉 LOGIN SUCCESSFUL!")
                    print(f"Credentials: {username}/{password}")
                    return username, password
                else:
                    print(f"❌ Login failed: {login_response.status_code}")
                    print(f"Response: {login_response.text[:200]}")
                    
            else:
                print(f"❌ Health check failed: {response.status_code}")
                
        except requests.exceptions.RequestException as e:
            print(f"❌ Connection error: {e}")
        except Exception as e:
            print(f"❌ Unexpected error: {e}")
    
    print("\n🔍 No working credentials found. Let's check the Grafana configuration...")
    return None, None

def check_grafana_config():
    """Check Grafana configuration from Kubernetes"""
    import subprocess
    import json
    
    print("\n🔧 Checking Grafana Configuration")
    print("=" * 40)
    
    try:
        # Get Grafana pod name
        result = subprocess.run(
            ["kubectl", "get", "pods", "-n", "visionflow-local", "-l", "app=grafana", "-o", "jsonpath={.items[0].metadata.name}"],
            capture_output=True, text=True, check=True
        )
        pod_name = result.stdout.strip()
        
        if pod_name:
            print(f"📦 Grafana pod: {pod_name}")
            
            # Get environment variables
            result = subprocess.run(
                ["kubectl", "exec", "-n", "visionflow-local", pod_name, "--", "env", "|", "grep", "GF_SECURITY"],
                shell=True, capture_output=True, text=True
            )
            
            if result.stdout:
                print("🔑 Environment variables:")
                for line in result.stdout.strip().split('\n'):
                    if line:
                        print(f"   {line}")
            else:
                print("❌ No GF_SECURITY environment variables found")
                
            # Check Grafana logs
            print("\n📋 Recent Grafana logs:")
            result = subprocess.run(
                ["kubectl", "logs", "-n", "visionflow-local", pod_name, "--tail=10"],
                capture_output=True, text=True, check=True
            )
            
            for line in result.stdout.strip().split('\n'):
                if line and ("admin" in line.lower() or "password" in line.lower() or "login" in line.lower()):
                    print(f"   {line}")
                    
        else:
            print("❌ No Grafana pod found")
            
    except subprocess.CalledProcessError as e:
        print(f"❌ Error checking configuration: {e}")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")

def main():
    """Main function"""
    print("🚀 Grafana Credential Test")
    print("=" * 30)
    
    # Test login
    username, password = test_grafana_login()
    
    if not username:
        # Check configuration
        check_grafana_config()
        
        print("\n💡 Troubleshooting Tips:")
        print("1. Check if Grafana pod is running: kubectl get pods -n visionflow-local | grep grafana")
        print("2. Check Grafana logs: kubectl logs -n visionflow-local deployment/grafana")
        print("3. Verify environment variables: kubectl describe deployment grafana -n visionflow-local")
        print("4. Try accessing Grafana directly: http://localhost:30300")
        print("5. Check if port forwarding is working: ./scripts/dev-env-manager.sh status")

if __name__ == "__main__":
    main()
