#!/usr/bin/env python3
"""
Test script for VisionFlow Monitoring Infrastructure
Tests the current health services and monitoring setup
"""

import requests
from datetime import datetime

def test_api_health():
    """Test API Gateway health endpoint"""
    try:
        response = requests.get("http://localhost:30000/health", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ API Gateway: {data['status']} - {data['service']}")
            return True
        else:
            print(f"❌ API Gateway: HTTP {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ API Gateway: {e}")
        return False

def test_grafana_access():
    """Test Grafana accessibility"""
    try:
        response = requests.get("http://localhost:30300", timeout=5)
        if response.status_code == 200:
            print("✅ Grafana: Accessible (login page)")
            return True
        else:
            print(f"❌ Grafana: HTTP {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Grafana: {e}")
        return False

def test_prometheus_metrics():
    """Test Prometheus metrics collection"""
    try:
        # Test basic query
        response = requests.get("http://localhost:30090/api/v1/query?query=up", timeout=5)
        if response.status_code == 200:
            data = response.json()
            if data['status'] == 'success':
                targets = data['data']['result']
                print(f"✅ Prometheus: Collecting metrics from {len(targets)} targets")
                for target in targets:
                    job = target['metric']['job']
                    instance = target['metric']['instance']
                    value = target['value'][1]
                    status = "UP" if value == "1" else "DOWN"
                    print(f"   - {job} ({instance}): {status}")
                return True
            else:
                print(f"❌ Prometheus: Query failed - {data}")
                return False
        else:
            print(f"❌ Prometheus: HTTP {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Prometheus: {e}")
        return False

def test_prometheus_targets():
    """Test Prometheus targets status"""
    try:
        response = requests.get("http://localhost:30090/targets", timeout=5)
        if response.status_code == 200:
            print("✅ Prometheus: Targets page accessible")
            return True
        else:
            print(f"❌ Prometheus: Targets page HTTP {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Prometheus: Targets page {e}")
        return False

def test_service_endpoints():
    """Test all service endpoints"""
    services = {
        "API Gateway": "http://localhost:30000/health",
        "Generation Service": "http://localhost:30002/health",
        "Orchestration Service": "http://localhost:30001/health",
        "Prometheus": "http://localhost:30090/api/v1/query?query=up",
        "Grafana": "http://localhost:30300"
    }
    
    print("\n🔍 Testing Service Endpoints:")
    print("=" * 50)
    
    results = {}
    for name, url in services.items():
        try:
            if "health" in url:
                response = requests.get(url, timeout=5)
                if response.status_code == 200:
                    print(f"✅ {name}: Healthy")
                    results[name] = True
                else:
                    print(f"❌ {name}: HTTP {response.status_code}")
                    results[name] = False
            elif "prometheus" in url:
                response = requests.get(url, timeout=5)
                if response.status_code == 200:
                    print(f"✅ {name}: Responding")
                    results[name] = True
                else:
                    print(f"❌ {name}: HTTP {response.status_code}")
                    results[name] = False
            elif "grafana" in url:
                response = requests.get(url, timeout=5)
                if response.status_code == 200:
                    print(f"✅ {name}: Accessible")
                    results[name] = True
                else:
                    print(f"❌ {name}: HTTP {response.status_code}")
                    results[name] = False
        except Exception as e:
            print(f"❌ {name}: {e}")
            results[name] = False
    
    return results

def show_monitoring_access():
    """Show how to access monitoring dashboards"""
    print("\n🌐 Monitoring Dashboard Access:")
    print("=" * 50)
    print("📊 Grafana Dashboard:")
    print("   URL: http://localhost:30300")
    print("   Username: admin")
    print("   Password: admin123")
    print("   Dashboard: VisionFlow Video Generation Dashboard")
    print()
    print("📈 Prometheus Metrics:")
    print("   URL: http://localhost:30090")
    print("   Query Examples:")
    print("   - Basic: http://localhost:30090/api/v1/query?query=up")
    print("   - Targets: http://localhost:30090/targets")
    print("   - Metrics: http://localhost:30090/metrics")
    print()
    print("📱 Web Monitor:")
    print("   File: visionflow/video_generation_monitor.html")
    print("   Open in browser for real-time updates")

def main():
    """Main test function"""
    print("🎬 VisionFlow Monitoring Infrastructure Test")
    print("=" * 50)
    print(f"⏰ Test Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Test basic functionality
    print("🔍 Testing Basic Infrastructure:")
    print("-" * 30)
    
    api_ok = test_api_health()
    grafana_ok = test_grafana_access()
    prometheus_ok = test_prometheus_metrics()
    
    print()
    
    # Test service endpoints
    service_results = test_service_endpoints()
    
    print()
    
    # Show monitoring access
    show_monitoring_access()
    
    print()
    print("📋 Test Summary:")
    print("=" * 50)
    
    # Count results
    total_services = len(service_results)
    healthy_services = sum(service_results.values())
    
    print(f"Services Tested: {total_services}")
    print(f"Healthy Services: {healthy_services}")
    print(f"Success Rate: {(healthy_services/total_services)*100:.1f}%")
    
    if api_ok and grafana_ok and prometheus_ok:
        print("\n🎉 Monitoring Infrastructure: READY!")
        print("You can now:")
        print("1. Open Grafana dashboard")
        print("2. View Prometheus metrics")
        print("3. Use the web monitor")
        print("4. Deploy full API for video generation testing")
    else:
        print("\n⚠️  Some issues detected. Check the logs above.")
    
    print("\n💡 Next Steps:")
    print("- Deploy full API with video generation endpoints")
    print("- Send video generation requests to test real-time monitoring")
    print("- Customize Grafana dashboards for your needs")

if __name__ == "__main__":
    main()
