#!/usr/bin/env python3
"""
Enhanced VisionFlow Dashboard Import Tool
Imports a comprehensive dashboard with integrated layout and sophisticated visualizations
"""

import json
import requests
import sys
from pathlib import Path

def import_enhanced_dashboard():
    """Import the enhanced VisionFlow dashboard into Grafana"""
    
    print("🎬 VisionFlow Enhanced Dashboard Import Tool")
    print("=" * 50)
    
    # Detect if running inside container or from host
    try:
        # Try to connect to localhost first (container)
        response = requests.get("http://localhost:3000/api/health", timeout=5)
        grafana_url = "http://localhost:3000"
        print("🔄 Running from container, using localhost:3000")
    except:
        # Fallback to host port
        grafana_url = "http://localhost:30300"
        print("🔄 Running from host, using localhost:30300")
    
    # Test connection to Grafana
    print("🔍 Testing connection to Grafana...")
    try:
        response = requests.get(f"{grafana_url}/api/health", timeout=10)
        if response.status_code == 200:
            print("✅ Grafana is accessible")
        else:
            print(f"❌ Grafana returned status {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Failed to connect to Grafana: {e}")
        return False
    
    # Load dashboard JSON
    dashboard_path = Path(__file__).parent / "enhanced-dashboard.json"
    if not dashboard_path.exists():
        print(f"❌ Dashboard file not found: {dashboard_path}")
        return False
    
    try:
        with open(dashboard_path, 'r') as f:
            dashboard_data = json.load(f)
        print("📊 Dashboard JSON loaded successfully")
    except Exception as e:
        print(f"❌ Failed to load dashboard JSON: {e}")
        return False
    
    # Import dashboard
    print("📊 Importing Enhanced VisionFlow dashboard...")
    try:
        response = requests.post(
            f"{grafana_url}/api/dashboards/db",
            json={"dashboard": dashboard_data["dashboard"], "overwrite": True},
            headers={"Content-Type": "application/json"},
            auth=("admin", "admin123"),
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            dashboard_id = result["id"]
            dashboard_url = result["url"]
            
            print("✅ Enhanced dashboard imported successfully!")
            print(f"   Dashboard ID: {dashboard_id}")
            print(f"   URL: {grafana_url}{dashboard_url}")
            
            print("\n🎉 Enhanced dashboard import completed!")
            print("🌐 Open Grafana:", grafana_url)
            print("   Username: admin")
            print("   Password: admin123")
            print("   Look for: VisionFlow Enhanced Video Generation Dashboard")
            
            return True
        else:
            print(f"❌ Failed to import dashboard: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Dashboard import failed: {e}")
        return False

if __name__ == "__main__":
    success = import_enhanced_dashboard()
    sys.exit(0 if success else 1)
