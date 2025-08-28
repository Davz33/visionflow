#!/usr/bin/env python3
"""
Script to import the enhanced evaluation dashboard to Grafana
Includes all existing features plus new individual video details
"""

import json
import requests
import time

# Grafana connection details
GRAFANA_URL = "http://localhost:3000"
GRAFANA_USER = "admin"
GRAFANA_PASSWORD = "admin"

def wait_for_grafana():
    """Wait for Grafana to be ready"""
    print("Waiting for Grafana to be ready...")
    max_attempts = 30
    for attempt in range(max_attempts):
        try:
            response = requests.get(f"{GRAFANA_URL}/api/health", timeout=5)
            if response.status_code == 200:
                print("✅ Grafana is ready!")
                return True
        except requests.exceptions.RequestException:
            pass
        
        print(f"Attempt {attempt + 1}/{max_attempts}: Grafana not ready yet...")
        time.sleep(2)
    
    print("❌ Grafana failed to become ready")
    return False

def import_enhanced_dashboard():
    """Import the enhanced evaluation dashboard"""
    print("Importing enhanced evaluation dashboard...")
    
    # Read the enhanced dashboard JSON
    try:
        with open("evaluation_dashboard_enhanced.json", "r") as f:
            dashboard_data = json.load(f)
    except FileNotFoundError:
        print("❌ evaluation_dashboard_enhanced.json not found!")
        return False
    
    # Prepare the import payload - dashboard_data already contains the nested structure
    import_payload = {
        **dashboard_data,
        "overwrite": True,
        "inputs": []
    }
    
    try:
        response = requests.post(
            f"{GRAFANA_URL}/api/dashboards/db",
            json=import_payload,
            auth=(GRAFANA_USER, GRAFANA_PASSWORD),
            timeout=10
        )
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Enhanced dashboard imported successfully!")
            print(f"Dashboard URL: {GRAFANA_URL}{result.get('url', '')}")
            return True
        else:
            print(f"❌ Dashboard import failed: {response.status_code}")
            print(f"Response: {response.text}")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Error importing dashboard: {e}")
        return False

def verify_enhanced_dashboard():
    """Verify the enhanced dashboard was imported"""
    print("Verifying enhanced dashboard import...")
    
    try:
        response = requests.get(
            f"{GRAFANA_URL}/api/search",
            auth=(GRAFANA_USER, GRAFANA_PASSWORD),
            timeout=10
        )
        
        if response.status_code == 200:
            dashboards = response.json()
            for dashboard in dashboards:
                if "enhanced" in dashboard.get("title", "").lower():
                    print(f"✅ Found enhanced dashboard: {dashboard['title']}")
                    return True
            
            print("❌ Enhanced dashboard not found in search results")
            return False
        else:
            print(f"❌ Failed to search dashboards: {response.status_code}")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Error verifying dashboard: {e}")
        return False

def main():
    """Main execution flow"""
    print("🚀 Starting enhanced dashboard import...")
    
    # Wait for Grafana
    if not wait_for_grafana():
        return False
    
    # Import enhanced dashboard
    if not import_enhanced_dashboard():
        return False
    
    # Verify import
    if not verify_enhanced_dashboard():
        return False
    
    print("🎉 Enhanced dashboard import completed successfully!")
    print(f"Access Grafana at: {GRAFANA_URL}")
    print("Login with: admin/admin")
    print("\n📊 New Features Added:")
    print("- Individual Video Details Table")
    print("- Video Metadata Summary")
    print("- Processing Time by Category")
    print("- Quality Score Distribution")
    print("- Quick Access to HTML Viewer")
    print("- Enhanced Templating Variables")
    return True

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
