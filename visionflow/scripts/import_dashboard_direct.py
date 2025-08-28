#!/usr/bin/env python3
"""
Simple script to directly import the evaluation dashboard to Grafana
Bypasses complex K8s job setup for quick testing
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

def create_admin_user():
    """Create admin user if it doesn't exist"""
    print("Attempting to create admin user...")
    
    # Try to create admin user
    admin_data = {
        "login": "admin",
        "email": "admin@localhost",
        "password": "admin",
        "name": "Admin"
    }
    
    try:
        response = requests.post(
            f"{GRAFANA_URL}/api/admin/users",
            json=admin_data,
            auth=(GRAFANA_USER, GRAFANA_PASSWORD),
            timeout=10
        )
        
        if response.status_code == 200:
            print("✅ Admin user created successfully")
        elif response.status_code == 412:
            print("ℹ️ Admin user already exists")
        else:
            print(f"⚠️ Unexpected response creating admin user: {response.status_code}")
            print(f"Response: {response.text}")
            
    except requests.exceptions.RequestException as e:
        print(f"⚠️ Error creating admin user: {e}")

def import_dashboard():
    """Import the evaluation dashboard"""
    print("Importing evaluation dashboard...")
    
    # Read the dashboard JSON
    try:
        with open("evaluation_dashboard.json", "r") as f:
            dashboard_data = json.load(f)
    except FileNotFoundError:
        print("❌ evaluation_dashboard.json not found!")
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
            print("✅ Dashboard imported successfully!")
            print(f"Dashboard URL: {GRAFANA_URL}{result.get('url', '')}")
            return True
        else:
            print(f"❌ Dashboard import failed: {response.status_code}")
            print(f"Response: {response.text}")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Error importing dashboard: {e}")
        return False

def verify_dashboard():
    """Verify the dashboard was imported"""
    print("Verifying dashboard import...")
    
    try:
        response = requests.get(
            f"{GRAFANA_URL}/api/search",
            auth=(GRAFANA_USER, GRAFANA_PASSWORD),
            timeout=10
        )
        
        if response.status_code == 200:
            dashboards = response.json()
            for dashboard in dashboards:
                if "evaluation" in dashboard.get("title", "").lower():
                    print(f"✅ Found evaluation dashboard: {dashboard['title']}")
                    return True
            
            print("❌ Evaluation dashboard not found in search results")
            return False
        else:
            print(f"❌ Failed to search dashboards: {response.status_code}")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Error verifying dashboard: {e}")
        return False

def main():
    """Main execution flow"""
    print("🚀 Starting direct dashboard import...")
    
    # Wait for Grafana
    if not wait_for_grafana():
        return False
    
    # Try to create admin user
    create_admin_user()
    
    # Import dashboard
    if not import_dashboard():
        return False
    
    # Verify import
    if not verify_dashboard():
        return False
    
    print("🎉 Dashboard import completed successfully!")
    print(f"Access Grafana at: {GRAFANA_URL}")
    print("Login with: admin/admin")
    return True

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
