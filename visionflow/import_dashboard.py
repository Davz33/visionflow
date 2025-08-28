#!/usr/bin/env python3
"""
Script to manually import the VisionFlow dashboard into Grafana
"""

import requests
import json
import time
import os

def import_dashboard():
    """Import the VisionFlow dashboard into Grafana"""
    
    # Grafana connection details - detect if running inside container
    if os.path.exists('/.dockerenv'):
        # Running inside container
        grafana_url = "http://localhost:3000"
        print("🔄 Running inside container, using localhost:3000")
    else:
        # Running from host
        grafana_url = "http://localhost:30300"
        print("🔄 Running from host, using localhost:30300")
    
    username = "admin"
    password = "admin123"
    
    # Dashboard JSON content
    dashboard_json = {
        "dashboard": {
            "id": None,
            "title": "VisionFlow Video Generation Dashboard",
            "tags": ["visionflow", "video-generation", "monitoring"],
            "timezone": "browser",
            "panels": [
                {
                    "id": 1,
                    "title": "Active Video Generation Jobs",
                    "type": "stat",
                    "targets": [
                        {
                            "expr": "visionflow_video_generation_jobs_active",
                            "refId": "A"
                        }
                    ],
                    "fieldConfig": {
                        "defaults": {
                            "color": {
                                "mode": "thresholds"
                            },
                            "thresholds": {
                                "steps": [
                                    {"color": "green", "value": None},
                                    {"color": "yellow", "value": 5},
                                    {"color": "red", "value": 10}
                                ]
                            }
                        }
                    },
                    "gridPos": {"h": 8, "w": 6, "x": 0, "y": 0}
                },
                {
                    "id": 2,
                    "title": "Job Queue Length",
                    "type": "stat",
                    "targets": [
                        {
                            "expr": "visionflow_video_generation_queue_length",
                            "refId": "A"
                        }
                    ],
                    "fieldConfig": {
                        "defaults": {
                            "color": {
                                "mode": "thresholds"
                            },
                            "thresholds": {
                                "steps": [
                                    {"color": "green", "value": None},
                                    {"color": "yellow", "value": 5},
                                    {"color": "red", "value": 15}
                                ]
                            }
                        }
                    },
                    "gridPos": {"h": 8, "w": 6, "x": 6, "y": 0}
                },
                {
                    "id": 3,
                    "title": "Success Rate",
                    "type": "stat",
                    "targets": [
                        {
                            "expr": "visionflow_video_generation_success_rate * 100",
                            "refId": "A"
                        }
                    ],
                    "fieldConfig": {
                        "defaults": {
                            "unit": "percent",
                            "color": {
                                "mode": "thresholds"
                            },
                            "thresholds": {
                                "steps": [
                                    {"color": "red", "value": None},
                                    {"color": "yellow", "value": 80},
                                    {"color": "green", "value": 95}
                                ]
                            }
                        }
                    },
                    "gridPos": {"h": 8, "w": 6, "x": 0, "y": 8}
                },
                {
                    "id": 4,
                    "title": "Average Wait Time",
                    "type": "stat",
                    "targets": [
                        {
                            "expr": "visionflow_video_generation_average_wait_time",
                            "refId": "A"
                        }
                    ],
                    "fieldConfig": {
                        "defaults": {
                            "unit": "s",
                            "color": {
                                "mode": "thresholds"
                            },
                            "thresholds": {
                                "steps": [
                                    {"color": "green", "value": None},
                                    {"color": "yellow", "value": 30},
                                    {"color": "red", "value": 60}
                                ]
                            }
                        }
                    },
                    "gridPos": {"h": 8, "w": 6, "x": 6, "y": 8}
                }
            ],
            "time": {
                "from": "now-1h",
                "to": "now"
            },
            "refresh": "5s"
        },
        "folderId": 0,
        "overwrite": True
    }

    try:
        # Test connection to Grafana
        print("🔍 Testing connection to Grafana...")
        response = requests.get(f"{grafana_url}/api/health", timeout=10)
        if response.status_code == 200:
            print("✅ Grafana is accessible")
        else:
            print(f"❌ Grafana health check failed: {response.status_code}")
            return False

        # Import the dashboard
        print("📊 Importing VisionFlow dashboard...")
        response = requests.post(
            f"{grafana_url}/api/dashboards/db",
            json=dashboard_json,
            auth=(username, password),
            headers={"Content-Type": "application/json"},
            timeout=30
        )

        if response.status_code == 200:
            result = response.json()
            dashboard_id = result.get("id")
            dashboard_url = result.get("url")
            print(f"✅ Dashboard imported successfully!")
            print(f"   Dashboard ID: {dashboard_id}")
            print(f"   URL: {grafana_url}{dashboard_url}")
            return True
        else:
            print(f"❌ Dashboard import failed: {response.status_code}")
            print(f"   Response: {response.text}")
            return False

    except requests.exceptions.RequestException as e:
        print(f"❌ Connection error: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

if __name__ == "__main__":
    print("🎬 VisionFlow Dashboard Import Tool")
    print("========================================")
    
    # Try to import the dashboard
    success = import_dashboard()
    
    if success:
        print("\n🎉 Dashboard import completed!")
        print("🌐 Open Grafana: http://localhost:30300")
        print("   Username: admin")
        print("   Password: admin123")
        print("   Look for: VisionFlow Video Generation Dashboard")
    else:
        print("\n❌ Dashboard import failed!")
        print("   Check Grafana logs for more details")
