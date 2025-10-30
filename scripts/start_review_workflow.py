#!/usr/bin/env python3
"""
Human Review Workflow Startup Script
Quick launcher for the complete human review workflow system
"""

import sys
import time
import subprocess
from pathlib import Path
from typing import List

def check_requirements():
    """Check if all required files and dependencies exist"""
    required_files = [
        "human_review_dashboard.html",
        "evaluation_viewer.html", 
        "review_workflow_service.py",
        "review_api_server.py",
        "evaluation_datasets/large_scale_samples/evaluation_results.json"
    ]
    
    missing_files = []
    for file_path in required_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)
    
    if missing_files:
        print("❌ Missing required files:")
        for file_path in missing_files:
            print(f"   - {file_path}")
        return False
    
    # Check Python dependencies
    try:
        import fastapi
        import uvicorn
        import sqlite3
    except ImportError as e:
        print(f"❌ Missing Python dependency: {e}")
        print("   Run: pip install fastapi uvicorn")
        return False
    
    return True

def initialize_workflow():
    """Initialize the review workflow database and data"""
    print("🔧 Initializing review workflow...")
    
    try:
        result = subprocess.run([
            sys.executable, "review_workflow_service.py", 
            "--init", 
            "--create-reviewer", "demo_reviewer", "Demo Reviewer", "demo@example.com"
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Review workflow initialized successfully")
            return True
        else:
            print(f"❌ Failed to initialize workflow: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ Error during initialization: {e}")
        return False

def start_api_server(host: str = "127.0.0.1", port: int = 8082):
    """Start the review workflow API server"""
    print(f"🚀 Starting Review Workflow API Server on {host}:{port}...")
    
    try:
        process = subprocess.Popen([
            sys.executable, "review_api_server.py",
            "--host", host,
            "--port", str(port),
            "--init"
        ])
        
        # Give the server a moment to start
        time.sleep(3)
        
        # Check if process is still running
        if process.poll() is None:
            print(f"✅ API server started successfully (PID: {process.pid})")
            return process
        else:
            print("❌ API server failed to start")
            return None
            
    except Exception as e:
        print(f"❌ Error starting API server: {e}")
        return None

def show_urls(host: str = "127.0.0.1", port: int = 8082):
    """Display access URLs for the user"""
    print("\n" + "="*70)
    print("🎉 HUMAN REVIEW WORKFLOW SYSTEM - READY!")
    print("="*70)
    
    print(f"\n📱 HUMAN REVIEW DASHBOARD:")
    print(f"   🌐 http://{host}:{port}/")
    print(f"   👥 Complete workflow management interface")
    print(f"   ✅ Review queue, annotations, consensus tracking")
    
    print(f"\n📊 EVALUATION VIEWER:")
    print(f"   🌐 http://{host}:{port}/viewer")
    print(f"   🎬 Browse all evaluation results by category")
    print(f"   📈 Visual score breakdowns and video playback")
    
    print(f"\n🔧 API ENDPOINTS:")
    print(f"   📋 API Documentation: http://{host}:{port}/docs")
    print(f"   📊 API Info: http://{host}:{port}/api")
    print(f"   💾 Health Check: http://{host}:{port}/api/health")
    
    print(f"\n🎯 WORKFLOW FEATURES:")
    print(f"   ✅ Priority-based review queue")
    print(f"   ✅ Human score overrides and annotations")
    print(f"   ✅ Review comments and tagging system")
    print(f"   ✅ Multi-reviewer consensus tracking")
    print(f"   ✅ AI vs Human agreement analytics")
    print(f"   ✅ Workflow performance metrics")
    
    print(f"\n🛠️  USAGE:")
    print(f"   1. Open the dashboard URL in your browser")
    print(f"   2. Videos requiring review are in the queue")
    print(f"   3. Click videos to review, rate, and annotate")
    print(f"   4. Submit approvals, rejections, or flags")
    print(f"   5. Track consensus and workflow statistics")
    
    print(f"\n⚠️  TO STOP THE SYSTEM:")
    print(f"   Press Ctrl+C in this terminal")
    
    print("\n" + "="*70)

def cleanup_processes(processes: List[subprocess.Popen]):
    """Clean up server processes"""
    print("\n🛑 Shutting down servers...")
    
    for process in processes:
        if process and process.poll() is None:
            try:
                process.terminate()
                process.wait(timeout=5)
                print(f"✅ Process {process.pid} terminated gracefully")
            except subprocess.TimeoutExpired:
                process.kill()
                print(f"⚠️ Process {process.pid} force-killed")
            except Exception as e:
                print(f"❌ Error terminating process {process.pid}: {e}")

def main():
    """Main function to start the complete workflow system"""
    print("🎬 WAN VIDEO EVALUATION - HUMAN REVIEW WORKFLOW SYSTEM")
    print("=" * 70)
    
    # Check requirements
    if not check_requirements():
        print("\n❌ Requirements check failed. Please ensure all files are present.")
        return 1
    
    # Initialize workflow
    if not initialize_workflow():
        print("\n❌ Workflow initialization failed.")
        return 1
    
    # Start API server
    api_process = start_api_server()
    if not api_process:
        print("\n❌ Failed to start API server.")
        return 1
    
    processes = [api_process]
    
    try:
        # Show access information
        show_urls()
        
        # Wait for user interrupt
        print("\n⏳ System running... Press Ctrl+C to stop")
        
        while True:
            time.sleep(1)
            
            # Check if processes are still running
            for process in processes:
                if process and process.poll() is not None:
                    print(f"\n⚠️ Process {process.pid} stopped unexpectedly")
                    return 1
    
    except KeyboardInterrupt:
        print("\n\n🛑 Shutdown requested by user")
    
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
    
    finally:
        cleanup_processes(processes)
        print("✅ Human Review Workflow System stopped")
    
    return 0

if __name__ == "__main__":
    exit(main())
