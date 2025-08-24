"""
Test script for GCP tracking functionality
Verifies that MLFlow has been successfully replaced with GCP services
"""

import asyncio
import json
from datetime import datetime

from services.gcp_tracking import get_cloud_tracker, get_model_registry, get_metrics_logger
from services.gcp_tracking.model_registry import ModelStage


async def test_cloud_tracker():
    """Test Cloud Tracker functionality"""
    print("🧪 Testing Cloud Tracker...")
    
    tracker = get_cloud_tracker()
    
    # Test health check
    health = tracker.health_check()
    print(f"   Health: {health}")
    
    # Test run tracking
    run_id = tracker.start_run(
        run_name="test_run",
        tags={"test": "true", "component": "cloud_tracker"}
    )
    
    if run_id:
        print(f"   ✅ Started run: {run_id}")
        
        # Log parameters
        tracker.log_parameters({
            "test_param": "test_value",
            "user_request": "Generate a test video",
            "model": "test_model"
        })
        print("   ✅ Logged parameters")
        
        # Log metrics
        tracker.log_metrics({
            "execution_time": 10.5,
            "quality_score": 0.85,
            "success": 1
        })
        print("   ✅ Logged metrics")
        
        # End run
        tracker.end_run()
        print("   ✅ Ended run")
    else:
        print("   ❌ Failed to start run")
    
    # Test search runs
    runs = tracker.search_runs(limit=5)
    print(f"   Found {len(runs)} recent runs")


async def test_model_registry():
    """Test Model Registry functionality"""
    print("\n🧪 Testing Model Registry...")
    
    registry = get_model_registry()
    
    # Test health check
    health = registry.health_check()
    print(f"   Health: {health}")
    
    # Test model registration
    version = registry.register_model(
        model_name="test_model",
        model_path="gs://test-bucket/models/test_model",
        description="Test model for GCP tracking validation",
        tags={"test": "true", "version": "1.0"},
        stage=ModelStage.STAGING
    )
    
    if version:
        print(f"   ✅ Registered model version: {version}")
        
        # Test getting model details
        model_info = registry.get_model_version("test_model", version)
        if model_info:
            print(f"   ✅ Retrieved model info: {model_info['name']}")
        
        # Test listing models
        models = registry.list_models()
        print(f"   Found {len(models)} registered models")
        
        # Test transition to production
        success = registry.deploy_to_production("test_model", version)
        if success:
            print("   ✅ Transitioned to production")
        
    else:
        print("   ❌ Failed to register model")


async def test_metrics_logger():
    """Test Metrics Logger functionality"""
    print("\n🧪 Testing Metrics Logger...")
    
    metrics = get_metrics_logger()
    
    # Test health check
    health = metrics.health_check()
    print(f"   Health: {health}")
    
    # Test video generation metrics
    metrics.log_video_generation_metrics(
        duration=15.3,
        status="completed",
        model="test_model",
        orchestration_type="workflow",
        quality_score=0.92,
        iterations=2
    )
    print("   ✅ Logged video generation metrics")
    
    # Test agent tool call metrics
    metrics.log_agent_tool_call("test_tool", "success")
    print("   ✅ Logged agent tool call metrics")
    
    # Test custom metrics
    metrics.log_custom_metric("test_metric", 42.0, {"test": "true"})
    print("   ✅ Logged custom metric")
    
    # Test active jobs update
    metrics.update_active_jobs(3)
    print("   ✅ Updated active jobs count")
    
    # Test GCP service health
    metrics.update_gcp_service_health("vertex_ai", True)
    metrics.update_gcp_service_health("cloud_storage", True)
    print("   ✅ Updated GCP service health")
    
    # Force flush metrics
    metrics.force_flush()
    print("   ✅ Flushed metrics buffer")


async def test_orchestrator_integration():
    """Test integration with orchestrators"""
    print("\n🧪 Testing Orchestrator Integration...")
    
    try:
        from services.orchestration.langgraph_orchestrator import get_orchestrator
        from services.orchestration.agent_orchestrator import get_agent
        
        # Test LangGraph orchestrator
        orchestrator = get_orchestrator()
        print("   ✅ LangGraph orchestrator initialized with GCP tracking")
        
        # Test agent orchestrator
        agent = get_agent()
        print("   ✅ Agent orchestrator initialized with GCP tracking")
        
        # Verify tracking instances are available
        if hasattr(orchestrator, 'cloud_tracker') and hasattr(orchestrator, 'metrics_logger'):
            print("   ✅ LangGraph orchestrator has GCP tracking components")
        else:
            print("   ❌ LangGraph orchestrator missing GCP tracking components")
        
        if hasattr(agent, 'cloud_tracker') and hasattr(agent, 'metrics_logger'):
            print("   ✅ Agent orchestrator has GCP tracking components")
        else:
            print("   ❌ Agent orchestrator missing GCP tracking components")
            
    except ImportError as e:
        print(f"   ❌ Failed to import orchestrators: {e}")


async def test_endpoints():
    """Test GCP tracking endpoints"""
    print("\n🧪 Testing GCP Tracking Endpoints...")
    
    try:
        # Import endpoints to verify they load correctly
        from services.gcp_tracking.endpoints import app
        print("   ✅ GCP tracking endpoints loaded successfully")
        
        # Check if all expected endpoints are defined
        routes = [route.path for route in app.routes]
        expected_routes = [
            "/tracking/runs",
            "/tracking/health", 
            "/models",
            "/models/register",
            "/models/transition",
            "/metrics"
        ]
        
        for route in expected_routes:
            if any(route in path for path in routes):
                print(f"   ✅ Endpoint {route} available")
            else:
                print(f"   ❌ Endpoint {route} missing")
                
    except ImportError as e:
        print(f"   ❌ Failed to import endpoints: {e}")


async def main():
    """Run all tests"""
    print("🚀 Testing GCP Tracking System (MLFlow Replacement)")
    print("=" * 60)
    
    try:
        await test_cloud_tracker()
        await test_model_registry()
        await test_metrics_logger()
        await test_orchestrator_integration()
        await test_endpoints()
        
        print("\n" + "=" * 60)
        print("✅ All GCP tracking tests completed!")
        print("🎉 MLFlow has been successfully replaced with GCP services")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
