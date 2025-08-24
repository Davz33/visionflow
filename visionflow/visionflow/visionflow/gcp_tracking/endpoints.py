"""
GCP Tracking API endpoints for VisionFlow
Provides access to tracking data, metrics, and model registry
"""

from datetime import datetime
from typing import Dict, Any, List, Optional

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from .cloud_tracker import get_cloud_tracker
from .model_registry import get_model_registry, ModelStage
from .metrics_logger import get_metrics_logger
from ...shared.monitoring import get_logger

logger = get_logger("gcp_tracking_endpoints")

app = FastAPI(
    title="VisionFlow GCP Tracking API",
    description="GCP-based tracking and monitoring endpoints",
    version="1.0.0"
)


class ModelRegistrationRequest(BaseModel):
    """Request to register model"""
    model_path: str = Field(..., description="Path to model")
    model_name: str = Field(..., description="Model name")
    version: Optional[str] = Field(None, description="Model version (auto-generated if not provided)")
    description: Optional[str] = Field(None, description="Model description")
    tags: Optional[Dict[str, str]] = Field(default_factory=dict, description="Model tags")
    stage: str = Field(default="none", description="Initial stage (none, staging, production)")


class ModelTransitionRequest(BaseModel):
    """Request to transition model stage"""
    model_name: str = Field(..., description="Model name")
    version: str = Field(..., description="Model version")
    stage: str = Field(..., description="Target stage (staging/production/archived)")


@app.get("/tracking/runs")
async def list_tracking_runs(
    limit: int = Query(default=50, description="Maximum number of runs to return"),
    filter_task: Optional[str] = Query(default=None, description="Filter by task type")
):
    """List tracking runs from Cloud Logging"""
    try:
        tracker = get_cloud_tracker()
        
        filter_conditions = {}
        if filter_task:
            filter_conditions["task"] = filter_task
        
        runs = tracker.search_runs(
            filter_conditions=filter_conditions,
            limit=limit
        )
        
        return {
            "total_runs": len(runs),
            "runs": runs
        }
        
    except Exception as e:
        logger.error(f"Failed to list tracking runs: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/tracking/health")
async def tracking_health_check():
    """Check GCP tracking services health"""
    try:
        tracker = get_cloud_tracker()
        registry = get_model_registry()
        metrics = get_metrics_logger()
        
        tracker_health = tracker.health_check()
        registry_health = registry.health_check()
        metrics_health = metrics.health_check()
        
        overall_health = (
            tracker_health.get("cloud_logging_available", False) and
            tracker_health.get("cloud_storage_available", False) and
            metrics_health.get("cloud_monitoring_available", False)
        )
        
        return {
            "status": "healthy" if overall_health else "unhealthy",
            "tracking": tracker_health,
            "model_registry": registry_health,
            "metrics": metrics_health,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }


@app.get("/models")
async def list_models():
    """List all registered models"""
    try:
        registry = get_model_registry()
        models = registry.list_models()
        
        return {
            "models": models,
            "total_models": len(models)
        }
        
    except Exception as e:
        logger.error(f"Failed to list models: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/models/register")
async def register_model(request: ModelRegistrationRequest):
    """Register new model version"""
    try:
        registry = get_model_registry()
        
        # Convert stage string to enum
        try:
            stage = ModelStage(request.stage.lower())
        except ValueError:
            stage = ModelStage.NONE
        
        version = registry.register_model(
            model_name=request.model_name,
            model_path=request.model_path,
            version=request.version,
            description=request.description,
            tags=request.tags,
            stage=stage
        )
        
        if not version:
            raise HTTPException(status_code=400, detail="Failed to register model")
        
        return {
            "model_name": request.model_name,
            "version": version,
            "stage": stage.value,
            "message": "Model registered successfully"
        }
        
    except Exception as e:
        logger.error(f"Failed to register model: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/models/{model_name}")
async def get_model_details(model_name: str):
    """Get detailed model information"""
    try:
        registry = get_model_registry()
        
        # Get model versions
        versions = registry.list_model_versions(model_name)
        
        # Get production and staging models
        prod_model = registry.get_production_model(model_name)
        staging_model = registry.get_staging_model(model_name)
        
        return {
            "model_name": model_name,
            "total_versions": len(versions),
            "production_version": prod_model.get("version") if prod_model else None,
            "staging_version": staging_model.get("version") if staging_model else None,
            "versions": versions
        }
        
    except Exception as e:
        logger.error(f"Failed to get model details: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/models/transition")
async def transition_model_stage(request: ModelTransitionRequest):
    """Transition model to new stage"""
    try:
        registry = get_model_registry()
        
        # Convert stage string to enum
        try:
            stage = ModelStage(request.stage.lower())
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid stage: {request.stage}")
        
        success = registry.transition_model_stage(
            model_name=request.model_name,
            version=request.version,
            new_stage=stage,
            archive_existing=True
        )
        
        if not success:
            raise HTTPException(status_code=400, detail="Failed to transition model")
        
        return {
            "model_name": request.model_name,
            "version": request.version,
            "new_stage": stage.value,
            "message": f"Model transitioned to {stage.value} successfully"
        }
        
    except Exception as e:
        logger.error(f"Failed to transition model: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/models/{model_name}/rollback")
async def rollback_production_model(
    model_name: str,
    target_version: Optional[str] = Query(default=None, description="Target version to rollback to")
):
    """Rollback production model to previous version"""
    try:
        registry = get_model_registry()
        
        success = registry.rollback_production(
            model_name=model_name,
            target_version=target_version
        )
        
        if not success:
            raise HTTPException(status_code=400, detail="Failed to rollback model")
        
        # Get new production version
        prod_model = registry.get_production_model(model_name)
        new_version = prod_model.get("version") if prod_model else "unknown"
        
        return {
            "model_name": model_name,
            "rolled_back_to_version": new_version,
            "message": f"Model rolled back to version {new_version} successfully"
        }
        
    except Exception as e:
        logger.error(f"Failed to rollback model: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/metrics")
async def get_prometheus_metrics():
    """Get Prometheus-formatted metrics"""
    try:
        metrics_logger = get_metrics_logger()
        return metrics_logger.get_prometheus_metrics()
        
    except Exception as e:
        logger.error(f"Failed to get metrics: {e}")
        return "# Metrics unavailable\n"


@app.post("/metrics/custom")
async def log_custom_metric(
    metric_name: str = Query(..., description="Metric name"),
    value: float = Query(..., description="Metric value"),
    labels: Optional[Dict[str, str]] = None
):
    """Log custom metric"""
    try:
        metrics_logger = get_metrics_logger()
        metrics_logger.log_custom_metric(metric_name, value, labels or {})
        
        return {
            "message": f"Logged metric {metric_name}={value}",
            "labels": labels
        }
        
    except Exception as e:
        logger.error(f"Failed to log custom metric: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "visionflow.services.gcp_tracking.endpoints:app",
        host="0.0.0.0", 
        port=8003,
        reload=False
    )
