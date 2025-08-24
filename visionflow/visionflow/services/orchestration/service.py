"""
FastAPI service for LangGraph orchestration endpoints
Supports both workflow and agent patterns based on request complexity
"""

import asyncio
import json
import uuid
from datetime import datetime
from typing import Dict, Any, Optional, List

from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from .langgraph_orchestrator import get_orchestrator
from .agent_orchestrator import get_agent
from ...shared.config import get_settings
from ...shared.monitoring import get_logger
from ...shared.models import JobResponse, JobStatusResponse

logger = get_logger("orchestration_service")
settings = get_settings()

app = FastAPI(
    title="VisionFlow Orchestration Service",
    description="LangChain/LangGraph orchestration for video generation",
    version="0.1.0"
)


class OrchestrationRequest(BaseModel):
    """Request model for orchestration"""
    user_request: str = Field(..., description="User's video generation request")
    orchestration_type: Optional[str] = Field(
        default="auto", 
        description="Orchestration type: 'workflow' for structured patterns, 'agent' for open-ended tasks, 'auto' for automatic selection"
    )
    priority: Optional[str] = Field(default="normal", description="Request priority")
    callback_url: Optional[str] = Field(None, description="Callback URL for completion")
    user_id: Optional[str] = Field(None, description="User identifier")
    session_id: Optional[str] = Field(None, description="Session identifier")


class WorkflowStatusResponse(BaseModel):
    """Response model for workflow status"""
    job_id: str
    status: str
    current_step: Optional[str] = None
    progress: Optional[float] = None
    video_url: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    created_at: datetime
    updated_at: datetime


# In-memory job tracking (in production, use Redis or database)
job_store: Dict[str, Dict[str, Any]] = {}


def update_job_status(job_id: str, status: str, **kwargs):
    """Update job status in store"""
    if job_id not in job_store:
        job_store[job_id] = {
            "job_id": job_id,
            "status": status,
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow()
        }
    else:
        job_store[job_id]["status"] = status
        job_store[job_id]["updated_at"] = datetime.utcnow()
    
    # Update any additional fields
    job_store[job_id].update(kwargs)


def determine_orchestration_type(user_request: str, requested_type: str) -> str:
    """Determine whether to use workflow or agent pattern"""
    if requested_type in ["workflow", "agent"]:
        return requested_type
    
    # Auto-selection logic based on request complexity
    request_lower = user_request.lower()
    
    # Use agent for open-ended, exploratory, or multi-step requests
    if any(keyword in request_lower for keyword in [
        "explore", "try different", "experiment", "multiple", "various", 
        "edit", "modify", "iterate", "improve", "refine", "customize"
    ]):
        return "agent"
    
    # Use workflow for structured, predictable requests
    return "workflow"


async def run_orchestration_workflow(job_id: str, request: OrchestrationRequest):
    """Background task to run orchestration (workflow or agent pattern)"""
    
    try:
        update_job_status(job_id, "processing", current_step="initializing")
        
        # Determine orchestration type
        orchestration_type = determine_orchestration_type(
            request.user_request, 
            request.orchestration_type
        )
        
        logger.info(f"Starting {orchestration_type} orchestration for job {job_id}", extra={
            "job_id": job_id,
            "orchestration_type": orchestration_type,
            "user_request": request.user_request[:100] + "..." if len(request.user_request) > 100 else request.user_request
        })
        
        # Run appropriate orchestration pattern
        if orchestration_type == "agent":
            agent = get_agent()
            result = await agent.run_agent(
                user_request=request.user_request,
                job_id=job_id
            )
        else:  # workflow
            orchestrator = get_orchestrator()
            result = await orchestrator.orchestrate_video_generation(
                user_request=request.user_request,
                job_id=job_id
            )
        
        # Update job with results
        update_job_status(
            job_id=job_id,
            status=result["status"],
            video_url=result.get("video_url"),
            metadata=result.get("metadata"),
            error=result.get("error"),
            orchestration_type=orchestration_type
        )
        
        logger.info(f"{orchestration_type.capitalize()} orchestration completed for job {job_id}", extra={
            "job_id": job_id,
            "orchestration_type": orchestration_type,
            "status": result["status"],
            "video_url": result.get("video_url"),
            "execution_time": result.get("execution_time"),
            "quality_score": result.get("quality_score"),
            "iterations": result.get("iteration_count") or result.get("agent_iterations"),
            "tool_calls": result.get("tool_calls")
        })
        
        # If callback URL provided, send notification
        if request.callback_url:
            # In production, implement callback notification
            logger.info(f"Would send callback to {request.callback_url}")
            
    except Exception as e:
        logger.error(f"Orchestration failed for job {job_id}: {e}", extra={
            "job_id": job_id,
            "error": str(e)
        })
        
        update_job_status(
            job_id=job_id,
            status="failed",
            error=str(e)
        )


@app.post("/orchestrate", response_model=JobResponse)
async def create_orchestration_job(
    request: OrchestrationRequest,
    background_tasks: BackgroundTasks
):
    """Create a new orchestration job"""
    
    job_id = str(uuid.uuid4())
    
    # Initialize job in store
    update_job_status(
        job_id=job_id,
        status="queued",
        user_request=request.user_request,
        priority=request.priority,
        user_id=request.user_id,
        session_id=request.session_id
    )
    
    # Start background orchestration
    background_tasks.add_task(run_orchestration_workflow, job_id, request)
    
    logger.info(f"Created orchestration job {job_id}", extra={
        "job_id": job_id,
        "user_request": request.user_request
    })
    
    return JobResponse(
        job_id=job_id,
        status="queued",
        message="Orchestration job created successfully"
    )


@app.get("/jobs/{job_id}/status", response_model=WorkflowStatusResponse)
async def get_job_status(job_id: str):
    """Get status of orchestration job"""
    
    if job_id not in job_store:
        raise HTTPException(
            status_code=404,
            detail=f"Job {job_id} not found"
        )
    
    job_data = job_store[job_id]
    
    return WorkflowStatusResponse(**job_data)


@app.get("/jobs", response_model=List[WorkflowStatusResponse])
async def list_jobs(
    status: Optional[str] = None,
    user_id: Optional[str] = None,
    limit: int = 50
):
    """List orchestration jobs with optional filtering"""
    
    jobs = list(job_store.values())
    
    # Apply filters
    if status:
        jobs = [job for job in jobs if job.get("status") == status]
    
    if user_id:
        jobs = [job for job in jobs if job.get("user_id") == user_id]
    
    # Sort by created_at descending
    jobs.sort(key=lambda x: x.get("created_at", datetime.min), reverse=True)
    
    # Apply limit
    jobs = jobs[:limit]
    
    return [WorkflowStatusResponse(**job) for job in jobs]


@app.delete("/jobs/{job_id}")
async def cancel_job(job_id: str):
    """Cancel a running orchestration job"""
    
    if job_id not in job_store:
        raise HTTPException(
            status_code=404,
            detail=f"Job {job_id} not found"
        )
    
    job_data = job_store[job_id]
    
    if job_data["status"] in ["completed", "failed", "cancelled"]:
        raise HTTPException(
            status_code=400,
            detail=f"Cannot cancel job in {job_data['status']} status"
        )
    
    update_job_status(job_id, "cancelled")
    
    logger.info(f"Cancelled orchestration job {job_id}")
    
    return {"message": f"Job {job_id} cancelled successfully"}


@app.get("/jobs/{job_id}/stream")
async def stream_job_status(job_id: str):
    """Stream real-time job status updates"""
    
    if job_id not in job_store:
        raise HTTPException(
            status_code=404,
            detail=f"Job {job_id} not found"
        )
    
    async def generate_status_stream():
        """Generate SSE stream of job status updates"""
        last_status = None
        
        while True:
            if job_id in job_store:
                current_status = job_store[job_id]["status"]
                
                if current_status != last_status:
                    job_data = job_store[job_id]
                    yield f"data: {json.dumps(job_data, default=str)}\n\n"
                    last_status = current_status
                    
                    # Stop streaming if job is complete
                    if current_status in ["completed", "failed", "cancelled"]:
                        break
            
            await asyncio.sleep(1)  # Poll every second
    
    return StreamingResponse(
        generate_status_stream(),
        media_type="text/plain",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Content-Type": "text/event-stream"
        }
    )


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    orchestrator = get_orchestrator()
    
    return {
        "status": "healthy",
        "service": "visionflow-orchestration",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "0.1.0",
        "orchestrator_initialized": orchestrator is not None
    }


@app.get("/metrics")
async def get_metrics():
    """Prometheus metrics endpoint"""
    
    # Basic metrics about jobs
    total_jobs = len(job_store)
    status_counts = {}
    
    for job in job_store.values():
        status = job.get("status", "unknown")
        status_counts[status] = status_counts.get(status, 0) + 1
    
    metrics = [
        f"visionflow_orchestration_jobs_total {total_jobs}",
        f"visionflow_orchestration_service_up 1",
    ]
    
    # Add status-specific metrics
    for status, count in status_counts.items():
        metrics.append(f'visionflow_orchestration_jobs_by_status{{status="{status}"}} {count}')
    
    return "\n".join(metrics) + "\n"


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "visionflow.services.orchestration.service:app",
        host="0.0.0.0",
        port=8001,
        reload=False,
        log_level="info"
    )
