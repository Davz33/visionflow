#!/usr/bin/env python3
"""
Review Workflow API Server
FastAPI server for human review workflow dashboard
"""

import json
import os
from typing import Dict, List, Any, Optional
from datetime import datetime
from pathlib import Path

try:
    from fastapi import FastAPI, HTTPException, status
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.staticfiles import StaticFiles
    from fastapi.responses import HTMLResponse
    from pydantic import BaseModel
    import uvicorn
except ImportError:
    print("❌ FastAPI dependencies not found. Installing...")
    import subprocess
    subprocess.check_call(["pip", "install", "fastapi", "uvicorn", "python-multipart"])
    from fastapi import FastAPI, HTTPException, status
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.staticfiles import StaticFiles
    from fastapi.responses import HTMLResponse
    from pydantic import BaseModel
    import uvicorn

from review_workflow_service import ReviewWorkflowService

# Pydantic models for API
class ReviewSubmission(BaseModel):
    decision: str  # approved, rejected, flagged
    score: Optional[float] = None
    comments: str = ""
    tags: List[str] = []

class ReviewerCreate(BaseModel):
    reviewer_id: str
    name: str
    email: str = ""
    specialization: str = ""

# Initialize FastAPI app
app = FastAPI(
    title="Human Review Workflow API",
    description="API for managing human review workflow operations",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify actual origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize review service
review_service = ReviewWorkflowService()

# Current reviewer (in production, this would come from authentication)
CURRENT_REVIEWER = "demo_reviewer"

@app.on_event("startup")
async def startup_event():
    """Initialize the review workflow on startup"""
    # Ensure demo reviewer exists
    review_service.create_reviewer(
        CURRENT_REVIEWER, 
        "Demo Reviewer", 
        "demo@example.com",
        "General Video Quality"
    )
    
    # Initialize review items from evaluation results if they don't exist
    evaluation_file = Path("evaluation_datasets/large_scale_samples/evaluation_results.json")
    if evaluation_file.exists():
        queue = review_service.get_review_queue()
        if not queue:
            count = review_service.create_review_items_from_evaluation(str(evaluation_file))
            print(f"✅ Initialized {count} review items from evaluation results")

# API Routes

@app.get("/api/review-queue")
async def get_review_queue(
    status: Optional[str] = None,
    priority: Optional[str] = None
) -> List[Dict[str, Any]]:
    """Get the review queue with optional filters"""
    try:
        queue = review_service.get_review_queue(
            status_filter=status,
            priority_filter=priority,
            reviewer_id=None  # Don't filter by reviewer for queue view
        )
        return queue
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/review-item/{video_id}")
async def get_review_item(video_id: str) -> Dict[str, Any]:
    """Get a specific review item by video ID"""
    try:
        queue = review_service.get_review_queue()
        item = next((item for item in queue if item['video_id'] == video_id), None)
        
        if not item:
            raise HTTPException(status_code=404, detail="Review item not found")
        
        return item
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/review-item/{video_id}/assign")
async def assign_reviewer(video_id: str) -> Dict[str, str]:
    """Assign current reviewer to a video"""
    try:
        success = review_service.assign_reviewer(video_id, CURRENT_REVIEWER)
        if not success:
            raise HTTPException(status_code=400, detail="Failed to assign reviewer")
        
        return {"message": f"Assigned {CURRENT_REVIEWER} to {video_id}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/review-item/{video_id}/review")
async def submit_review(video_id: str, review: ReviewSubmission) -> Dict[str, str]:
    """Submit a review for a video"""
    try:
        success = review_service.submit_review(
            video_id=video_id,
            reviewer_id=CURRENT_REVIEWER,
            decision=review.decision,
            score=review.score,
            comments=review.comments,
            tags=review.tags
        )
        
        if not success:
            raise HTTPException(status_code=400, detail="Failed to submit review")
        
        return {"message": f"Review submitted for {video_id}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/statistics")
async def get_statistics() -> Dict[str, Any]:
    """Get workflow statistics"""
    try:
        stats = review_service.get_workflow_statistics()
        return stats
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/reviewers")
async def create_reviewer(reviewer: ReviewerCreate) -> Dict[str, str]:
    """Create a new reviewer"""
    try:
        success = review_service.create_reviewer(
            reviewer.reviewer_id,
            reviewer.name,
            reviewer.email,
            reviewer.specialization
        )
        
        if not success:
            raise HTTPException(status_code=400, detail="Reviewer already exists")
        
        return {"message": f"Created reviewer {reviewer.name}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/export")
async def export_data() -> Dict[str, str]:
    """Export all review data"""
    try:
        export_path = f"review_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        review_service.export_review_data(export_path)
        return {"message": f"Data exported to {export_path}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Health check
@app.get("/api/health")
async def health_check() -> Dict[str, str]:
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

# Serve static files (evaluation viewer and review dashboard)
if Path("evaluation_viewer.html").exists():
    app.mount("/static", StaticFiles(directory="."), name="static")

# Serve the review dashboard as the root page
@app.get("/", response_class=HTMLResponse)
async def serve_dashboard():
    """Serve the human review dashboard"""
    dashboard_file = Path("human_review_dashboard.html")
    if dashboard_file.exists():
        return dashboard_file.read_text()
    else:
        return HTMLResponse(
            content="<h1>Human Review Dashboard</h1><p>Dashboard file not found. Please ensure human_review_dashboard.html exists.</p>",
            status_code=404
        )

# Serve the evaluation viewer
@app.get("/viewer", response_class=HTMLResponse)
async def serve_viewer():
    """Serve the evaluation viewer"""
    viewer_file = Path("evaluation_viewer.html")
    if viewer_file.exists():
        return viewer_file.read_text()
    else:
        return HTMLResponse(
            content="<h1>Evaluation Viewer</h1><p>Viewer file not found. Please ensure evaluation_viewer.html exists.</p>",
            status_code=404
        )

# API documentation
@app.get("/api")
async def api_docs():
    """API documentation and endpoints"""
    return {
        "title": "Human Review Workflow API",
        "version": "1.0.0",
        "endpoints": {
            "GET /api/review-queue": "Get review queue with optional filters",
            "GET /api/review-item/{video_id}": "Get specific review item",
            "POST /api/review-item/{video_id}/assign": "Assign reviewer to video",
            "POST /api/review-item/{video_id}/review": "Submit review for video",
            "GET /api/statistics": "Get workflow statistics",
            "POST /api/reviewers": "Create new reviewer",
            "GET /api/export": "Export all review data",
            "GET /api/health": "Health check"
        },
        "dashboard_urls": {
            "review_dashboard": "/",
            "evaluation_viewer": "/viewer",
            "api_docs": "/docs"
        }
    }

def main():
    """Run the API server"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Review Workflow API Server")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8082, help="Port to bind to")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    parser.add_argument("--init", action="store_true", help="Initialize review workflow")
    
    args = parser.parse_args()
    
    if args.init:
        print("🚀 Initializing Review Workflow Service...")
        
        # Create demo reviewer
        review_service.create_reviewer(
            "demo_reviewer", 
            "Demo Reviewer", 
            "demo@example.com",
            "General Video Quality"
        )
        
        # Initialize from evaluation results
        evaluation_file = "evaluation_datasets/large_scale_samples/evaluation_results.json"
        if Path(evaluation_file).exists():
            count = review_service.create_review_items_from_evaluation(evaluation_file)
            print(f"✅ Created {count} review items")
        else:
            print(f"⚠️ Evaluation file not found: {evaluation_file}")
        
        # Show statistics
        stats = review_service.get_workflow_statistics()
        print(f"📊 Review queue: {sum(stats['status_distribution'].values())} items")
        print(f"🎯 Priorities: {stats['priority_distribution']}")
    
    print(f"🌐 Starting Review Workflow API Server...")
    print(f"📱 Dashboard: http://{args.host}:{args.port}/")
    print(f"📊 Evaluation Viewer: http://{args.host}:{args.port}/viewer")
    print(f"🔧 API Docs: http://{args.host}:{args.port}/docs")
    print(f"📋 API Info: http://{args.host}:{args.port}/api")
    
    uvicorn.run(
        "review_api_server:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info"
    )

if __name__ == "__main__":
    main()
