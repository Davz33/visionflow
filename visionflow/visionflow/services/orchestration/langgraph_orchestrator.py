"""
LangGraph Orchestrator for VisionFlow - Implementing proper LangGraph patterns
Based on "Building Effective Agents with LangGraph" best practices
"""

import asyncio
import json
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional
from typing_extensions import TypedDict
from pathlib import Path

from google.cloud import storage, aiplatform
from langchain.schema import BaseMessage, HumanMessage, SystemMessage
from langchain.tools import tool
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.runnables import RunnableConfig
from langchain_google_vertexai import ChatVertexAI
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field

from ...shared.config import get_settings
from ...shared.monitoring import get_logger
# from ..gcp_tracking import get_cloud_tracker, get_metrics_logger

logger = get_logger("langgraph_orchestrator")
settings = get_settings()


# Structured outputs for routing decisions (following knowledge base pattern)
class ComplexityRouting(BaseModel):
    """Routing decision based on complexity analysis"""
    complexity: str = Field(description="Complexity level: simple, complex, or hybrid")
    reasoning: str = Field(description="Reasoning for complexity assessment")


class QualityGrade(BaseModel):
    """Quality evaluation with feedback"""
    grade: str = Field(description="Quality grade: excellent, good, poor")
    feedback: str = Field(description="Specific feedback for improvement")
    score: float = Field(description="Quality score from 0.0 to 1.0")


class VideoGenerationPlan(BaseModel):
    """Dynamic plan for video generation workers"""
    sections: List[Dict[str, str]] = Field(description="List of generation sections/tasks")
    estimated_duration: int = Field(description="Estimated total duration in seconds")
    complexity_level: str = Field(description="Overall complexity assessment")


# State containers (following knowledge base pattern)
class VideoGenerationState(TypedDict):
    """Main workflow state container"""
    messages: List[BaseMessage]
    user_request: str
    complexity_routing: Optional[ComplexityRouting]
    generation_plan: Optional[VideoGenerationPlan]
    completed_sections: List[Dict[str, Any]]  # For orchestrator-worker pattern
    scene_description: Optional[str]
    generation_params: Optional[Dict[str, Any]]
    quality_grade: Optional[QualityGrade]
    video_url: Optional[str]
    metadata: Optional[Dict[str, Any]]
    error: Optional[str]
    status: str
    iteration_count: int  # For evaluator-optimizer loops


class WorkerState(TypedDict):
    """Individual worker state (for orchestrator-worker pattern)"""
    section_name: str
    section_description: str
    worker_id: str
    completed_sections: List[Dict[str, Any]]  # Shared with main state
    status: str


class VisionFlowCallbackHandler(BaseCallbackHandler):
    """Custom callback handler for monitoring LangGraph operations"""
    
    def __init__(self, job_id: str):
        self.job_id = job_id
        self.start_time = datetime.utcnow()
    
    def on_chain_start(self, serialized: Dict[str, Any], inputs: Dict[str, Any], **kwargs) -> None:
        logger.info(f"LangGraph workflow started for job {self.job_id}", extra={
            "job_id": self.job_id,
            "workflow": serialized.get("name", "unknown"),
            "inputs": inputs
        })
    
    def on_chain_end(self, outputs: Dict[str, Any], **kwargs) -> None:
        duration = (datetime.utcnow() - self.start_time).total_seconds()
        logger.info(f"LangGraph workflow completed for job {self.job_id}", extra={
            "job_id": self.job_id,
            "duration": duration,
            "outputs": outputs
        })
    
    def on_chain_error(self, error: Exception, **kwargs) -> None:
        logger.error(f"LangGraph workflow error for job {self.job_id}: {error}", extra={
            "job_id": self.job_id,
            "error": str(error)
        })


# Tools (following knowledge base @tool pattern)
@tool
def analyze_complexity(user_request: str) -> ComplexityRouting:
    """Analyze request complexity for routing decisions"""
    request_length = len(user_request.split())
    
    # Complexity heuristics
    if request_length < 10 and not any(keyword in user_request.lower() 
                                       for keyword in ["character", "style", "multi", "complex"]):
        complexity = "simple"
        reasoning = "Short request with basic requirements"
    elif request_length > 20 or any(keyword in user_request.lower() 
                                    for keyword in ["character", "style", "scene", "narrative"]):
        complexity = "complex"
        reasoning = "Long request or contains complex elements requiring context retrieval"
    else:
        complexity = "hybrid"
        reasoning = "Medium complexity request with some advanced features"
    
    return ComplexityRouting(complexity=complexity, reasoning=reasoning)


@tool
def create_generation_plan(user_request: str, complexity: str) -> VideoGenerationPlan:
    """Create dynamic plan for video generation (orchestrator function)"""
    # This LLM call would analyze the request and create a plan
    # Simplified for demonstration
    
    if complexity == "simple":
        sections = [
            {"name": "scene_generation", "description": f"Generate simple scene for: {user_request}"}
        ]
        duration = 5
    elif complexity == "complex":
        sections = [
            {"name": "character_analysis", "description": f"Analyze characters mentioned in: {user_request}"},
            {"name": "scene_composition", "description": f"Design scene composition for: {user_request}"},
            {"name": "style_application", "description": f"Apply appropriate style for: {user_request}"},
            {"name": "narrative_flow", "description": f"Ensure narrative coherence for: {user_request}"}
        ]
        duration = 15
    else:  # hybrid
        sections = [
            {"name": "scene_analysis", "description": f"Analyze scene requirements for: {user_request}"},
            {"name": "content_generation", "description": f"Generate content for: {user_request}"}
        ]
        duration = 10
    
    return VideoGenerationPlan(
        sections=sections,
        estimated_duration=duration,
        complexity_level=complexity
    )


@tool
def evaluate_video_quality(video_description: str, original_request: str) -> QualityGrade:
    """Evaluate generated video quality and provide feedback"""
    # This would use an LLM to evaluate quality
    # Simplified for demonstration
    
    if len(video_description) < 50:
        grade = "poor"
        feedback = "Description too brief, needs more detail and specificity"
        score = 0.3
    elif len(video_description) > 200:
        grade = "excellent" 
        feedback = "Comprehensive description with good detail"
        score = 0.9
    else:
        grade = "good"
        feedback = "Adequate description, could use more visual details"
        score = 0.7
    
    return QualityGrade(grade=grade, feedback=feedback, score=score)


@tool
def generate_scene_description(request: str, context: Optional[Dict] = None) -> str:
    """Generate enhanced scene description"""
    # Enhanced prompt with context if available
    base_description = f"Cinematic scene: {request}"
    
    if context:
        enhanced_description = f"{base_description}. Enhanced with context: {context}"
    else:
        enhanced_description = f"{base_description}. High-quality cinematic composition with dynamic lighting."
    
    return enhanced_description


@tool
def upload_to_gcs(file_path: str, bucket_name: str) -> str:
    """Upload file to Google Cloud Storage"""
    try:
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob_name = f"videos/{uuid.uuid4()}.mp4"
        blob = bucket.blob(blob_name)
        
        # For demo, we'll simulate upload
        # blob.upload_from_filename(file_path)
        # blob.make_public()
        
        # Return mock URL
        return f"https://storage.googleapis.com/{bucket_name}/{blob_name}"
    except Exception as e:
        logger.error(f"Failed to upload to GCS: {e}")
        raise


class LangGraphOrchestrator:
    """Main orchestrator implementing proper LangGraph patterns from knowledge base"""
    
    def __init__(self):
        self.settings = get_settings()
        self.llm = self._initialize_llm()
        self.workflow = self._create_workflow()
        # self.cloud_tracker = get_cloud_tracker()
        # self.metrics_logger = get_metrics_logger()
        
    def _initialize_llm(self) -> ChatVertexAI:
        """Initialize Vertex AI LLM"""
        return ChatVertexAI(
            model_name="gemini-pro",
            project=settings.monitoring.vertex_ai_project,
            location=settings.monitoring.vertex_ai_region,
            temperature=0.7,
            max_output_tokens=1024
        )
    
    def _create_workflow(self) -> StateGraph:
        """Create LangGraph workflow implementing knowledge base patterns"""
        
        # Define the workflow graph
        workflow = StateGraph(VideoGenerationState)
        
        # Add nodes
        workflow.add_node("analyze_complexity", self._analyze_complexity_node)
        workflow.add_node("create_plan", self._create_plan_node)
        workflow.add_node("execute_workers", self._execute_workers_node)
        workflow.add_node("worker_task", self._worker_task_node)  # For orchestrator-worker
        workflow.add_node("generate_description", self._generate_description_node)
        workflow.add_node("evaluate_quality", self._evaluate_quality_node)
        workflow.add_node("generate_video", self._generate_video_node)
        workflow.add_node("post_process", self._post_process_node)
        workflow.add_node("handle_error", self._handle_error_node)
        
        # Define edges (workflow flow)
        workflow.add_edge(START, "analyze_complexity")
        
        # Routing pattern - LLM decides path based on complexity
        workflow.add_conditional_edges(
            "analyze_complexity",
            self._route_by_complexity,
            {
                "simple": "generate_description",
                "complex": "create_plan", 
                "hybrid": "create_plan",
                "error": "handle_error"
            }
        )
        
        # Orchestrator-worker pattern for complex requests
        workflow.add_edge("create_plan", "execute_workers")
        workflow.add_edge("execute_workers", "generate_description")
        
        # Evaluator-optimizer pattern
        workflow.add_edge("generate_description", "evaluate_quality")
        workflow.add_conditional_edges(
            "evaluate_quality",
            self._route_by_quality,
            {
                "excellent": "generate_video",
                "good": "generate_video",
                "poor": "generate_description",  # Loop back for improvement
                "error": "handle_error"
            }
        )
        
        workflow.add_edge("generate_video", "post_process")
        workflow.add_edge("post_process", END)
        workflow.add_edge("handle_error", END)
        
        return workflow.compile()
    
    async def _analyze_complexity_node(self, state: VideoGenerationState) -> VideoGenerationState:
        """Analyze request complexity for routing (following knowledge base routing pattern)"""
        try:
            complexity_routing = analyze_complexity(state["user_request"])
            state["complexity_routing"] = complexity_routing
            state["status"] = "complexity_analyzed"
            
            logger.info("Complexity analysis completed", extra={
                "complexity": complexity_routing.complexity,
                "reasoning": complexity_routing.reasoning
            })
            
            return state
        except Exception as e:
            logger.error(f"Complexity analysis failed: {e}")
            state["error"] = str(e)
            state["status"] = "error"
            return state
    
    async def _create_plan_node(self, state: VideoGenerationState) -> VideoGenerationState:
        """Create generation plan (orchestrator function)"""
        try:
            complexity = state["complexity_routing"].complexity
            plan = create_generation_plan(state["user_request"], complexity)
            state["generation_plan"] = plan
            state["completed_sections"] = []  # Initialize for workers
            state["status"] = "plan_created"
            
            logger.info(f"Generation plan created with {len(plan.sections)} sections")
            return state
        except Exception as e:
            logger.error(f"Plan creation failed: {e}")
            state["error"] = str(e)
            state["status"] = "error"
            return state
    
    async def _execute_workers_node(self, state: VideoGenerationState) -> VideoGenerationState:
        """Execute workers using LangGraph send API (orchestrator-worker pattern)"""
        try:
            plan = state["generation_plan"]
            
            # This is where we'd use LangGraph's send API to spawn workers dynamically
            # For now, we'll simulate parallel execution
            worker_tasks = []
            
            for section in plan.sections:
                # Create worker state
                worker_state = WorkerState(
                    section_name=section["name"],
                    section_description=section["description"],
                    worker_id=str(uuid.uuid4()),
                    completed_sections=state["completed_sections"],
                    status="pending"
                )
                
                # Simulate worker execution
                task = self._execute_worker(worker_state)
                worker_tasks.append(task)
            
            # Execute all workers in parallel
            worker_results = await asyncio.gather(*worker_tasks, return_exceptions=True)
            
            # Update state with completed sections
            for result in worker_results:
                if isinstance(result, dict) and "section_result" in result:
                    state["completed_sections"].append(result)
            
            state["status"] = "workers_completed"
            logger.info(f"Completed {len(state['completed_sections'])} worker tasks")
            
            return state
        except Exception as e:
            logger.error(f"Worker execution failed: {e}")
            state["error"] = str(e)
            state["status"] = "error"
            return state
    
    async def _execute_worker(self, worker_state: WorkerState) -> Dict[str, Any]:
        """Individual worker execution"""
        try:
            # Simulate worker processing
            await asyncio.sleep(0.5)  # Simulate work
            
            result = {
                "section_name": worker_state["section_name"],
                "section_result": f"Processed {worker_state['section_description']}",
                "worker_id": worker_state["worker_id"],
                "completion_time": datetime.utcnow().isoformat()
            }
            
            logger.info(f"Worker {worker_state['worker_id']} completed section {worker_state['section_name']}")
            return result
        except Exception as e:
            logger.error(f"Worker {worker_state['worker_id']} failed: {e}")
            return {"error": str(e)}
    
    async def _worker_task_node(self, state: WorkerState) -> WorkerState:
        """Worker node for LangGraph (when using send API)"""
        # This would be used with LangGraph's send API for true dynamic worker spawning
        pass
    
    async def _generate_description_node(self, state: VideoGenerationState) -> VideoGenerationState:
        """Generate scene description (can be improved through evaluator-optimizer loop)"""
        try:
            # Include context from completed workers if available
            context = None
            if state.get("completed_sections"):
                context = {"worker_results": state["completed_sections"]}
            
            description = generate_scene_description(state["user_request"], context)
            state["scene_description"] = description
            state["status"] = "description_generated"
            
            # Initialize iteration count for evaluator-optimizer loop
            if "iteration_count" not in state:
                state["iteration_count"] = 0
            state["iteration_count"] += 1
            
            return state
        except Exception as e:
            logger.error(f"Description generation failed: {e}")
            state["error"] = str(e)
            state["status"] = "error"
            return state
    
    async def _evaluate_quality_node(self, state: VideoGenerationState) -> VideoGenerationState:
        """Evaluate quality and provide feedback (evaluator-optimizer pattern)"""
        try:
            quality_grade = evaluate_video_quality(
                state["scene_description"], 
                state["user_request"]
            )
            state["quality_grade"] = quality_grade
            state["status"] = "quality_evaluated"
            
            logger.info(f"Quality evaluation: {quality_grade.grade} (score: {quality_grade.score})")
            
            # If quality is poor and we haven't tried too many times, provide feedback
            if quality_grade.grade == "poor" and state["iteration_count"] < 3:
                logger.info(f"Quality insufficient, iteration {state['iteration_count']}, providing feedback: {quality_grade.feedback}")
            
            return state
        except Exception as e:
            logger.error(f"Quality evaluation failed: {e}")
            state["error"] = str(e)
            state["status"] = "error"
            return state
    
    async def _generate_video_node(self, state: VideoGenerationState) -> VideoGenerationState:
        """Generate video with final parameters"""
        try:
            # Create generation parameters
            params = {
                "prompt": state["scene_description"],
                "duration": 10 if state.get("generation_plan") else 5,
                "resolution": "512x512",
                "fps": 24,
                "quality": state["quality_grade"].grade if state.get("quality_grade") else "good"
            }
            
            state["generation_params"] = params
            
            # Simulate video generation
            await asyncio.sleep(2)
            
            # Upload to storage
            video_url = upload_to_gcs(
                "/tmp/generated_video.mp4", 
                settings.storage.bucket_name
            )
            
            state["video_url"] = video_url
            state["status"] = "video_generated"
            
            return state
        except Exception as e:
            logger.error(f"Video generation failed: {e}")
            state["error"] = str(e)
            state["status"] = "error"
            return state
    
    async def _post_process_node(self, state: VideoGenerationState) -> VideoGenerationState:
        """Post-process and create metadata"""
        try:
            metadata = {
                "generation_time": datetime.utcnow().isoformat(),
                "complexity": state["complexity_routing"].complexity if state.get("complexity_routing") else "unknown",
                "quality_score": state["quality_grade"].score if state.get("quality_grade") else 0.5,
                "iterations": state.get("iteration_count", 1),
                "worker_sections": len(state.get("completed_sections", [])),
                "parameters": state["generation_params"]
            }
            
            state["metadata"] = metadata
            state["status"] = "completed"
            
            logger.info("Video generation workflow completed successfully", extra={
                "video_url": state["video_url"],
                "quality_score": metadata["quality_score"],
                "iterations": metadata["iterations"]
            })
            
            return state
        except Exception as e:
            logger.error(f"Post-processing failed: {e}")
            state["error"] = str(e)
            state["status"] = "error"
            return state
    
    async def _handle_error_node(self, state: VideoGenerationState) -> VideoGenerationState:
        """Handle errors and provide fallback response"""
        logger.error(f"Workflow error: {state.get('error', 'Unknown error')}")
        
        state["status"] = "failed"
        state["video_url"] = None
        
        return state
    
    def _route_by_complexity(self, state: VideoGenerationState) -> str:
        """Route based on complexity analysis (following knowledge base routing pattern)"""
        if "error" in state:
            return "error"
        
        complexity_routing = state.get("complexity_routing")
        if not complexity_routing:
            return "error"
        
        return complexity_routing.complexity
    
    def _route_by_quality(self, state: VideoGenerationState) -> str:
        """Route based on quality evaluation (evaluator-optimizer pattern)"""
        if "error" in state:
            return "error"
        
        quality_grade = state.get("quality_grade")
        if not quality_grade:
            return "error"
        
        # If poor quality and haven't exceeded max iterations, loop back
        if quality_grade.grade == "poor" and state.get("iteration_count", 0) < 3:
            return "poor"
        
        return quality_grade.grade
    
    async def orchestrate_video_generation(
        self, 
        user_request: str,
        job_id: str = None
    ) -> Dict[str, Any]:
        """Main orchestration method implementing LangGraph best practices"""
        
        if not job_id:
            job_id = str(uuid.uuid4())
        
        # Start GCP tracking run
        tracking_run_id = self.cloud_tracker.start_run(
            run_name=f"langgraph_orchestration_{job_id}",
            tags={
                "job_id": job_id,
                "task": "video_generation_langgraph",
                "user_request_length": len(user_request)
            }
        )
        
        # Initialize state (simple container following knowledge base pattern)
        initial_state = VideoGenerationState(
            messages=[HumanMessage(content=user_request)],
            user_request=user_request,
            complexity_routing=None,
            generation_plan=None,
            completed_sections=[],
            scene_description=None,
            generation_params=None,
            quality_grade=None,
            video_url=None,
            metadata=None,
            error=None,
            status="started",
            iteration_count=0
        )
        
        # Set up callback handler
        callback_handler = VisionFlowCallbackHandler(job_id)
        config = RunnableConfig(callbacks=[callback_handler])
        
        start_time = datetime.utcnow()
        
        try:
            # Log initial parameters
            if tracking_run_id:
                self.cloud_tracker.log_parameters({
                    "user_request": user_request[:500],
                    "job_id": job_id,
                    "workflow_version": "2.0.0",
                    "orchestration_type": "langgraph_best_practices"
                })
            
            # Run the LangGraph workflow
            result = await self.workflow.ainvoke(initial_state, config=config)
            
            # Calculate execution time
            execution_time = (datetime.utcnow() - start_time).total_seconds()
            
            # Log metrics to both GCP tracking and metrics logger
            if tracking_run_id:
                metrics = {
                    "execution_time_seconds": execution_time,
                    "success": 1 if result["status"] == "completed" else 0,
                    "quality_score": result.get("quality_grade", {}).get("score", 0.0) if result.get("quality_grade") else 0.0,
                    "iteration_count": result.get("iteration_count", 1),
                    "worker_sections": len(result.get("completed_sections", [])),
                    "complexity_numeric": 1 if result.get("complexity_routing", {}).get("complexity") == "simple" else 3 if result.get("complexity_routing", {}).get("complexity") == "complex" else 2
                }
                
                self.cloud_tracker.log_metrics(metrics)
                
                # Log to real-time metrics for monitoring
                self.metrics_logger.log_video_generation_metrics(
                    duration=execution_time,
                    status=result["status"],
                    model="vertex_ai_gemini",
                    orchestration_type="workflow",
                    quality_score=result.get("quality_grade", {}).get("score") if result.get("quality_grade") else None,
                    iterations=result.get("iteration_count", 1)
                )
            
            return {
                "job_id": job_id,
                "tracking_run_id": tracking_run_id,
                "status": result["status"],
                "video_url": result.get("video_url"),
                "metadata": result.get("metadata"),
                "error": result.get("error"),
                "execution_time": execution_time,
                "quality_score": result.get("quality_grade", {}).get("score") if result.get("quality_grade") else None,
                "iteration_count": result.get("iteration_count", 1)
            }
            
        except Exception as e:
            logger.error(f"LangGraph orchestration failed for job {job_id}: {e}")
            
            # Log error to GCP tracking
            if tracking_run_id:
                execution_time = (datetime.utcnow() - start_time).total_seconds()
                self.cloud_tracker.log_metrics({
                    "execution_time_seconds": execution_time,
                    "success": 0,
                    "error": 1
                })
                self.cloud_tracker.log_parameters({
                    "error_message": str(e)[:500]
                })
                
                # Log to metrics for monitoring
                self.metrics_logger.log_video_generation_metrics(
                    duration=execution_time,
                    status="failed",
                    model="vertex_ai_gemini",
                    orchestration_type="workflow"
                )
            
            return {
                "job_id": job_id,
                "tracking_run_id": tracking_run_id,
                "status": "failed",
                "video_url": None,
                "metadata": None,
                "error": str(e),
                "execution_time": (datetime.utcnow() - start_time).total_seconds()
            }
        
        finally:
            # End GCP tracking run
            if tracking_run_id:
                self.cloud_tracker.end_run()


# Singleton instance
_orchestrator_instance = None

def get_orchestrator() -> LangGraphOrchestrator:
    """Get or create orchestrator instance"""
    global _orchestrator_instance
    if _orchestrator_instance is None:
        _orchestrator_instance = LangGraphOrchestrator()
    return _orchestrator_instance
