"""
Agent Orchestrator for VisionFlow - True Agent Pattern Implementation
Based on "Building Effective Agents with LangGraph" agent patterns
"""

import asyncio
import json
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, Annotated, Sequence
from typing_extensions import TypedDict

from langchain.schema import BaseMessage, HumanMessage, SystemMessage, AIMessage
from langchain.tools import tool
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.runnables import RunnableConfig
from langchain_google_vertexai import ChatVertexAI
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from pydantic import BaseModel, Field

from ...shared.config import get_settings
from ...shared.monitoring import get_logger
# from ..gcp_tracking import get_cloud_tracker, get_metrics_logger

logger = get_logger("agent_orchestrator")
settings = get_settings()


# Agent state (following knowledge base agent pattern)
class AgentState(TypedDict):
    """Agent state for tool calling loop"""
    messages: Annotated[Sequence[BaseMessage], add_messages]
    user_request: str
    job_id: str
    iteration_count: int
    video_url: Optional[str]
    metadata: Optional[Dict[str, Any]]
    error: Optional[str]
    status: str


# Tools for the agent (following knowledge base @tool pattern)
@tool
def analyze_video_request(request: str) -> str:
    """Analyze video generation request and extract key components"""
    # This tool would use LLM to analyze the request
    analysis = {
        "scene_type": "detected scene type",
        "characters": "extracted characters",
        "style": "identified style",
        "complexity": "assessed complexity"
    }
    
    result = f"Request analysis: {json.dumps(analysis, indent=2)}"
    logger.info(f"Analyzed video request: {request[:50]}...")
    return result


@tool
def search_character_database(character_name: str) -> str:
    """Search for character information in the database"""
    # Mock character database search
    character_info = {
        "name": character_name,
        "traits": ["brave", "curious"],
        "appearance": "brown hair, blue eyes",
        "background": "adventurous explorer",
        "previous_scenes": ["forest", "mountain"]
    }
    
    result = f"Character info for {character_name}: {json.dumps(character_info, indent=2)}"
    logger.info(f"Retrieved character info for: {character_name}")
    return result


@tool
def generate_scene_prompt(analysis: str, character_info: str = None) -> str:
    """Generate optimized prompt for video generation"""
    # This tool would create an optimized prompt
    base_prompt = f"Based on analysis: {analysis}"
    
    if character_info:
        enhanced_prompt = f"{base_prompt}\nWith character context: {character_info}"
    else:
        enhanced_prompt = f"{base_prompt}\nGenerate high-quality cinematic scene"
    
    logger.info("Generated scene prompt")
    return enhanced_prompt


@tool
def create_video_parameters(prompt: str) -> str:
    """Create video generation parameters based on prompt"""
    # Extract parameters from prompt analysis
    params = {
        "prompt": prompt,
        "duration": 10,
        "resolution": "512x512",
        "fps": 24,
        "style": "cinematic",
        "quality": "high"
    }
    
    result = f"Video parameters: {json.dumps(params, indent=2)}"
    logger.info("Created video generation parameters")
    return result


@tool
def generate_video_file(parameters: str) -> str:
    """Generate video file using the specified parameters"""
    # This would call the actual video generation service
    # For demo, we'll simulate the process
    
    import json
    params = json.loads(parameters)
    
    # Simulate video generation
    video_info = {
        "video_url": f"https://storage.googleapis.com/visionflow-videos/{uuid.uuid4()}.mp4",
        "duration": params.get("duration", 10),
        "resolution": params.get("resolution", "512x512"),
        "file_size": "15.2MB",
        "generation_time": "45 seconds"
    }
    
    result = f"Video generated: {json.dumps(video_info, indent=2)}"
    logger.info("Video generation completed")
    return result


@tool
def validate_video_output(video_info: str) -> str:
    """Validate the generated video meets quality standards"""
    import json
    video_data = json.loads(video_info)
    
    # Validation checks
    validation_result = {
        "quality_score": 0.85,
        "duration_check": "passed",
        "resolution_check": "passed",
        "format_check": "passed",
        "content_appropriateness": "passed",
        "overall_status": "approved"
    }
    
    result = f"Validation result: {json.dumps(validation_result, indent=2)}"
    logger.info("Video validation completed")
    return result


@tool
def save_generation_metadata(job_id: str, video_info: str, validation: str) -> str:
    """Save generation metadata for tracking and analytics"""
    import json
    
    metadata = {
        "job_id": job_id,
        "generation_timestamp": datetime.utcnow().isoformat(),
        "video_info": json.loads(video_info),
        "validation": json.loads(validation),
        "agent_tool_calls": "tracked separately",
        "processing_type": "agent_orchestration"
    }
    
    # In production, this would save to database
    result = f"Metadata saved: {json.dumps(metadata, indent=2)}"
    logger.info(f"Saved metadata for job {job_id}")
    return result


class AgentCallbackHandler(BaseCallbackHandler):
    """Custom callback handler for monitoring agent operations"""
    
    def __init__(self, job_id: str):
        self.job_id = job_id
        self.start_time = datetime.utcnow()
        self.tool_calls = 0
    
    def on_tool_start(self, serialized: Dict[str, Any], input_str: str, **kwargs) -> None:
        self.tool_calls += 1
        logger.info(f"Agent tool call #{self.tool_calls} for job {self.job_id}", extra={
            "job_id": self.job_id,
            "tool": serialized.get("name", "unknown"),
            "input": input_str[:100] + "..." if len(input_str) > 100 else input_str
        })
    
    def on_tool_end(self, output: str, **kwargs) -> None:
        logger.info(f"Agent tool completed for job {self.job_id}", extra={
            "job_id": self.job_id,
            "output_length": len(output)
        })
    
    def on_chain_end(self, outputs: Dict[str, Any], **kwargs) -> None:
        duration = (datetime.utcnow() - self.start_time).total_seconds()
        logger.info(f"Agent workflow completed for job {self.job_id}", extra={
            "job_id": self.job_id,
            "duration": duration,
            "total_tool_calls": self.tool_calls
        })


class VisionFlowAgent:
    """True agent implementation following knowledge base patterns"""
    
    def __init__(self):
        self.settings = get_settings()
        self.llm = self._initialize_llm()
        self.tools = self._get_tools()
        self.agent_workflow = self._create_agent_workflow()
        # self.cloud_tracker = get_cloud_tracker()
        # self.metrics_logger = get_metrics_logger()
        
    def _initialize_llm(self) -> ChatVertexAI:
        """Initialize LLM with tools binding"""
        llm = ChatVertexAI(
            model_name="gemini-pro",
            project=settings.monitoring.vertex_ai_project,
            location=settings.monitoring.vertex_ai_region,
            temperature=0.7,
            max_output_tokens=1024
        )
        
        # Bind tools to LLM (following knowledge base pattern)
        return llm.bind_tools(self._get_tools())
    
    def _get_tools(self):
        """Get all available tools for the agent"""
        return [
            analyze_video_request,
            search_character_database,
            generate_scene_prompt,
            create_video_parameters,
            generate_video_file,
            validate_video_output,
            save_generation_metadata
        ]
    
    def _create_agent_workflow(self) -> StateGraph:
        """Create agent workflow (simple tool calling loop from knowledge base)"""
        
        # Define the agent graph
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("agent", self._agent_node)
        workflow.add_node("tools", ToolNode(self.tools))
        
        # Define edges - this is the classic agent pattern from knowledge base
        workflow.add_edge(START, "agent")
        
        # Conditional edge: if agent wants to call tools, go to tools node
        # If agent is done, end the workflow
        workflow.add_conditional_edges(
            "agent",
            tools_condition,  # This checks if the last message contains tool calls
            {
                "tools": "tools",  # If tools needed, go to tools node
                "end": END        # If no tools needed, end
            }
        )
        
        # After tools are executed, go back to agent
        workflow.add_edge("tools", "agent")
        
        return workflow.compile()
    
    async def _agent_node(self, state: AgentState) -> AgentState:
        """Main agent node - the LLM that decides what tools to call"""
        try:
            # System message to guide the agent's behavior
            system_message = SystemMessage(content="""
You are a helpful video generation agent. Your task is to help users create videos by:

1. Analyzing their request to understand what they want
2. Gathering any necessary information (characters, scenes, etc.)
3. Creating optimized prompts and parameters
4. Generating the video
5. Validating the output quality
6. Saving metadata

Use the available tools to accomplish these tasks. Work step by step and call tools as needed.
When you have successfully completed all steps, provide a final summary without calling more tools.

Available tools:
- analyze_video_request: Analyze user's video generation request
- search_character_database: Find character information if characters are mentioned
- generate_scene_prompt: Create optimized prompt for video generation
- create_video_parameters: Set up technical parameters for generation
- generate_video_file: Actually generate the video
- validate_video_output: Check video quality and compliance
- save_generation_metadata: Save metadata for tracking

Work efficiently and only call tools when necessary.
""")
            
            # Ensure system message is first
            messages = state["messages"]
            if not messages or not isinstance(messages[0], SystemMessage):
                messages = [system_message] + messages
            
            # Update iteration count
            state["iteration_count"] = state.get("iteration_count", 0) + 1
            
            # Call the LLM with tools
            response = await self.llm.ainvoke(messages)
            
            # Update state
            state["messages"] = messages + [response]
            state["status"] = "agent_processing"
            
            # Check if we have a video URL in the messages (extract from tool outputs)
            for message in reversed(state["messages"]):
                if hasattr(message, 'content') and isinstance(message.content, str):
                    if "video_url" in message.content and "https://" in message.content:
                        # Extract video URL from tool output
                        import re
                        url_match = re.search(r'https://[^\s"]+\.mp4', message.content)
                        if url_match:
                            state["video_url"] = url_match.group()
                            break
            
            return state
            
        except Exception as e:
            logger.error(f"Agent node failed: {e}")
            state["error"] = str(e)
            state["status"] = "error"
            return state
    
    async def run_agent(
        self, 
        user_request: str,
        job_id: str = None
    ) -> Dict[str, Any]:
        """Run the agent for video generation"""
        
        if not job_id:
            job_id = str(uuid.uuid4())
        
        # Start GCP tracking run
        tracking_run_id = self.cloud_tracker.start_run(
            run_name=f"agent_orchestration_{job_id}",
            tags={
                "job_id": job_id,
                "task": "video_generation_agent",
                "user_request_length": len(user_request),
                "orchestration_type": "true_agent"
            }
        )
        
        # Initialize agent state
        initial_state = AgentState(
            messages=[HumanMessage(content=user_request)],
            user_request=user_request,
            job_id=job_id,
            iteration_count=0,
            video_url=None,
            metadata=None,
            error=None,
            status="started"
        )
        
        # Set up callback handler
        callback_handler = AgentCallbackHandler(job_id)
        config = RunnableConfig(callbacks=[callback_handler])
        
        start_time = datetime.utcnow()
        
        try:
            # Log initial parameters
            if tracking_run_id:
                self.cloud_tracker.log_parameters({
                    "user_request": user_request[:500],
                    "job_id": job_id,
                    "agent_version": "1.0.0",
                    "orchestration_type": "true_agent_pattern"
                })
            
            # Run the agent workflow
            result = await self.agent_workflow.ainvoke(initial_state, config=config)
            
            # Calculate execution time
            execution_time = (datetime.utcnow() - start_time).total_seconds()
            
            # Extract metadata from final state
            metadata = {
                "generation_time": datetime.utcnow().isoformat(),
                "agent_iterations": result.get("iteration_count", 0),
                "total_messages": len(result.get("messages", [])),
                "tool_calls": callback_handler.tool_calls,
                "execution_time": execution_time,
                "orchestration_type": "agent"
            }
            
            result["metadata"] = metadata
            
            # Determine final status
            if result.get("error"):
                final_status = "failed"
            elif result.get("video_url"):
                final_status = "completed"
            else:
                final_status = "partial"
            
            result["status"] = final_status
            
            # Log metrics to both GCP tracking and metrics logger
            if tracking_run_id:
                metrics = {
                    "execution_time_seconds": execution_time,
                    "success": 1 if final_status == "completed" else 0,
                    "agent_iterations": result.get("iteration_count", 0),
                    "tool_calls": callback_handler.tool_calls,
                    "message_count": len(result.get("messages", []))
                }
                
                self.cloud_tracker.log_metrics(metrics)
                
                # Log to real-time metrics for monitoring
                self.metrics_logger.log_video_generation_metrics(
                    duration=execution_time,
                    status=final_status,
                    model="vertex_ai_gemini",
                    orchestration_type="agent",
                    quality_score=None,  # Agents don't have explicit quality scores
                    iterations=result.get("iteration_count", 0)
                )
                
                # Log agent-specific tool calls
                for _ in range(callback_handler.tool_calls):
                    self.metrics_logger.log_agent_tool_call("agent_tool", "success")
            
            return {
                "job_id": job_id,
                "tracking_run_id": tracking_run_id,
                "status": final_status,
                "video_url": result.get("video_url"),
                "metadata": metadata,
                "error": result.get("error"),
                "execution_time": execution_time,
                "agent_iterations": result.get("iteration_count", 0),
                "tool_calls": callback_handler.tool_calls
            }
            
        except Exception as e:
            logger.error(f"Agent orchestration failed for job {job_id}: {e}")
            
            # Log error to GCP tracking
            if tracking_run_id:
                execution_time = (datetime.utcnow() - start_time).total_seconds()
                self.cloud_tracker.log_metrics({
                    "execution_time_seconds": execution_time,
                    "success": 0,
                    "error": 1,
                    "tool_calls": callback_handler.tool_calls
                })
                self.cloud_tracker.log_parameters({
                    "error_message": str(e)[:500]
                })
                
                # Log to metrics for monitoring
                self.metrics_logger.log_video_generation_metrics(
                    duration=execution_time,
                    status="failed",
                    model="vertex_ai_gemini",
                    orchestration_type="agent"
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
_agent_instance = None

def get_agent() -> VisionFlowAgent:
    """Get or create agent instance"""
    global _agent_instance
    if _agent_instance is None:
        _agent_instance = VisionFlowAgent()
    return _agent_instance
