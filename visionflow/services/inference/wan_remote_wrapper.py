#!/usr/bin/env python3
"""
WAN 2.1 Remote Execution Wrapper
Keeps your exact WAN model code, just delegates execution to remote GPUs
"""

import asyncio
import os
import json
import time
import uuid
from typing import Dict, Any, Optional, Union
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class ExecutionBackend(Enum):
    LOCAL = "local"
    RAY = "ray"
    MODAL = "modal"
    RUNPOD = "runpod"
    VAST_AI = "vast_ai"

@dataclass
class RemoteJobRequest:
    """Request for remote WAN 2.1 execution"""
    request_id: str
    wan_request_dict: Dict[str, Any]  # Your existing VideoGenerationRequest serialized
    execution_config: Dict[str, Any] = None

@dataclass 
class RemoteJobResponse:
    """Response from remote WAN 2.1 execution"""
    request_id: str
    status: str  # pending, running, completed, failed
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    execution_time: Optional[float] = None
    remote_logs: Optional[str] = None

class WANRemoteExecutor:
    """
    Wraps your existing WAN 2.1 service for remote execution
    NO CHANGES to your model code - just execution delegation
    """
    
    def __init__(self, backend: ExecutionBackend = ExecutionBackend.LOCAL):
        self.backend = backend
        self.jobs = {}  # Job tracking
        self._init_backend()
        
        # Import your existing WAN service locally for fallback
        try:
            from visionflow.services.generation.wan_video_service import WanVideoGenerationService
            self.local_wan_service = WanVideoGenerationService()
            logger.info("✅ Local WAN service loaded for fallback")
        except ImportError as e:
            logger.warning(f"Could not load local WAN service: {e}")
            self.local_wan_service = None
    
    def _init_backend(self):
        """Initialize the selected backend"""
        if self.backend == ExecutionBackend.RAY:
            self._init_ray()
        elif self.backend == ExecutionBackend.MODAL:
            self._init_modal()
        elif self.backend == ExecutionBackend.RUNPOD:
            self._init_runpod()
        elif self.backend == ExecutionBackend.VAST_AI:
            self._init_vast_ai()
        # LOCAL needs no initialization
    
    def _init_ray(self):
        """Initialize Ray backend with your WAN code"""
        try:
            import ray
            
            if not ray.is_initialized():
                ray.init(address=os.getenv('RAY_ADDRESS', 'auto'))
            
            # Define remote WAN worker with YOUR code
            @ray.remote(num_gpus=1, memory=16*1024*1024*1024)  # 16GB RAM
            class WANWorker:
                def __init__(self):
                    # Import and initialize YOUR WAN service on the worker
                    import sys
                    sys.path.append('/app')  # Adjust path as needed
                    
                    from visionflow.services.generation.wan_video_service import WanVideoGenerationService
                    from visionflow.shared.models import VideoGenerationRequest
                    
                    self.wan_service = WanVideoGenerationService()
                    logger.info("WAN 2.1 service initialized on Ray worker")
                
                async def generate_video_remote(self, request_dict: dict) -> dict:
                    """Run YOUR exact WAN generation code remotely"""
                    try:
                        # Reconstruct your VideoGenerationRequest
                        from visionflow.shared.models import VideoGenerationRequest
                        
                        request = VideoGenerationRequest(**request_dict)
                        
                        # Call YOUR existing generate_video method
                        result = await self.wan_service.generate_video(request)
                        
                        return {
                            "status": "completed",
                            "result": result,
                            "execution_time": result.get("generation_time"),
                        }
                        
                    except Exception as e:
                        logger.error(f"Remote WAN generation failed: {e}")
                        return {
                            "status": "failed",
                            "error": str(e)
                        }
            
            # Create worker pool
            self.ray_workers = [WANWorker.remote() for _ in range(2)]
            self.worker_index = 0
            
        except ImportError:
            logger.error("Ray not available. Install with: pip install ray")
            raise
    
    def _init_modal(self):
        """Initialize Modal backend with your WAN code"""
        try:
            import modal
            
            # Create Modal app that packages YOUR WAN code
            self.app = modal.App("wan-21-remote")
            
            # Define the Modal function with your dependencies
            @self.app.function(
                gpu="A100",  # or "H100" for 14B model
                timeout=3600,
                memory=16384,  # 16GB
                # Mount your WAN service code
                mounts=[
                    modal.Mount.from_local_dir("./visionflow", remote_path="/app/visionflow")
                ],
                # Your Python dependencies
                pip=[
                    "torch>=2.0.0",
                    "diffusers>=0.21.0", 
                    "transformers>=4.25.0",
                    "accelerate>=0.24.0",
                    "huggingface_hub>=0.19.0",
                    # Add other WAN dependencies here
                ]
            )
            async def generate_video_modal(request_dict: dict):
                """Run YOUR WAN code on Modal"""
                import sys
                sys.path.append('/app')
                
                # Import YOUR services
                from visionflow.services.generation.wan_video_service import WanVideoGenerationService
                from visionflow.shared.models import VideoGenerationRequest
                
                try:
                    # Initialize YOUR WAN service
                    wan_service = WanVideoGenerationService()
                    
                    # Create request object
                    request = VideoGenerationRequest(**request_dict)
                    
                    # Call YOUR method
                    result = await wan_service.generate_video(request)
                    
                    return {
                        "status": "completed", 
                        "result": result,
                        "execution_time": result.get("generation_time")
                    }
                    
                except Exception as e:
                    return {
                        "status": "failed",
                        "error": str(e)
                    }
            
            self.modal_function = generate_video_modal
            
        except ImportError:
            logger.error("Modal not available. Install with: pip install modal")
            raise
    
    def _init_runpod(self):
        """Initialize RunPod backend"""
        try:
            import runpod
            
            # Set API key
            runpod.api_key = os.getenv('RUNPOD_API_KEY')
            
            # Create endpoint with your WAN code
            # This would deploy your WAN service as a RunPod endpoint
            self.runpod_endpoint_id = os.getenv('RUNPOD_ENDPOINT_ID')
            
        except ImportError:
            logger.error("RunPod not available. Install with: pip install runpod")
            raise
    
    def _init_vast_ai(self):
        """Initialize Vast.ai backend"""
        # Vast.ai integration would go here
        # Using their API to spin up instances with your WAN code
        pass
    
    async def generate_video_remote(self, request, **kwargs) -> Dict[str, Any]:
        """
        Generate video remotely using your exact WAN 2.1 code
        
        Args:
            request: Your existing VideoGenerationRequest object
            **kwargs: Additional parameters
            
        Returns:
            Same format as your local WAN service
        """
        
        # Fallback to local if remote not available
        if self.backend == ExecutionBackend.LOCAL or not self._is_backend_available():
            if self.local_wan_service:
                logger.info("Using local WAN 2.1 execution")
                return await self.local_wan_service.generate_video(request)
            else:
                raise RuntimeError("No execution backend available")
        
        # Serialize request for remote execution
        request_dict = self._serialize_request(request)
        
        job_request = RemoteJobRequest(
            request_id=str(uuid.uuid4()),
            wan_request_dict=request_dict,
            execution_config=kwargs
        )
        
        # Submit to remote backend
        job_id = await self._submit_remote_job(job_request)
        
        # Wait for completion (with polling)
        return await self._wait_for_completion(job_id)
    
    def _serialize_request(self, request) -> Dict[str, Any]:
        """Convert your VideoGenerationRequest to dict for remote transmission"""
        if hasattr(request, '__dict__'):
            return request.__dict__
        elif hasattr(request, '_asdict'):
            return request._asdict()
        else:
            # Handle dataclass
            return asdict(request)
    
    async def _submit_remote_job(self, job_request: RemoteJobRequest) -> str:
        """Submit job to selected backend"""
        
        if self.backend == ExecutionBackend.RAY:
            return await self._submit_ray_job(job_request)
        elif self.backend == ExecutionBackend.MODAL:
            return await self._submit_modal_job(job_request)
        elif self.backend == ExecutionBackend.RUNPOD:
            return await self._submit_runpod_job(job_request)
        elif self.backend == ExecutionBackend.VAST_AI:
            return await self._submit_vast_ai_job(job_request)
        else:
            raise ValueError(f"Unsupported backend: {self.backend}")
    
    async def _submit_ray_job(self, job_request: RemoteJobRequest) -> str:
        """Submit to Ray worker"""
        # Get next worker (round-robin)
        worker = self.ray_workers[self.worker_index]
        self.worker_index = (self.worker_index + 1) % len(self.ray_workers)
        
        # Submit job
        future = worker.generate_video_remote.remote(job_request.wan_request_dict)
        
        job_id = job_request.request_id
        self.jobs[job_id] = {
            "future": future,
            "status": "running",
            "submitted_at": time.time()
        }
        
        return job_id
    
    async def _submit_modal_job(self, job_request: RemoteJobRequest) -> str:
        """Submit to Modal"""
        future = self.modal_function.spawn(job_request.wan_request_dict)
        
        job_id = job_request.request_id
        self.jobs[job_id] = {
            "future": future,
            "status": "running", 
            "submitted_at": time.time()
        }
        
        return job_id
    
    async def _submit_runpod_job(self, job_request: RemoteJobRequest) -> str:
        """Submit to RunPod"""
        import runpod
        
        # Submit via RunPod API
        response = runpod.run_sync(
            endpoint_id=self.runpod_endpoint_id,
            input=job_request.wan_request_dict
        )
        
        job_id = response.get('id', job_request.request_id)
        self.jobs[job_id] = {
            "runpod_id": response['id'],
            "status": "running",
            "submitted_at": time.time()
        }
        
        return job_id
    
    async def _submit_vast_ai_job(self, job_request: RemoteJobRequest) -> str:
        """Submit to Vast.ai"""
        # Implementation would use Vast.ai API
        raise NotImplementedError("Vast.ai backend not implemented yet")
    
    async def _wait_for_completion(self, job_id: str) -> Dict[str, Any]:
        """Wait for remote job completion"""
        job_data = self.jobs[job_id]
        
        while True:
            status = await self._check_job_status(job_id)
            
            if status["status"] in ["completed", "failed"]:
                if status["status"] == "completed":
                    return status["result"]
                else:
                    raise RuntimeError(f"Remote generation failed: {status.get('error')}")
            
            await asyncio.sleep(2)  # Poll every 2 seconds
    
    async def _check_job_status(self, job_id: str) -> Dict[str, Any]:
        """Check status of remote job"""
        job_data = self.jobs.get(job_id)
        if not job_data:
            return {"status": "not_found"}
        
        if self.backend == ExecutionBackend.RAY:
            import ray
            future = job_data["future"]
            ready, _ = ray.wait([future], timeout=0)
            
            if ready:
                result = ray.get(ready[0])
                job_data["status"] = result["status"]
                job_data["result"] = result.get("result")
                job_data["error"] = result.get("error")
                return job_data
            else:
                return {"status": "running"}
        
        elif self.backend == ExecutionBackend.MODAL:
            future = job_data["future"]
            try:
                if future.is_finished():
                    result = future.get()
                    job_data["status"] = result["status"]
                    job_data["result"] = result.get("result")
                    job_data["error"] = result.get("error")
                    return job_data
                else:
                    return {"status": "running"}
            except Exception as e:
                return {"status": "failed", "error": str(e)}
        
        elif self.backend == ExecutionBackend.RUNPOD:
            import runpod
            status = runpod.get_job(job_data["runpod_id"])
            
            if status["status"] == "COMPLETED":
                return {
                    "status": "completed",
                    "result": status["output"]
                }
            elif status["status"] == "FAILED":
                return {
                    "status": "failed", 
                    "error": status.get("error")
                }
            else:
                return {"status": "running"}
        
        return {"status": "unknown"}
    
    def _is_backend_available(self) -> bool:
        """Check if remote backend is available"""
        if self.backend == ExecutionBackend.RAY:
            try:
                import ray
                return ray.is_initialized()
            except:
                return False
        elif self.backend == ExecutionBackend.MODAL:
            return hasattr(self, 'modal_function')
        elif self.backend == ExecutionBackend.RUNPOD:
            return hasattr(self, 'runpod_endpoint_id') and self.runpod_endpoint_id
        else:
            return False

# Factory function for easy integration
def create_wan_remote_service(backend: str = "local", **config) -> WANRemoteExecutor:
    """
    Create WAN remote service with specified backend
    
    Usage:
        # Local execution (your existing code)
        wan_service = create_wan_remote_service("local")
        
        # Ray distributed execution
        wan_service = create_wan_remote_service("ray")
        
        # Modal serverless execution 
        wan_service = create_wan_remote_service("modal")
        
        # Use exactly like your existing service:
        result = await wan_service.generate_video_remote(request)
    """
    backend_enum = ExecutionBackend(backend.lower())
    return WANRemoteExecutor(backend_enum)

# Drop-in replacement for your existing service
class DistributedWANService:
    """
    Drop-in replacement for WanVideoGenerationService
    Automatically chooses best execution backend
    """
    
    def __init__(self, prefer_remote: bool = False, fallback_local: bool = True):
        self.prefer_remote = prefer_remote
        self.fallback_local = fallback_local
        
        # Try to initialize remote backends in order of preference
        self.remote_executor = None
        
        if prefer_remote:
            for backend in [ExecutionBackend.RAY, ExecutionBackend.MODAL, ExecutionBackend.RUNPOD]:
                try:
                    self.remote_executor = WANRemoteExecutor(backend)
                    if self.remote_executor._is_backend_available():
                        logger.info(f"Using {backend.value} for WAN 2.1 execution")
                        break
                except Exception as e:
                    logger.warning(f"Could not initialize {backend.value}: {e}")
                    continue
        
        # Fallback to local
        if not self.remote_executor or not self.remote_executor._is_backend_available():
            if fallback_local:
                self.remote_executor = WANRemoteExecutor(ExecutionBackend.LOCAL)
                logger.info("Using local WAN 2.1 execution")
            else:
                raise RuntimeError("No execution backend available")
    
    async def generate_video(self, request) -> Dict[str, Any]:
        """
        Same interface as your existing WanVideoGenerationService.generate_video()
        """
        return await self.remote_executor.generate_video_remote(request)

# Example usage - minimal changes to your existing code
async def example_migration():
    """Example of how to migrate your existing code"""
    
    # BEFORE: Your existing code
    # from visionflow.services.generation.wan_video_service import WanVideoGenerationService
    # wan_service = WanVideoGenerationService()
    # result = await wan_service.generate_video(request)
    
    # AFTER: Just change the import and initialization
    wan_service = DistributedWANService(prefer_remote=True)
    # Everything else stays the same!
    # result = await wan_service.generate_video(request)
    
    print("✅ Migrated to distributed WAN 2.1 with zero code changes!")

if __name__ == "__main__":
    asyncio.run(example_migration())
