"""Real WAN 2.1 Video Generation Service using HuggingFace Diffusers."""

import asyncio
import gc
import os
import tempfile
import torch
import psutil
from pathlib import Path
from typing import Dict, Any, Optional, Union
from dataclasses import dataclass

from huggingface_hub import login
from diffusers import WanPipeline, AutoencoderKLWan
from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler
from diffusers.utils import export_to_video

from ...shared.models import VideoGenerationRequest, VideoQuality
from ...shared.monitoring import get_logger
from .resource_config import get_resource_limits, ResourceLimits
from .video_metadata_tracker import metadata_tracker

logger = get_logger(__name__)

@dataclass
class WanModelConfig:
    """Configuration for WAN model variants."""
    model_id: str
    flow_shift: float
    recommended_vram_gb: int
    max_resolution: tuple
    description: str

# Available WAN 2.1 models
WAN_MODELS = {
    "wan2.1-t2v-1.3b": WanModelConfig(
        model_id="Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
        flow_shift=3.0,
        recommended_vram_gb=8,
        max_resolution=(832, 480),
        description="Lightweight model for 480P generation"
    ),
    "wan2.1-t2v-14b": WanModelConfig(
        model_id="Wan-AI/Wan2.1-T2V-14B-Diffusers", 
        flow_shift=5.0,
        recommended_vram_gb=15,
        max_resolution=(1280, 720),
        description="High-quality model for 720P generation"
    )
}

class WanVideoGenerationService:
    """Real WAN 2.1 Video Generation Service."""
    
    def __init__(self, resource_limits: Optional[ResourceLimits] = None):
        self.resource_limits = resource_limits or get_resource_limits()
        self.device = self._detect_device()
        self.current_model = None
        self.pipeline = None
        self.model_config = None
        self.generation_count = 0
        self._authenticate_huggingface()
        self._configure_resource_limits()
        logger.info(f"WAN Video Generation Service initialized on device: {self.device}")
        logger.info(f"Resource limits: GPU memory fraction={self.resource_limits.gpu_memory_fraction}, Max RAM={self.resource_limits.max_system_ram_gb}GB")
    
    def _configure_resource_limits(self):
        """Configure resource limits to prevent system crashes."""
        if self.device == "cuda" and torch.cuda.is_available():
            # Set GPU memory fraction limit
            torch.cuda.set_per_process_memory_fraction(self.resource_limits.gpu_memory_fraction)
            logger.info(f"🔒 GPU memory limited to {self.resource_limits.gpu_memory_fraction * 100:.1f}% of available VRAM")
            
            # Get total GPU memory and set hard limit
            total_gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            max_gpu_memory_gb = total_gpu_memory * self.resource_limits.gpu_memory_fraction
            logger.info(f"📊 Total GPU memory: {total_gpu_memory:.1f}GB, Max allowed: {max_gpu_memory_gb:.1f}GB")
        elif self.device == "mps":
            # Apple Silicon MPS uses unified memory - need to be more conservative
            logger.info(f"🍎 Apple Silicon MPS detected - using conservative memory limits")
            logger.info(f"🔒 MPS memory fraction: {self.resource_limits.mps_memory_fraction * 100:.1f}%")
            
            # On Apple Silicon, GPU and system RAM are unified, so we need to be extra careful
            total_system_memory = psutil.virtual_memory().total / (1024**3)
            logger.info(f"📊 Unified memory: {total_system_memory:.1f}GB (shared between CPU and GPU)")
    
    def _check_system_resources(self) -> bool:
        """Check if system has enough resources before processing."""
        memory_info = self._get_memory_info()
        
        # Check system RAM
        if memory_info["system_ram_percent"] > self.resource_limits.system_ram_warning_threshold:
            logger.warning(f"⚠️ System RAM usage high: {memory_info['system_ram_percent']:.1f}%")
            return False
        
        # Check GPU memory if using CUDA
        if self.device == "cuda" and torch.cuda.is_available():
            gpu_memory_used_gb = memory_info.get("gpu_memory_allocated_gb", 0)
            total_gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            gpu_memory_percent = (gpu_memory_used_gb / total_gpu_memory) * 100
            
            if gpu_memory_percent > (self.resource_limits.gpu_memory_fraction * 100):
                logger.warning(f"⚠️ GPU memory usage high: {gpu_memory_percent:.1f}%")
                return False
        
        return True
    
    def _validate_request(self, request: VideoGenerationRequest) -> bool:
        """Validate video generation request against resource limits."""
        # Check video duration
        if request.duration > self.resource_limits.max_video_duration:
            logger.warning(f"⚠️ Video duration {request.duration}s exceeds limit {self.resource_limits.max_video_duration}s")
            return False
        
        # Check resolution
        width, height = map(int, request.resolution.split('x'))
        if width * height > self.resource_limits.max_resolution_pixels:
            logger.warning(f"⚠️ Resolution {width}x{height} exceeds pixel limit {self.resource_limits.max_resolution_pixels}")
            return False
        
        return True
    
    def _authenticate_huggingface(self):
        """Authenticate with HuggingFace Hub."""
        token = os.getenv('HUGGINGFACE_TOKEN')
        if not token:
            logger.warning("HUGGINGFACE_TOKEN not found. Some models may not be accessible.")
            return
            
        try:
            login(token, add_to_git_credential=False)
            logger.info("✅ HuggingFace authentication successful")
        except Exception as e:
            logger.error(f"❌ HuggingFace authentication failed: {e}")
    
    def _detect_device(self) -> str:
        """Detect the best available device for inference."""
        if torch.cuda.is_available():
            device = "cuda"
            vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            logger.info(f"CUDA detected with {vram_gb:.1f}GB VRAM")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = "mps" 
            logger.info("Apple Silicon MPS detected")
        else:
            device = "cpu"
            logger.info("Using CPU (slow but compatible)")
        
        return device
    
    def _get_memory_info(self) -> Dict[str, float]:
        """Get current memory usage information."""
        memory_info = {
            "system_ram_used_gb": psutil.virtual_memory().used / (1024**3),
            "system_ram_percent": psutil.virtual_memory().percent,
        }
        
        if self.device == "cuda" and torch.cuda.is_available():
            memory_info.update({
                "gpu_memory_allocated_gb": torch.cuda.memory_allocated() / (1024**3),
                "gpu_memory_reserved_gb": torch.cuda.memory_reserved() / (1024**3),
                "gpu_memory_percent": (torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated()) * 100
            })
        elif self.device == "mps":
            # For MPS, we primarily monitor system RAM since it's unified memory
            memory_info.update({
                "device_type": "mps",
                "unified_memory": True,
                "mps_memory_pressure": memory_info["system_ram_percent"]  # Unified memory pressure
            })
        
        return memory_info
    
    def _select_model(self, quality: VideoQuality, resolution: str) -> str:
        """Select the appropriate WAN model based on quality and resolution."""
        width, height = map(int, resolution.split('x'))
        total_pixels = width * height
        
        # Model selection logic
        if quality in [VideoQuality.LOW, VideoQuality.MEDIUM] or total_pixels <= 832 * 480:
            return "wan2.1-t2v-1.3b"
        else:
            return "wan2.1-t2v-14b"
    
    async def _load_model(self, model_key: str):
        """Load WAN model if not already loaded."""
        if self.current_model == model_key and self.pipeline is not None:
            logger.info(f"Model {model_key} already loaded")
            return
        
        # Check system resources before loading
        if not self._check_system_resources():
            raise RuntimeError("Insufficient system resources to load model")
        
        # Clear previous model if model swapping is enabled
        if self.pipeline is not None and self.resource_limits.enable_model_swapping:
            logger.info("Clearing previous model from memory")
            del self.pipeline
            self.pipeline = None
            if self.device == "cuda":
                torch.cuda.empty_cache()
            gc.collect()
        
        self.model_config = WAN_MODELS[model_key]
        logger.info(f"Loading WAN model: {self.model_config.model_id}")
        
        try:
            # Get cache directory from environment
            cache_dir = os.getenv('HF_HOME', os.getenv('TRANSFORMERS_CACHE'))
            logger.info(f"🔥 Using cache directory: {cache_dir}")
            
            # Load VAE with explicit cache directory
            vae = AutoencoderKLWan.from_pretrained(
                self.model_config.model_id, 
                subfolder="vae", 
                torch_dtype=torch.float32,
                cache_dir=cache_dir,
                local_files_only=False  # Allow cache fallback
            )
            
            # Configure scheduler
            scheduler = UniPCMultistepScheduler(
                prediction_type='flow_prediction',
                use_flow_sigmas=True, 
                num_train_timesteps=1000,
                flow_shift=self.model_config.flow_shift
            )
            
            # Load pipeline with explicit cache directory
            self.pipeline = WanPipeline.from_pretrained(
                self.model_config.model_id,
                vae=vae,
                torch_dtype=torch.bfloat16 if self.device != "cpu" else torch.float32,
                cache_dir=cache_dir,
                local_files_only=False  # Allow cache fallback
            )
            self.pipeline.scheduler = scheduler
            self.pipeline.to(self.device)
            
            # Enable memory efficient attention if available
            if hasattr(self.pipeline, 'enable_xformers_memory_efficient_attention'):
                try:
                    self.pipeline.enable_xformers_memory_efficient_attention()
                    logger.info("✅ xFormers memory efficient attention enabled")
                except Exception as e:
                    logger.warning(f"Could not enable xFormers: {e}")
            
            # Enable CPU offload for memory optimization
            if self.device == "cuda":
                try:
                    self.pipeline.enable_model_cpu_offload()
                    logger.info("✅ CPU offload enabled")
                except Exception as e:
                    logger.warning(f"Could not enable CPU offload: {e}")
            
            self.current_model = model_key
            
            memory_info = self._get_memory_info()
            logger.info(f"✅ Model loaded successfully. Memory usage: {memory_info}")
            
        except Exception as e:
            logger.error(f"❌ Failed to load model {self.model_config.model_id}: {e}")
            raise
    
    def _force_cleanup(self):
        """Force aggressive memory cleanup."""
        if self.resource_limits.enable_aggressive_cleanup:
            logger.info("🧹 Performing aggressive memory cleanup")
            gc.collect()
            
            if self.device == "cuda" and torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            elif self.device == "mps":
                # For MPS, we rely on system-level memory management
                # Force garbage collection is more important for unified memory
                logger.info("🍎 MPS memory cleanup - relying on unified memory management")
                import threading
                import time
                
                # Give the system a moment to release memory
                time.sleep(0.1)
            
            # Force Python garbage collection multiple times
            for _ in range(3):
                gc.collect()
    
    async def generate_video(self, request: VideoGenerationRequest) -> Dict[str, Any]:
        """Generate video using WAN 2.1 models."""
        logger.info(f"🎬 Starting video generation: '{request.prompt[:50]}...'")
        
        try:
            # Validate request against resource limits
            if not self._validate_request(request):
                return {
                    "status": "failed",
                    "error": "Request exceeds resource limits",
                    "memory_usage": self._get_memory_info()
                }
            
            # Check system resources before starting
            if not self._check_system_resources():
                return {
                    "status": "failed",
                    "error": "Insufficient system resources",
                    "memory_usage": self._get_memory_info()
                }
            
            # Select and load appropriate model
            model_key = self._select_model(request.quality, request.resolution)
            await self._load_model(model_key)
            
            # Parse resolution
            width, height = map(int, request.resolution.split('x'))
            
            # Calculate number of frames (assume 24 FPS)
            num_frames = int(request.duration * request.fps) + 1  # +1 for proper frame count
            
            # Prepare generation parameters
            generation_params = {
                "prompt": request.prompt,
                "negative_prompt": self._get_negative_prompt(),
                "height": height,
                "width": width, 
                "num_frames": num_frames,
                "guidance_scale": request.guidance_scale,
                "num_inference_steps": request.num_inference_steps,
            }
            
            if request.seed is not None:
                generator = torch.Generator(device=self.device).manual_seed(request.seed)
                generation_params["generator"] = generator
            
            logger.info(f"Generation parameters: {generation_params}")
            
            # Generate video
            logger.info("🚀 Running inference...")
            result = self.pipeline(**generation_params)
            video_frames = result.frames[0]
            
            # Export video to file
            output_path = self._create_output_path()
            export_to_video(video_frames, str(output_path), fps=request.fps)
            
            # Create generation result for metadata tracking
            generation_result = {
                "status": "completed",
                "video_path": str(output_path),
                "model_used": self.model_config.model_id,
                "resolution": f"{width}x{height}",
                "duration": request.duration,
                "fps": request.fps,
                "num_frames": num_frames,
                "memory_usage": final_memory,
                "generation_count": self.generation_count
            }
            
            # Track metadata for this generation
            try:
                model_info = {
                    "model_name": self.model_config.model_id,
                    "model_version": None,  # Could be extracted from model config
                    "device": self.device
                }
                
                generation_id = await metadata_tracker.track_video_generation(
                    video_path=str(output_path),
                    request=request,
                    generation_result=generation_result,
                    generation_time=generation_time,
                    model_info=model_info
                )
                
                generation_result["generation_id"] = generation_id
                logger.info(f"📋 Metadata tracked with ID: {generation_id}")
                
            except Exception as e:
                logger.warning(f"⚠️ Failed to track metadata: {e}")
                # Continue without metadata - don't fail the generation
            
            # Increment generation count and cleanup if needed
            self.generation_count += 1
            if self.generation_count % self.resource_limits.cleanup_interval_generations == 0:
                self._force_cleanup()
            
            # Get final memory info
            final_memory = self._get_memory_info()
            
            logger.info(f"✅ Video generation completed: {output_path}")
            
            return generation_result
            
        except Exception as e:
            logger.error(f"❌ Video generation failed: {e}")
            return {
                "status": "failed",
                "error": str(e),
                "memory_usage": self._get_memory_info()
            }
    
    def _get_negative_prompt(self) -> str:
        """Get standard negative prompt for better quality."""
        return ("Bright tones, overexposed, static, blurred details, subtitles, style, works, "
                "paintings, images, static, overall gray, worst quality, low quality, JPEG compression "
                "residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, "
                "deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, "
                "three legs, many people in the background, walking backwards")
    
    def _create_output_path(self) -> Path:
        """Create output path for generated video."""
        # Use host path instead of container path
        output_dir = Path("/Users/dav/coding/wan-open-eval/visionflow/generated")
        output_dir.mkdir(exist_ok=True)
        
        # Create unique filename
        import uuid
        filename = f"wan_video_{uuid.uuid4().hex[:8]}.mp4"
        return output_dir / filename
    
    async def get_model_status(self) -> Dict[str, Any]:
        """Get current model status and system information."""
        return {
            "current_model": self.current_model,
            "model_config": self.model_config.__dict__ if self.model_config else None,
            "device": self.device,
            "memory_usage": self._get_memory_info(),
            "available_models": list(WAN_MODELS.keys()),
            "resource_limits": {
                "gpu_memory_fraction": self.resource_limits.gpu_memory_fraction,
                "max_system_ram_gb": self.resource_limits.max_system_ram_gb,
                "max_video_duration": self.resource_limits.max_video_duration,
                "max_resolution_pixels": self.resource_limits.max_resolution_pixels,
                "enable_aggressive_cleanup": self.resource_limits.enable_aggressive_cleanup
            },
            "generation_count": self.generation_count
        }
    
    async def cleanup(self):
        """Clean up resources and free memory."""
        if self.pipeline is not None:
            logger.info("🧹 Cleaning up WAN pipeline")
            del self.pipeline
            self.pipeline = None
            self.current_model = None
            self.model_config = None
            
            if self.device == "cuda":
                torch.cuda.empty_cache()
            gc.collect()
            
            logger.info("✅ Cleanup completed")

# Global service instance with resource limits from configuration
wan_service = WanVideoGenerationService()
