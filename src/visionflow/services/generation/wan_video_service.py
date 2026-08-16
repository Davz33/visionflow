"""Wan2.2 video generation via HuggingFace Diffusers (Wan2.1 kept as legacy)."""

import gc
import os
import time
import json
import uuid
import shutil
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass

import torch
import psutil
from huggingface_hub import login
from diffusers import WanPipeline, AutoencoderKLWan
from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler
from diffusers.utils import export_to_video

from ...shared.models import VideoGenerationRequest, VideoQuality
from ...shared.monitoring import get_logger
from .resource_config import get_resource_limits, ResourceLimits
from .video_metadata_tracker import metadata_tracker
from .m4_optimizations import get_m4_optimizer, is_m4_available
from .m4_memory_manager import get_m4_memory_manager
from .wan_generate_spec import WanGenerateRequest, result_identity, validate_generate_request
from .wan_model_zoo import (
    WAN_MODELS as WAN_ZOO,
    current_zoo_metadata,
    get_model,
    select_model_for_quality,
)

logger = get_logger(__name__)

@dataclass
class WanModelConfig:
    """Runtime view of a zoo entry for the existing load/generate path."""
    model_id: str
    flow_shift: float
    recommended_vram_gb: int
    max_resolution: tuple
    description: str
    key: str = ""
    task: str = ""
    sample_fps: int = 24
    sample_steps: int = 50
    hf_revision: str = "main"
    upstream_git_sha: str = ""


def _spec_to_config(key: str) -> WanModelConfig:
    spec = get_model(key)
    return WanModelConfig(
        model_id=spec.hf_id,
        flow_shift=spec.flow_shift,
        recommended_vram_gb=spec.recommended_vram_gb,
        max_resolution=spec.max_resolution,
        description=spec.description,
        key=spec.key,
        task=spec.task,
        sample_fps=spec.sample_fps,
        sample_steps=spec.sample_steps,
        hf_revision=spec.hf_revision,
        upstream_git_sha=spec.upstream_git_sha,
    )


# Back-compat name: old imports still see WAN_MODELS keys.
WAN_MODELS = {key: _spec_to_config(key) for key in WAN_ZOO}

class WanVideoGenerationService:
    """Wan2.2 Diffusers generate path (Wan2.1 keys still load)."""
    
    def __init__(self, resource_limits: Optional[ResourceLimits] = None):
        self.resource_limits = resource_limits or get_resource_limits()
        self.device = self._detect_device()
        self.current_model = None
        self.pipeline = None
        self.model_config = None
        self.generation_count = 0
        
        # Initialize M4 optimizations if available
        self.m4_optimizer = get_m4_optimizer() if is_m4_available() else None
        self.m4_memory_manager = get_m4_memory_manager() if is_m4_available() else None
        
        self._authenticate_huggingface()
        self._configure_resource_limits()
        logger.info(f"Wan video service initialized on device: {self.device}")
        logger.info(f"Resource limits: GPU memory fraction={self.resource_limits.gpu_memory_fraction}, Max RAM={self.resource_limits.max_system_ram_gb}GB")
        if self.m4_optimizer:
            logger.info("🍎 M4 optimizations enabled")
    
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
        # Use M4 unified memory manager if available
        if self.m4_memory_manager:
            return self.m4_memory_manager.get_memory_info()
        
        # Fallback to standard memory info
        memory_info = {
            "system_ram_used_gb": psutil.virtual_memory().used / (1024**3),
            "system_ram_percent": psutil.virtual_memory().percent,
        }
        
        if self.device == "cuda" and torch.cuda.is_available():
            memory_info.update({
                "gpu_memory_allocated_gb": torch.cuda.memory_allocated() / (1024**3),
                "gpu_memory_reserved_gb": torch.cuda.memory_reserved() / (1024**3),
                "gpu_memory_percent": (torch.cuda.memory_allocated() / max(torch.cuda.max_memory_allocated(), 1)) * 100
            })
        elif self.device == "mps":
            # For MPS, we primarily monitor system RAM since it's unified memory
            memory_info.update({
                "device_type": "mps",
                "unified_memory": True,
                "mps_memory_pressure": memory_info["system_ram_percent"]  # Unified memory pressure
            })
        
        return memory_info
    
    def _select_model(
        self,
        quality: VideoQuality,
        resolution: str,
        request: Optional[VideoGenerationRequest] = None,
    ) -> str:
        """Prefer an explicit zoo key / supported task; else map quality."""
        from .wan_model_zoo import SUPPORTED_SIZES, normalize_size

        if request is not None:
            if request.model_key:
                return get_model(request.model_key).key
            if request.distilled and request.task and request.task.value == "animate-2-14B":
                return "animate-2-14B-distilled"
            if request.task:
                key = get_model(request.task.value).key
                try:
                    size_key = normalize_size(resolution)
                    allowed = SUPPORTED_SIZES.get(request.task.value, ())
                    if size_key in allowed:
                        return key
                except ValueError:
                    pass
        return select_model_for_quality(quality.value, resolution)
    
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
            cache_dir_env = os.getenv('HF_HOME') or os.getenv('TRANSFORMERS_CACHE')
            if cache_dir_env is None:
                cache_dir = Path.home() / ".cache" / "huggingface"
                cache_dir.mkdir(parents=True, exist_ok=True)
            else:
                cache_dir = Path(cache_dir_env)
                cache_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Using cache directory: {cache_dir}")
            
            # Choose torch_dtype: Tesla T4 (compute 7.5) does not support native fast bfloat16 well; use float16 if supported
            if self.device == "cuda":
                capability = torch.cuda.get_device_capability(0)
                # Compute capability >= 8.0 (Ampere+) supports bfloat16 well
                weight_dtype = torch.bfloat16 if capability[0] >= 8 else torch.float16
            elif self.device == "mps":
                weight_dtype = torch.float16
            else:
                weight_dtype = torch.float32

            # Load VAE with explicit cache directory
            vae = AutoencoderKLWan.from_pretrained(
                self.model_config.model_id, 
                subfolder="vae", 
                torch_dtype=weight_dtype,
                cache_dir=str(cache_dir),
                local_files_only=False  # Allow cache fallback
            )
            
            # Configure scheduler
            scheduler = UniPCMultistepScheduler(
                prediction_type='flow_prediction',
                use_flow_sigmas=True, 
                num_train_timesteps=1000,
                flow_shift=self.model_config.flow_shift
            )
            
            # Load pipeline with explicit cache directory and device_map="balanced" if supported
            try:
                if self.device == "cuda" and torch.cuda.get_device_properties(0).total_memory / (1024**3) < 20.0:
                    self.pipeline = WanPipeline.from_pretrained(
                        self.model_config.model_id,
                        vae=vae,
                        torch_dtype=weight_dtype,
                        cache_dir=str(cache_dir),
                        device_map="balanced",
                        local_files_only=False
                    )
                else:
                    self.pipeline = WanPipeline.from_pretrained(
                        self.model_config.model_id,
                        vae=vae,
                        torch_dtype=weight_dtype,
                        cache_dir=str(cache_dir),
                        local_files_only=False
                    )
            except (ValueError, TypeError):
                # Fallback if device_map strategy is rejected by specific pipeline class
                self.pipeline = WanPipeline.from_pretrained(
                    self.model_config.model_id,
                    vae=vae,
                    torch_dtype=weight_dtype,
                    cache_dir=str(cache_dir),
                    local_files_only=False
                )
            self.pipeline.scheduler = scheduler

            # Offload and memory placement strategy adapted to available VRAM
            if self.device == "cuda":
                total_gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                free_gpu_memory = torch.cuda.mem_get_info()[0] / (1024**3) if torch.cuda.is_available() else total_gpu_memory
                logger.info(f"📊 CUDA Memory: total={total_gpu_memory:.1f}GB, free={free_gpu_memory:.1f}GB")
                
                # Adaptive tiers based on available free VRAM and total VRAM
                if free_gpu_memory < 16.0 or total_gpu_memory < 20.0:
                    try:
                        if free_gpu_memory < 8.0:
                            # Severe constraint: sequential CPU offload (submodule by submodule during forward pass)
                            if hasattr(self.pipeline, "enable_sequential_cpu_offload"):
                                self.pipeline.enable_sequential_cpu_offload()
                                logger.info("⚡ Adaptive VRAM strategy: Sequential CPU offload enabled (<8GB free VRAM)")
                            elif hasattr(self.pipeline, "enable_model_cpu_offload"):
                                self.pipeline.enable_model_cpu_offload()
                                logger.info("⚡ Adaptive VRAM strategy: Model CPU offload fallback enabled")
                            else:
                                self.pipeline.to(self.device)
                        else:
                            # Moderate constraint: whole-model CPU offload (offloads entire components when idle)
                            if hasattr(self.pipeline, "enable_model_cpu_offload"):
                                self.pipeline.enable_model_cpu_offload()
                                logger.info("⚡ Adaptive VRAM strategy: Model CPU offload enabled (<16GB free VRAM)")
                            else:
                                self.pipeline.to(self.device)
                    except Exception as e:
                        logger.warning(f"Could not enable CPU offload: {e}")
                        self.pipeline.to(self.device)
                else:
                    self.pipeline.to(self.device)
                    logger.info(f"🚀 High VRAM detected ({free_gpu_memory:.1f}GB free): Keeping full model on GPU")
            else:
                self.pipeline.to(self.device)
            
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
            
            # Use M4 unified memory manager if available
            if self.m4_memory_manager:
                self.m4_memory_manager.cleanup_memory(aggressive=True)
            else:
                # Fallback to standard cleanup
                gc.collect()
                
                if self.device == "cuda" and torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                elif self.device == "mps":
                    # For MPS, we rely on system-level memory management
                    # Force garbage collection is more important for unified memory
                    logger.info("🍎 MPS memory cleanup - relying on unified memory management")
                    import time
                    
                    # Give the system a moment to release memory
                    time.sleep(0.1)
                
                # Force Python garbage collection multiple times
                for _ in range(3):
                    gc.collect()
    
    async def generate_video(self, request: VideoGenerationRequest) -> Dict[str, Any]:
        """Generate video using the pinned Wan zoo (Wan2.2 by default)."""
        logger.info(f"🎬 Starting video generation: '{request.prompt[:50]}...'")
        start_time = time.time()
        
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
            
            # Validate output path BEFORE starting expensive generation
            output_success, output_error, output_path = self._validate_output_path()
            if not output_success:
                logger.error(f"🚫 Pre-generation check failed: {output_error}")
                return {
                    "status": "failed",
                    "error": f"Output validation failed: {output_error}",
                    "memory_usage": self._get_memory_info()
                }
            logger.info(f"✅ Pre-generation validation passed - ready to save to: {output_path}")
            
            # Select and load appropriate model
            model_key = self._select_model(request.quality, request.resolution, request)
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

            if getattr(request, "image_path", None):
                generation_params["image"] = request.image_path
            
            logger.info(f"Generation parameters: {generation_params}")
            
            # Generate video with M4 optimizations
            logger.info("🚀 Running inference...")
            
            # Optimize generation parameters for M4 if available
            if self.m4_optimizer:
                # Optimize any tensors in generation params
                for key, value in generation_params.items():
                    if isinstance(value, torch.Tensor):
                        generation_params[key] = self.m4_optimizer.optimize_tensor_for_m4(value)
            
            result = self.pipeline(**generation_params)
            video_frames = result.frames[0]
            
            # Optimize output frames for M4
            if self.m4_optimizer and isinstance(video_frames, torch.Tensor):
                video_frames = self.m4_optimizer.optimize_tensor_for_m4(video_frames)
            
            # Export video to file (using pre-validated path)
            export_to_video(video_frames, str(output_path), fps=request.fps)
            
            # Get final memory info and calculate generation time
            final_memory = self._get_memory_info()
            generation_time = time.time() - start_time
            
            seed = request.seed if request.seed is not None else -1
            gen_req = WanGenerateRequest(
                task=(request.task.value if request.task else self.model_config.task) or "ti2v-5B",
                size=request.resolution.replace("x", "*"),
                prompt=request.prompt,
                image=request.image_path,
                audio=request.audio_path,
                pose_video=request.pose_video_path,
                use_prompt_extend=bool(request.use_prompt_extend),
                prompt_extend_method=request.prompt_extend_method,
                sample_solver=request.sample_solver,
                sample_steps=request.num_inference_steps,
                sample_guide_scale=request.guidance_scale,
                base_seed=seed,
                distilled=bool(request.distilled),
                model_key=model_key,
            )
            try:
                gen_req = validate_generate_request(gen_req)
            except ValueError as exc:
                logger.warning(f"Generate spec warning (continuing with Diffusers path): {exc}")
            identity = result_identity(gen_req, seed if seed >= 0 else 0)
            identity.update(current_zoo_metadata())

            # Create generation result for metadata tracking
            generation_result = {
                "status": "completed",
                "video_path": str(output_path),
                "model_used": self.model_config.model_id,
                "model_revision": identity.get("model_revision"),
                "hf_revision": identity.get("hf_revision"),
                "upstream_git_sha": identity.get("upstream_git_sha"),
                "upstream_repo": identity.get("upstream_repo"),
                "seed": request.seed,
                "task": identity.get("task"),
                "model_key": model_key,
                "use_prompt_extend": bool(request.use_prompt_extend),
                "sample_solver": request.sample_solver,
                "sample_steps": request.num_inference_steps,
                "size_class": identity.get("size_class"),
                "distilled": bool(request.distilled),
                "resolution": f"{width}x{height}",
                "duration": request.duration,
                "fps": request.fps,
                "num_frames": num_frames,
                "memory_usage": final_memory,
                "generation_count": self.generation_count,
                "generation_time": generation_time,
            }
            sidecar = output_path.with_suffix(".metadata.json")
            sidecar.write_text(
                json.dumps(generation_result, indent=2, default=str) + "\n",
                encoding="utf-8",
            )
            generation_result["metadata_path"] = str(sidecar)
            
            # Track metadata for this generation
            try:
                model_info = {
                    "model_name": self.model_config.model_id,
                    "model_version": self.model_config.hf_revision,
                    "device": self.device,
                    "upstream_git_sha": self.model_config.upstream_git_sha,
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
    
    def _validate_output_path(self) -> tuple[bool, str, Path]:
        """Validate that we can write to the output directory before generation starts.
        
        Returns:
            (success: bool, error_message: str, output_path: Path)
        """
        try:
            # Ensure output directory exists and is writable
            output_dir = Path("generated")
            output_dir.mkdir(exist_ok=True)
            
            # Create unique filename
            import uuid
            filename = f"wan_video_{uuid.uuid4().hex[:8]}.mp4"
            output_path = output_dir / filename
            
            # Test write permissions by creating a temporary test file
            test_file = output_dir / f"test_write_{uuid.uuid4().hex[:4]}.tmp"
            try:
                with open(test_file, 'w') as f:
                    f.write("test")
                test_file.unlink()  # Delete test file
            except Exception as e:
                return False, f"Cannot write to output directory: {e}", output_path
            
            # Check available disk space (require at least 100MB free)
            import shutil
            try:
                free_space = shutil.disk_usage(output_dir).free
                free_space_mb = free_space / (1024 * 1024)
                if free_space_mb < 100:
                    return False, f"Insufficient disk space: {free_space_mb:.1f}MB available, need at least 100MB", output_path
            except Exception as e:
                logger.warning(f"Could not check disk space: {e}")
                # Continue anyway - disk space check is not critical
            
            # Ensure filename doesn't already exist (very unlikely with UUIDs but better safe)
            if output_path.exists():
                return False, f"Output file already exists: {output_path}", output_path
            
            logger.info(f"✅ Output validation passed: {output_path}")
            return True, "", output_path
            
        except Exception as e:
            return False, f"Output path validation failed: {e}", Path("generated") / "fallback.mp4"

    def _create_output_path(self) -> Path:
        """Create output path for generated video."""
        success, error_msg, output_path = self._validate_output_path()
        if not success:
            raise RuntimeError(f"Cannot create output path: {error_msg}")
        return output_path
    
    async def get_model_status(self) -> Dict[str, Any]:
        """Get current model status and system information."""
        return {
            "current_model": self.current_model,
            "model_config": self.model_config.__dict__ if self.model_config else None,
            "device": self.device,
            "memory_usage": self._get_memory_info(),
            "available_models": list(WAN_MODELS.keys()),
            "zoo": current_zoo_metadata(),
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
