"""
Enhanced WAN 2.1 video generation service with real model implementation
Optimized for GPU usage and memory management
"""

import asyncio
import gc
import hashlib
import os
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Tuple

import cv2
import numpy as np
import torch
import torch.cuda
from diffusers import DiffusionPipeline, AnimateDiffPipeline
from diffusers.utils import export_to_video
from PIL import Image
import psutil

from ...shared.config import get_settings
from ...shared.models import (
    PromptOptimization,
    RoutingDecision,
    VideoGenerationRequest,
    GenerationResult,
)
from ...shared.monitoring import get_logger, track_video_generation

logger = get_logger("wan_model_service")


class GPUMemoryManager:
    """Manages GPU memory allocation and cleanup"""
    
    def __init__(self):
        self.device = self._get_optimal_device()
        self.memory_threshold = 0.9  # 90% threshold
        
    def _get_optimal_device(self) -> str:
        """Determine the best available device with memory info"""
        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            best_device = 0
            max_memory = 0
            
            for i in range(device_count):
                props = torch.cuda.get_device_properties(i)
                memory = props.total_memory
                if memory > max_memory:
                    max_memory = memory
                    best_device = i
            
            logger.info(f"Selected GPU {best_device} with {max_memory / 1e9:.1f}GB memory")
            return f"cuda:{best_device}"
        elif torch.backends.mps.is_available():
            logger.info("Using Apple Silicon MPS")
            return "mps"
        else:
            logger.warning("No GPU available, falling back to CPU")
            return "cpu"
    
    def check_memory(self) -> Dict[str, float]:
        """Check current memory usage"""
        if "cuda" in self.device:
            gpu_memory = torch.cuda.get_device_properties(self.device).total_memory
            gpu_allocated = torch.cuda.memory_allocated(self.device)
            gpu_reserved = torch.cuda.memory_reserved(self.device)
            
            return {
                "gpu_total_gb": gpu_memory / 1e9,
                "gpu_allocated_gb": gpu_allocated / 1e9,
                "gpu_reserved_gb": gpu_reserved / 1e9,
                "gpu_free_gb": (gpu_memory - gpu_reserved) / 1e9,
                "gpu_utilization": gpu_reserved / gpu_memory,
            }
        
        # CPU/MPS fallback
        memory = psutil.virtual_memory()
        return {
            "cpu_total_gb": memory.total / 1e9,
            "cpu_available_gb": memory.available / 1e9,
            "cpu_utilization": memory.percent / 100.0,
        }
    
    def cleanup_memory(self):
        """Force cleanup of GPU memory"""
        if "cuda" in self.device:
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        gc.collect()
        
    def is_memory_available(self, required_gb: float = 2.0) -> bool:
        """Check if sufficient memory is available"""
        memory_info = self.check_memory()
        
        if "cuda" in self.device:
            return memory_info["gpu_free_gb"] >= required_gb
        else:
            return memory_info["cpu_available_gb"] >= required_gb


class WANModelLoader:
    """Handles WAN 2.1 model loading and caching"""
    
    def __init__(self, memory_manager: GPUMemoryManager):
        self.memory_manager = memory_manager
        self.settings = get_settings()
        self.model_cache_dir = Path(self.settings.model.cache_dir)
        self.model_cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Model cache
        self._pipelines = {}
        self._last_access = {}
        self.max_cached_models = 2  # Limit due to GPU memory
        
    async def load_pipeline(self, model_path: str, model_type: str = "wan2-1-fast") -> DiffusionPipeline:
        """Load and cache the WAN 2.1 pipeline"""
        cache_key = f"{model_path}_{model_type}"
        
        # Return cached pipeline if available
        if cache_key in self._pipelines:
            self._last_access[cache_key] = time.time()
            logger.debug(f"Using cached pipeline: {cache_key}")
            return self._pipelines[cache_key]
        
        # Check memory before loading
        if not self.memory_manager.is_memory_available(4.0):  # Need ~4GB for WAN model
            await self._cleanup_old_models()
            
        logger.info(f"Loading WAN 2.1 model: {model_path}")
        start_time = time.time()
        
        try:
            # Load pipeline based on model type
            if "animate" in model_path.lower() or "video" in model_path.lower():
                pipeline = AnimateDiffPipeline.from_pretrained(
                    model_path,
                    cache_dir=str(self.model_cache_dir),
                    torch_dtype=torch.float16 if "cuda" in self.memory_manager.device else torch.float32,
                    use_safetensors=True,
                    variant="fp16" if "cuda" in self.memory_manager.device else None,
                )
            else:
                # Standard diffusion pipeline with video capability
                pipeline = DiffusionPipeline.from_pretrained(
                    model_path,
                    cache_dir=str(self.model_cache_dir),
                    torch_dtype=torch.float16 if "cuda" in self.memory_manager.device else torch.float32,
                    use_safetensors=True,
                    variant="fp16" if "cuda" in self.memory_manager.device else None,
                )
            
            # Move to device
            pipeline = pipeline.to(self.memory_manager.device)
            
            # Enable memory optimizations
            if hasattr(pipeline, "enable_xformers_memory_efficient_attention"):
                try:
                    pipeline.enable_xformers_memory_efficient_attention()
                    logger.info("Enabled xformers memory efficient attention")
                except Exception as e:
                    logger.warning(f"Could not enable xformers: {e}")
            
            if hasattr(pipeline, "enable_model_cpu_offload") and "cuda" in self.memory_manager.device:
                pipeline.enable_model_cpu_offload()
                logger.info("Enabled model CPU offload")
            
            if hasattr(pipeline, "enable_attention_slicing"):
                pipeline.enable_attention_slicing()
                logger.info("Enabled attention slicing")
            
            # Cache the pipeline
            self._pipelines[cache_key] = pipeline
            self._last_access[cache_key] = time.time()
            
            load_time = time.time() - start_time
            logger.info(f"Model loaded successfully in {load_time:.2f}s")
            
            return pipeline
            
        except Exception as e:
            logger.error(f"Failed to load model {model_path}: {e}")
            raise RuntimeError(f"Model loading failed: {e}")
    
    async def _cleanup_old_models(self):
        """Remove least recently used models to free memory"""
        if len(self._pipelines) <= 1:
            return
            
        # Sort by last access time
        sorted_models = sorted(
            self._last_access.items(),
            key=lambda x: x[1]
        )
        
        # Remove oldest model
        oldest_key = sorted_models[0][0]
        if oldest_key in self._pipelines:
            logger.info(f"Removing cached model: {oldest_key}")
            del self._pipelines[oldest_key]
            del self._last_access[oldest_key]
            
            # Force memory cleanup
            self.memory_manager.cleanup_memory()


class VideoProcessor:
    """Handles video processing and encoding"""
    
    def __init__(self):
        self.settings = get_settings()
        
    def frames_to_video(
        self, 
        frames: List[Union[np.ndarray, Image.Image]], 
        output_path: str, 
        fps: int = 24,
        quality: str = "high"
    ) -> bool:
        """Convert frames to MP4 video with proper encoding"""
        try:
            if not frames:
                raise ValueError("No frames provided")
            
            # Ensure output directory exists
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            
            # Convert PIL Images to numpy arrays if needed
            processed_frames = []
            for frame in frames:
                if isinstance(frame, Image.Image):
                    frame_array = np.array(frame.convert("RGB"))
                else:
                    frame_array = frame
                processed_frames.append(frame_array)
            
            # Get video dimensions
            height, width = processed_frames[0].shape[:2]
            
            # Quality settings
            quality_settings = {
                "low": {"crf": 28, "preset": "fast"},
                "medium": {"crf": 23, "preset": "medium"},
                "high": {"crf": 18, "preset": "slow"},
                "ultra": {"crf": 15, "preset": "slower"}
            }
            
            settings = quality_settings.get(quality, quality_settings["high"])
            
            # Use ffmpeg for better quality encoding
            try:
                import ffmpeg
                
                # Create temporary raw video
                temp_path = output_path.replace(".mp4", "_temp.mp4")
                
                # Write frames using OpenCV first
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                writer = cv2.VideoWriter(temp_path, fourcc, fps, (width, height))
                
                for frame in processed_frames:
                    # Convert RGB to BGR for OpenCV
                    bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    writer.write(bgr_frame)
                
                writer.release()
                
                # Re-encode with ffmpeg for better compression
                (
                    ffmpeg
                    .input(temp_path)
                    .output(
                        output_path,
                        vcodec='libx264',
                        crf=settings["crf"],
                        preset=settings["preset"],
                        pix_fmt='yuv420p'
                    )
                    .overwrite_output()
                    .run(quiet=True)
                )
                
                # Clean up temp file
                os.remove(temp_path)
                
            except ImportError:
                logger.warning("ffmpeg-python not available, using OpenCV encoding")
                # Fallback to OpenCV encoding
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
                
                for frame in processed_frames:
                    bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    writer.write(bgr_frame)
                
                writer.release()
            
            logger.info(f"Video saved: {output_path} ({len(frames)} frames, {fps} FPS)")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create video: {e}")
            return False
    
    def tensor_to_video(
        self, 
        video_tensor: torch.Tensor, 
        output_path: str, 
        fps: int = 24
    ) -> bool:
        """Convert video tensor to MP4 file"""
        try:
            # Use diffusers utility if available
            if hasattr(export_to_video, '__call__'):
                export_to_video(video_tensor, output_path, fps=fps)
                return True
            
            # Manual conversion
            if video_tensor.dim() == 5:  # (batch, channels, frames, height, width)
                video_tensor = video_tensor[0]  # Take first batch
            
            if video_tensor.dim() == 4:  # (channels, frames, height, width)
                video_tensor = video_tensor.permute(1, 2, 3, 0)  # (frames, height, width, channels)
            
            # Convert to numpy and scale to 0-255
            frames = video_tensor.cpu().numpy()
            frames = (frames * 255).astype(np.uint8)
            
            return self.frames_to_video(frames, output_path, fps)
            
        except Exception as e:
            logger.error(f"Failed to convert tensor to video: {e}")
            return False


class EnhancedVideoGenerationService:
    """Enhanced video generation service with real WAN 2.1 implementation"""
    
    def __init__(self):
        self.settings = get_settings()
        self.memory_manager = GPUMemoryManager()
        self.model_loader = WANModelLoader(self.memory_manager)
        self.video_processor = VideoProcessor()
        
        # Generation cache for duplicate requests
        self._generation_cache: Dict[str, str] = {}
        self.cache_max_size = 100
        
        logger.info(
            "Enhanced video generation service initialized",
            device=self.memory_manager.device,
            cache_dir=str(self.model_loader.model_cache_dir),
        )
    
    def _generate_cache_key(
        self,
        prompt: str,
        request: VideoGenerationRequest,
    ) -> str:
        """Generate cache key for request"""
        key_data = {
            "prompt": prompt,
            "duration": request.duration,
            "quality": request.quality.value,
            "fps": request.fps,
            "resolution": request.resolution,
            "seed": request.seed,
            "guidance_scale": request.guidance_scale,
            "num_inference_steps": request.num_inference_steps,
        }
        
        key_string = str(sorted(key_data.items()))
        return hashlib.md5(key_string.encode()).hexdigest()
    
    @track_video_generation("wan2.1")
    async def generate_video(
        self,
        request: VideoGenerationRequest,
        prompt_optimization: PromptOptimization,
        routing_decision: RoutingDecision,
    ) -> GenerationResult:
        """Generate video using real WAN 2.1 model"""
        
        # Check cache first
        cache_key = self._generate_cache_key(
            prompt_optimization.optimized_prompt,
            request,
        )
        
        if cache_key in self._generation_cache:
            cached_path = self._generation_cache[cache_key]
            if os.path.exists(cached_path):
                logger.info("Serving cached video", cache_key=cache_key)
                return self._create_cached_result(cached_path, cache_key)
        
        # Check memory availability
        memory_info = self.memory_manager.check_memory()
        logger.info("Memory status before generation", **memory_info)
        
        if not self.memory_manager.is_memory_available(3.0):
            self.memory_manager.cleanup_memory()
            
        start_time = time.time()
        
        try:
            # Load model pipeline
            pipeline = await self.model_loader.load_pipeline(
                self.settings.model.wan_model_path,
                "wan2-1-fast"
            )
            
            # Prepare generation parameters
            generation_params = self._prepare_generation_params(request, routing_decision)
            
            logger.info(
                "Starting video generation",
                prompt=prompt_optimization.optimized_prompt[:100] + "...",
                params={k: v for k, v in generation_params.items() if k != "generator"},
            )
            
            # Generate video in thread pool to avoid blocking
            video_path = await asyncio.get_event_loop().run_in_executor(
                None,
                self._generate_video_sync,
                pipeline,
                prompt_optimization.optimized_prompt,
                generation_params,
                request,
            )
            
            generation_time = time.time() - start_time
            
            # Calculate quality metrics
            quality_metrics = await self._calculate_quality_metrics(
                video_path, request, generation_time
            )
            
            # Cache the result
            if len(self._generation_cache) >= self.cache_max_size:
                # Remove oldest entry
                oldest_key = next(iter(self._generation_cache))
                del self._generation_cache[oldest_key]
                
            self._generation_cache[cache_key] = video_path
            
            # Log memory status after generation
            memory_info_after = self.memory_manager.check_memory()
            logger.info("Memory status after generation", **memory_info_after)
            
            logger.info(
                "Video generation completed",
                generation_time=generation_time,
                video_path=video_path,
                quality_metrics=quality_metrics,
            )
            
            return GenerationResult(
                video_path=video_path,
                model_used="wan2-1-fast",
                generation_time=generation_time,
                parameters={
                    "device": self.memory_manager.device,
                    "generation_params": generation_params,
                    "cache_key": cache_key,
                    "memory_usage": memory_info_after,
                },
                estimated_cost=self._estimate_cost(request, generation_time),
                quality_metrics=quality_metrics,
            )
            
        except Exception as e:
            logger.error(
                "Video generation failed",
                error=str(e),
                generation_time=time.time() - start_time,
            )
            # Cleanup memory on error
            self.memory_manager.cleanup_memory()
            raise RuntimeError(f"Video generation failed: {e}")
    
    def _prepare_generation_params(
        self,
        request: VideoGenerationRequest,
        routing_decision: RoutingDecision,
    ) -> Dict[str, Any]:
        """Prepare parameters for video generation"""
        width, height = map(int, request.resolution.split("x"))
        
        # Calculate number of frames based on duration and FPS
        num_frames = request.duration * request.fps
        
        params = {
            "width": width,
            "height": height,
            "num_frames": num_frames,
            "guidance_scale": request.guidance_scale,
            "num_inference_steps": request.num_inference_steps,
        }
        
        # Add seed if specified
        if request.seed is not None:
            params["generator"] = torch.Generator(device=self.memory_manager.device).manual_seed(request.seed)
        
        # Adjust parameters based on routing decision and memory
        if routing_decision.model_tier == "basic":
            params["num_inference_steps"] = min(params["num_inference_steps"], 15)
        elif routing_decision.model_tier == "ultra":
            params["num_inference_steps"] = max(params["num_inference_steps"], 30)
        
        # Memory-based adjustments
        memory_info = self.memory_manager.check_memory()
        if "cuda" in self.memory_manager.device:
            available_memory = memory_info.get("gpu_free_gb", 4.0)
            if available_memory < 6.0:
                # Reduce batch size or other memory-intensive parameters
                params["guidance_scale"] = min(params["guidance_scale"], 7.5)
        
        return params
    
    def _generate_video_sync(
        self,
        pipeline: DiffusionPipeline,
        prompt: str,
        params: Dict[str, Any],
        request: VideoGenerationRequest,
    ) -> str:
        """Synchronous video generation (runs in thread pool)"""
        
        # Generate unique filename
        timestamp = int(time.time())
        filename = f"video_{timestamp}_{hash(prompt) % 10000}.mp4"
        output_path = self.model_loader.model_cache_dir / "generated" / filename
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Call the pipeline
        with torch.inference_mode():
            result = pipeline(
                prompt,
                **params,
            )
        
        # Save video based on result type
        success = False
        if hasattr(result, "frames") and result.frames:
            # Frames output
            success = self.video_processor.frames_to_video(
                result.frames, 
                str(output_path), 
                request.fps,
                request.quality.value
            )
        elif hasattr(result, "videos") and len(result.videos) > 0:
            # Video tensor output
            success = self.video_processor.tensor_to_video(
                result.videos[0], 
                str(output_path), 
                request.fps
            )
        elif hasattr(result, "images") and result.images:
            # Static images - convert to video
            success = self.video_processor.frames_to_video(
                result.images * request.duration,  # Repeat images for duration
                str(output_path), 
                request.fps,
                request.quality.value
            )
        else:
            # Fallback - try to save result directly
            try:
                if hasattr(result, "save"):
                    result.save(str(output_path))
                    success = True
            except Exception as e:
                logger.error(f"Fallback save failed: {e}")
        
        if not success:
            raise RuntimeError("Failed to save generated video")
        
        return str(output_path)
    
    async def _calculate_quality_metrics(
        self,
        video_path: str,
        request: VideoGenerationRequest,
        generation_time: float,
    ) -> Dict[str, float]:
        """Calculate comprehensive quality metrics"""
        metrics = {
            "generation_time": generation_time,
            "file_exists": float(os.path.exists(video_path)),
            "expected_duration": float(request.duration),
        }
        
        if not os.path.exists(video_path):
            return metrics
        
        try:
            # Basic file metrics
            file_size = os.path.getsize(video_path)
            metrics["file_size_mb"] = file_size / (1024 * 1024)
            
            # Video analysis using OpenCV
            cap = cv2.VideoCapture(video_path)
            if cap.isOpened():
                fps = cap.get(cv2.CAP_PROP_FPS)
                frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
                actual_duration = frame_count / fps if fps > 0 else 0
                
                metrics.update({
                    "actual_fps": fps,
                    "frame_count": frame_count,
                    "actual_duration": actual_duration,
                    "duration_accuracy": min(1.0, actual_duration / request.duration) if request.duration > 0 else 0,
                })
                
                # Sample frames for quality analysis
                frame_quality_scores = []
                for i in range(0, int(frame_count), max(1, int(frame_count // 10))):
                    cap.set(cv2.CAP_PROP_POS_FRAMES, i)
                    ret, frame = cap.read()
                    if ret:
                        # Simple quality metrics: sharpness, contrast
                        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                        sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
                        contrast = gray.std()
                        frame_quality_scores.append({
                            "sharpness": sharpness,
                            "contrast": contrast
                        })
                
                if frame_quality_scores:
                    avg_sharpness = np.mean([f["sharpness"] for f in frame_quality_scores])
                    avg_contrast = np.mean([f["contrast"] for f in frame_quality_scores])
                    
                    metrics.update({
                        "avg_sharpness": avg_sharpness,
                        "avg_contrast": avg_contrast,
                        "visual_quality_score": min(1.0, (avg_sharpness / 1000 + avg_contrast / 100) / 2)
                    })
                
                cap.release()
            
            # Overall quality score
            metrics["overall_quality"] = (
                metrics["file_exists"] * 0.2 +
                metrics.get("duration_accuracy", 0) * 0.3 +
                metrics.get("visual_quality_score", 0.5) * 0.3 +
                (1.0 if generation_time < 300 else 0.5) * 0.2
            )
            
        except Exception as e:
            logger.warning(f"Could not calculate advanced metrics: {e}")
            metrics["overall_quality"] = metrics["file_exists"] * 0.5
        
        return metrics
    
    def _estimate_cost(self, request: VideoGenerationRequest, generation_time: float) -> float:
        """Estimate generation cost based on resources used"""
        base_cost = 0.10  # Base cost per video
        
        # Time-based cost
        time_cost = generation_time * 0.01  # $0.01 per second
        
        # Resolution-based cost
        width, height = map(int, request.resolution.split("x"))
        resolution_multiplier = (width * height) / (512 * 512)
        
        # Duration-based cost
        duration_multiplier = request.duration / 5  # 5 seconds baseline
        
        total_cost = base_cost + time_cost * resolution_multiplier * duration_multiplier
        return round(total_cost, 4)
    
    def _create_cached_result(self, video_path: str, cache_key: str) -> GenerationResult:
        """Create result object for cached video"""
        file_size = os.path.getsize(video_path) if os.path.exists(video_path) else 0
        
        return GenerationResult(
            video_path=video_path,
            model_used="wan2-1-fast-cached",
            generation_time=0.1,
            parameters={"cached": True, "cache_key": cache_key},
            estimated_cost=0.01,  # Minimal cost for cached result
            quality_metrics={
                "file_size_mb": file_size / (1024 * 1024),
                "cache_hit": 1.0,
                "overall_quality": 0.9,  # Assume good quality for cached
            }
        )


# Import our clean WAN service
from .wan_video_service import wan_service

# Service instance - use our clean implementation
enhanced_generation_service = wan_service
