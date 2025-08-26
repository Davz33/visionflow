"""Resource configuration for video generation services to prevent system crashes."""

import os
from dataclasses import dataclass
from typing import Optional

@dataclass
class ResourceLimits:
    """Configuration for resource limits to prevent system crashes."""
    
    # GPU/MPS memory limits (as fraction of total available memory)
    gpu_memory_fraction: float = 0.85  # Use 85% of available GPU memory (increased for dedicated GPUs)
    mps_memory_fraction: float = 0.5   # Apple Silicon MPS uses unified memory - be more conservative
    
    # System RAM limits (critical for Apple Silicon unified memory)
    max_system_ram_gb: float = 12.0  # Maximum system RAM to use (GB)
    system_ram_warning_threshold: float = 80.0  # Warning when RAM usage exceeds this percentage (lower for Mac)
    
    # Model loading limits
    max_concurrent_models: int = 1  # Only load one model at a time
    enable_model_swapping: bool = True  # Unload previous model before loading new one
    
    # Processing limits
    max_video_duration: int = 10  # Maximum video duration in seconds
    max_resolution_pixels: int = 1280 * 720  # Maximum resolution (width * height)
    
    # Memory cleanup settings
    enable_aggressive_cleanup: bool = True  # Force garbage collection after each generation
    cleanup_interval_generations: int = 1  # Clean up after every generation
    
    # Fallback settings
    fallback_to_cpu: bool = True  # Fall back to CPU if GPU memory is insufficient
    enable_memory_monitoring: bool = True  # Monitor memory usage during generation

def get_resource_limits() -> ResourceLimits:
    """Get resource limits from environment variables or use defaults."""
    return ResourceLimits(
        gpu_memory_fraction=float(os.getenv('WAN_GPU_MEMORY_FRACTION', '0.6')),
        mps_memory_fraction=float(os.getenv('WAN_MPS_MEMORY_FRACTION', '0.5')),
        max_system_ram_gb=float(os.getenv('WAN_MAX_SYSTEM_RAM_GB', '12.0')),
        system_ram_warning_threshold=float(os.getenv('WAN_RAM_WARNING_THRESHOLD', '80.0')),
        max_concurrent_models=int(os.getenv('WAN_MAX_CONCURRENT_MODELS', '1')),
        enable_model_swapping=os.getenv('WAN_ENABLE_MODEL_SWAPPING', 'true').lower() == 'true',
        max_video_duration=int(os.getenv('WAN_MAX_VIDEO_DURATION', '10')),
        max_resolution_pixels=int(os.getenv('WAN_MAX_RESOLUTION_PIXELS', str(1280 * 720))),
        enable_aggressive_cleanup=os.getenv('WAN_ENABLE_AGGRESSIVE_CLEANUP', 'true').lower() == 'true',
        cleanup_interval_generations=int(os.getenv('WAN_CLEANUP_INTERVAL', '1')),
        fallback_to_cpu=os.getenv('WAN_FALLBACK_TO_CPU', 'true').lower() == 'true',
        enable_memory_monitoring=os.getenv('WAN_ENABLE_MEMORY_MONITORING', 'true').lower() == 'true'
    )

# Environment variable examples for easy configuration:
# export WAN_GPU_MEMORY_FRACTION=0.6        # Use only 60% of GPU memory
# export WAN_MAX_SYSTEM_RAM_GB=8.0          # Limit to 8GB system RAM
# export WAN_MAX_VIDEO_DURATION=5            # Limit videos to 5 seconds
# export WAN_MAX_RESOLUTION_PIXELS=640480    # Limit to 640x480 resolution
