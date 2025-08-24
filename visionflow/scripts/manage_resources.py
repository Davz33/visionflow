#!/usr/bin/env python3
"""Resource management script for WAN video generation service."""

import os
import sys
import argparse
import psutil
import torch
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from visionflow.services.generation.resource_config import get_resource_limits, ResourceLimits

def get_system_info():
    """Get current system resource information."""
    memory = psutil.virtual_memory()
    cpu_percent = psutil.cpu_percent(interval=1)
    
    info = {
        "system_ram_total_gb": memory.total / (1024**3),
        "system_ram_used_gb": memory.used / (1024**3),
        "system_ram_percent": memory.percent,
        "cpu_percent": cpu_percent
    }
    
    if torch.cuda.is_available():
        gpu_props = torch.cuda.get_device_properties(0)
        info.update({
            "gpu_name": gpu_props.name,
            "gpu_memory_total_gb": gpu_props.total_memory / (1024**3),
            "gpu_memory_allocated_gb": torch.cuda.memory_allocated() / (1024**3),
            "gpu_memory_reserved_gb": torch.cuda.memory_reserved() / (1024**3)
        })
    
    return info

def print_system_info():
    """Print current system resource information."""
    info = get_system_info()
    
    print("🖥️  System Resource Information")
    print("=" * 50)
    print(f"RAM: {info['system_ram_used_gb']:.1f}GB / {info['system_ram_total_gb']:.1f}GB ({info['system_ram_percent']:.1f}%)")
    print(f"CPU: {info['cpu_percent']:.1f}%")
    
    if 'gpu_name' in info:
        print(f"GPU: {info['gpu_name']}")
        print(f"GPU Memory: {info['gpu_memory_allocated_gb']:.1f}GB / {info['gpu_memory_total_gb']:.1f}GB")
        print(f"GPU Memory Reserved: {info['gpu_memory_reserved_gb']:.1f}GB")
    
    print()

def print_current_limits():
    """Print current resource limits configuration."""
    limits = get_resource_limits()
    
    print("🔒 Current Resource Limits")
    print("=" * 50)
    print(f"GPU Memory Fraction: {limits.gpu_memory_fraction * 100:.1f}%")
    print(f"Max System RAM: {limits.max_system_ram_gb:.1f}GB")
    print(f"Max Video Duration: {limits.max_video_duration}s")
    print(f"Max Resolution Pixels: {limits.max_resolution_pixels:,}")
    print(f"Enable Aggressive Cleanup: {limits.enable_aggressive_cleanup}")
    print(f"Cleanup Interval: Every {limits.cleanup_interval_generations} generation(s)")
    print()

def set_environment_variable(var_name: str, value: str):
    """Set an environment variable for the current session."""
    os.environ[var_name] = value
    print(f"✅ Set {var_name}={value}")

def create_env_file():
    """Create a .env file with recommended resource limits."""
    # Detect if we're on Apple Silicon Mac
    is_apple_silicon = torch.backends.mps.is_available() if hasattr(torch.backends, 'mps') else False
    
    if is_apple_silicon:
        env_content = """# WAN Video Generation Resource Limits for Apple Silicon Mac
# Optimized for MPS (Metal Performance Shaders) and unified memory
# Adjust these values to prevent system crashes

# MPS memory limit (Apple Silicon uses unified memory)
WAN_MPS_MEMORY_FRACTION=0.5

# System RAM limit (conservative for unified memory)
WAN_MAX_SYSTEM_RAM_GB=8.0

# RAM warning threshold (lower for unified memory)
WAN_RAM_WARNING_THRESHOLD=75.0

# Video generation limits (conservative for Apple Silicon)
WAN_MAX_VIDEO_DURATION=5
WAN_MAX_RESOLUTION_PIXELS=640480

# Memory management (aggressive cleanup for unified memory)
WAN_ENABLE_AGGRESSIVE_CLEANUP=true
WAN_CLEANUP_INTERVAL=1

# Model management (important for unified memory)
WAN_ENABLE_MODEL_SWAPPING=true
WAN_MAX_CONCURRENT_MODELS=1

# Fallback settings
WAN_FALLBACK_TO_CPU=true
WAN_ENABLE_MEMORY_MONITORING=true

# Apple Silicon specific optimizations
PYTORCH_ENABLE_MPS_FALLBACK=1
"""
    else:
        env_content = """# WAN Video Generation Resource Limits
# Adjust these values to prevent system crashes

# GPU memory limit (as fraction of total VRAM)
WAN_GPU_MEMORY_FRACTION=0.6

# System RAM limit
WAN_MAX_SYSTEM_RAM_GB=8.0

# RAM warning threshold (percentage)
WAN_RAM_WARNING_THRESHOLD=80.0

# Video generation limits
WAN_MAX_VIDEO_DURATION=5
WAN_MAX_RESOLUTION_PIXELS=640480

# Memory management
WAN_ENABLE_AGGRESSIVE_CLEANUP=true
WAN_CLEANUP_INTERVAL=1

# Model management
WAN_ENABLE_MODEL_SWAPPING=true
WAN_MAX_CONCURRENT_MODELS=1

# Fallback settings
WAN_FALLBACK_TO_CPU=true
WAN_ENABLE_MEMORY_MONITORING=true
"""
    
    env_file = project_root / ".env"
    with open(env_file, "w") as f:
        f.write(env_content)
    
    platform_type = "Apple Silicon Mac" if is_apple_silicon else "Standard"
    print(f"✅ Created {env_file} ({platform_type} optimized)")
    print("📝 Edit this file to adjust resource limits")
    print("🔄 Restart your service after making changes")

def recommend_limits():
    """Recommend resource limits based on current system."""
    info = get_system_info()
    
    print("💡 Recommended Resource Limits")
    print("=" * 50)
    
    # Detect if we're on Apple Silicon Mac
    is_apple_silicon = torch.backends.mps.is_available() if hasattr(torch.backends, 'mps') else False
    
    if is_apple_silicon:
        print("🍎 Apple Silicon Mac detected - using MPS-optimized limits")
        print("⚠️  Note: MPS uses unified memory (shared between CPU and GPU)")
    
    # RAM recommendations
    total_ram = info['system_ram_total_gb']
    if is_apple_silicon:
        # More conservative for Apple Silicon due to unified memory
        if total_ram <= 8:
            recommended_ram = total_ram * 0.5  # Very conservative
            recommended_mps_fraction = 0.4
            print(f"⚠️  Low unified memory ({total_ram:.1f}GB): Limit to {recommended_ram:.1f}GB")
        elif total_ram <= 16:
            recommended_ram = total_ram * 0.6
            recommended_mps_fraction = 0.5
            print(f"📊 Medium unified memory ({total_ram:.1f}GB): Limit to {recommended_ram:.1f}GB")
        else:
            recommended_ram = total_ram * 0.7
            recommended_mps_fraction = 0.6
            print(f"🚀 High unified memory ({total_ram:.1f}GB): Limit to {recommended_ram:.1f}GB")
        
        print(f"🔒 Recommended MPS memory fraction: {recommended_mps_fraction * 100:.0f}%")
    else:
        # Standard recommendations for non-Apple Silicon
        if total_ram <= 8:
            recommended_ram = total_ram * 0.6
            print(f"⚠️  Low RAM system ({total_ram:.1f}GB): Limit to {recommended_ram:.1f}GB")
        elif total_ram <= 16:
            recommended_ram = total_ram * 0.7
            print(f"📊 Medium RAM system ({total_ram:.1f}GB): Limit to {recommended_ram:.1f}GB")
        else:
            recommended_ram = total_ram * 0.8
            print(f"🚀 High RAM system ({total_ram:.1f}GB): Limit to {recommended_ram:.1f}GB")
    
    # GPU recommendations
    if 'gpu_memory_total_gb' in info and not is_apple_silicon:
        gpu_memory = info['gpu_memory_total_gb']
        if gpu_memory <= 8:
            recommended_gpu_fraction = 0.5
            print(f"⚠️  Low VRAM GPU ({gpu_memory:.1f}GB): Use {recommended_gpu_fraction * 100:.0f}% of VRAM")
        elif gpu_memory <= 16:
            recommended_gpu_fraction = 0.7
            print(f"📊 Medium VRAM GPU ({gpu_memory:.1f}GB): Use {recommended_gpu_fraction * 100:.0f}% of VRAM")
        else:
            recommended_gpu_fraction = 0.8
            print(f"🚀 High VRAM GPU ({gpu_memory:.1f}GB): Use {recommended_gpu_fraction * 100:.0f}% of VRAM")
    
    print()
    print("💡 Quick setup commands:")
    print(f"export WAN_MAX_SYSTEM_RAM_GB={recommended_ram:.1f}")
    if is_apple_silicon:
        print(f"export WAN_MPS_MEMORY_FRACTION={recommended_mps_fraction}")
        print("export WAN_RAM_WARNING_THRESHOLD=75.0  # Lower threshold for unified memory")
        print("export WAN_MAX_VIDEO_DURATION=5        # Shorter videos for Apple Silicon")
    elif 'gpu_memory_total_gb' in info:
        print(f"export WAN_GPU_MEMORY_FRACTION={recommended_gpu_fraction}")
    print()

def main():
    parser = argparse.ArgumentParser(description="Manage WAN video generation resource limits")
    parser.add_argument("--info", action="store_true", help="Show current system resource information")
    parser.add_argument("--limits", action="store_true", help="Show current resource limits configuration")
    parser.add_argument("--recommend", action="store_true", help="Recommend resource limits for your system")
    parser.add_argument("--create-env", action="store_true", help="Create a .env file with recommended limits")
    parser.add_argument("--set-gpu-fraction", type=float, help="Set GPU memory fraction (0.0-1.0)")
    parser.add_argument("--set-max-ram", type=float, help="Set maximum system RAM in GB")
    parser.add_argument("--set-max-duration", type=int, help="Set maximum video duration in seconds")
    
    args = parser.parse_args()
    
    if not any(vars(args).values()):
        # Default: show all information
        print_system_info()
        print_current_limits()
        recommend_limits()
        return
    
    if args.info:
        print_system_info()
    
    if args.limits:
        print_current_limits()
    
    if args.recommend:
        recommend_limits()
    
    if args.create_env:
        create_env_file()
    
    if args.set_gpu_fraction is not None:
        if 0.0 <= args.set_gpu_fraction <= 1.0:
            set_environment_variable("WAN_GPU_MEMORY_FRACTION", str(args.set_gpu_fraction))
        else:
            print("❌ GPU memory fraction must be between 0.0 and 1.0")
    
    if args.set_max_ram is not None:
        if args.set_max_ram > 0:
            set_environment_variable("WAN_MAX_SYSTEM_RAM_GB", str(args.set_max_ram))
        else:
            print("❌ Maximum RAM must be greater than 0")
    
    if args.set_max_duration is not None:
        if args.set_max_duration > 0:
            set_environment_variable("WAN_MAX_VIDEO_DURATION", str(args.set_max_duration))
        else:
            print("❌ Maximum duration must be greater than 0")

if __name__ == "__main__":
    main()
