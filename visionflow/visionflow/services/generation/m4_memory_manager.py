"""
M4 Unified Memory Architecture Optimized Memory Manager
Handles memory management specific to Apple Silicon's unified memory architecture
"""

import gc
import os
import platform
import psutil
from typing import Dict, Optional
import torch

from ...shared.monitoring import get_logger

logger = get_logger("m4_memory_manager")


class M4UnifiedMemoryManager:
    """
    Memory manager optimized for M4's unified memory architecture
    
    Key differences from traditional GPU memory:
    - Unified memory shared between CPU, GPU, and Neural Engine
    - No separate VRAM - all memory is system RAM
    - Memory pressure affects entire system
    - Can be more aggressive with memory allocation due to unified nature
    """
    
    def __init__(self, memory_warning_threshold: float = 0.85, memory_critical_threshold: float = 0.95):
        """
        Initialize M4 unified memory manager
        
        Args:
            memory_warning_threshold: Percentage of memory usage to trigger warning (0-1)
            memory_critical_threshold: Percentage of memory usage to trigger critical state (0-1)
        """
        self.is_m4 = self._detect_m4()
        self.is_mps_available = torch.backends.mps.is_available() if hasattr(torch.backends, 'mps') else False
        self.memory_warning_threshold = memory_warning_threshold
        self.memory_critical_threshold = memory_critical_threshold
        
        if self.is_m4:
            logger.info("🍎 M4 Unified Memory Manager initialized")
            self._configure_unified_memory()
        else:
            logger.warning("M4 not detected - using standard memory management")
    
    def _detect_m4(self) -> bool:
        """Detect if running on M4 chip"""
        if platform.system() != "Darwin":
            return False
        
        try:
            arch = platform.machine()
            if arch != "arm64":
                return False
            
            cpu_info = os.popen('sysctl -n machdep.cpu.brand_string').read().strip()
            return "M4" in cpu_info or ("Apple" in cpu_info and self.is_mps_available)
        except Exception:
            return False
    
    def _configure_unified_memory(self):
        """Configure settings for unified memory architecture"""
        if not self.is_mps_available:
            return
        
        try:
            # Configure MPS memory settings
            # Unified memory allows more flexible memory management
            if hasattr(torch.backends.mps, 'recommended_memory_fraction'):
                # Be more aggressive with unified memory (up to 85%)
                torch.backends.mps.recommended_memory_fraction = 0.85
            
            logger.info("✅ Unified memory configured")
        except Exception as e:
            logger.warning(f"Could not configure unified memory: {e}")
    
    def get_memory_info(self) -> Dict[str, float]:
        """
        Get current unified memory usage information
        
        Returns comprehensive memory statistics for unified memory architecture
        """
        vm = psutil.virtual_memory()
        
        memory_info = {
            "total_memory_gb": vm.total / (1024**3),
            "available_memory_gb": vm.available / (1024**3),
            "used_memory_gb": vm.used / (1024**3),
            "memory_percent": vm.percent / 100.0,
            "memory_pressure": self._get_memory_pressure(),
        }
        
        # Add MPS-specific memory info if available
        if self.is_mps_available:
            try:
                # On unified memory, MPS memory is part of system memory
                # But we can still track MPS allocations
                mps_allocated = torch.mps.current_allocated_memory() if hasattr(torch.mps, 'current_allocated_memory') else 0
                memory_info.update({
                    "mps_allocated_mb": mps_allocated / (1024**2),
                    "is_unified_memory": True,
                })
            except Exception as e:
                logger.debug(f"Could not get MPS memory info: {e}")
        
        return memory_info
    
    def _get_memory_pressure(self) -> float:
        """
        Get memory pressure level (0-1)
        
        Higher values indicate more memory pressure
        """
        vm = psutil.virtual_memory()
        pressure = vm.percent / 100.0
        
        # Check for memory pressure warnings on macOS
        try:
            # On macOS, we can check memory pressure status
            pressure_status = os.popen('memory_pressure').read().strip()
            if 'WARNING' in pressure_status:
                pressure = max(pressure, 0.7)
            elif 'CRITICAL' in pressure_status:
                pressure = max(pressure, 0.9)
        except Exception:
            pass
        
        return pressure
    
    def check_memory_availability(self, required_gb: float) -> tuple[bool, Dict[str, float]]:
        """
        Check if sufficient unified memory is available
        
        Args:
            required_gb: Required memory in GB
            
        Returns:
            (is_available: bool, memory_info: dict)
        """
        memory_info = self.get_memory_info()
        available_gb = memory_info["available_memory_gb"]
        
        # For unified memory, we can use more aggressively
        # Reserve 20% for system overhead
        usable_memory = available_gb * 0.8
        
        is_available = usable_memory >= required_gb
        
        if not is_available:
            logger.warning(
                f"⚠️ Insufficient unified memory: {required_gb:.2f}GB required, "
                f"{usable_memory:.2f}GB usable available"
            )
        
        return is_available, memory_info
    
    def get_memory_status(self) -> str:
        """
        Get human-readable memory status
        
        Returns:
            Status string: "healthy", "warning", or "critical"
        """
        memory_info = self.get_memory_info()
        memory_percent = memory_info["memory_percent"]
        
        if memory_percent >= self.memory_critical_threshold:
            return "critical"
        elif memory_percent >= self.memory_warning_threshold:
            return "warning"
        else:
            return "healthy"
    
    def cleanup_memory(self, aggressive: bool = False):
        """
        Clean up unified memory
        
        Args:
            aggressive: If True, perform aggressive cleanup (multiple GC passes)
        """
        logger.info("🧹 Cleaning up unified memory")
        
        # Force Python garbage collection
        gc.collect()
        
        if aggressive:
            # Multiple passes for aggressive cleanup
            for i in range(3):
                gc.collect()
                logger.debug(f"Aggressive GC pass {i+1}")
        
        # Clear MPS cache if available
        if self.is_mps_available:
            try:
                if hasattr(torch.mps, 'empty_cache'):
                    torch.mps.empty_cache()
                    logger.info("✅ Cleared MPS cache")
            except Exception as e:
                logger.debug(f"Could not clear MPS cache: {e}")
        
        # Get final memory info
        memory_info = self.get_memory_info()
        logger.info(
            f"✅ Memory cleanup completed. "
            f"Available: {memory_info['available_memory_gb']:.2f}GB "
            f"({memory_info['memory_percent']*100:.1f}% used)"
        )
    
    def get_optimal_batch_size(
        self,
        model_size_gb: float,
        input_size_gb: float,
        reserve_percent: float = 0.2
    ) -> int:
        """
        Calculate optimal batch size for unified memory
        
        Args:
            model_size_gb: Size of model in GB
            input_size_gb: Size of single input batch in GB
            reserve_percent: Percentage of memory to reserve (default 20%)
            
        Returns:
            Optimal batch size
        """
        memory_info = self.get_memory_info()
        total_memory = memory_info["total_memory_gb"]
        
        # Reserve some memory for system overhead
        usable_memory = total_memory * (1 - reserve_percent)
        
        # Calculate how much memory is available for batches
        available_for_batches = usable_memory - model_size_gb
        
        if available_for_batches <= 0:
            logger.warning("No memory available for batches - model too large")
            return 1
        
        # Calculate batch size
        optimal_batch = max(1, int(available_for_batches / input_size_gb))
        
        logger.info(
            f"📊 Optimal batch size: {optimal_batch} "
            f"(model: {model_size_gb:.2f}GB, input: {input_size_gb:.2f}GB, "
            f"available: {available_for_batches:.2f}GB)"
        )
        
        return optimal_batch
    
    def monitor_memory_during_execution(self, callback, *args, **kwargs):
        """
        Monitor memory usage during execution and cleanup if needed
        
        Args:
            callback: Function to execute
            *args, **kwargs: Arguments to pass to callback
        """
        initial_memory = self.get_memory_info()
        
        try:
            result = callback(*args, **kwargs)
            
            final_memory = self.get_memory_info()
            memory_delta = final_memory["memory_percent"] - initial_memory["memory_percent"]
            
            if memory_delta > 0.1:  # More than 10% increase
                logger.warning(f"⚠️ Significant memory increase during execution: {memory_delta*100:.1f}%")
                self.cleanup_memory(aggressive=True)
            
            return result
        except Exception as e:
            # Cleanup on error
            self.cleanup_memory(aggressive=True)
            raise


# Global instance
_m4_memory_manager: Optional[M4UnifiedMemoryManager] = None


def get_m4_memory_manager() -> M4UnifiedMemoryManager:
    """Get or create M4 unified memory manager"""
    global _m4_memory_manager
    if _m4_memory_manager is None:
        _m4_memory_manager = M4UnifiedMemoryManager()
    return _m4_memory_manager
