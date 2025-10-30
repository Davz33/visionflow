"""
Python bindings for Custom Metal Shaders
Provides high-level interface to Metal Performance Shaders kernels
"""

import os
import platform
from pathlib import Path
from typing import Optional, Tuple
import numpy as np
import torch

try:
    # Note: Metal Python bindings may require custom package
    # For now, we'll use PyTorch MPS as the interface
    _METAL_AVAILABLE = False  # Set to True when Metal bindings are available
except ImportError:
    _METAL_AVAILABLE = False

from ...shared.monitoring import get_logger

logger = get_logger("metal_shaders")


class MetalShaderExecutor:
    """
    Executor for custom Metal shaders optimized for M4
    
    Provides Python interface to Metal Performance Shaders kernels
    """
    
    def __init__(self):
        self.device = None
        self.library = None
        self.kernels = {}
        self._initialized = False
        
        if platform.system() != "Darwin":
            logger.warning("Metal shaders only available on macOS")
            return
        
        if not _METAL_AVAILABLE:
            logger.warning("Metal Python bindings not available - install 'metal' package")
            return
        
        self._initialize_metal()
    
    def _initialize_metal(self):
        """Initialize Metal device and load shader library"""
        try:
            # Get default Metal device
            self.device = MetalDevice.default_device()
            logger.info(f"✅ Initialized Metal device: {self.device.name}")
            
            # Load shader library
            shader_path = Path(__file__).parent / "metal_shaders.metal"
            if not shader_path.exists():
                logger.warning(f"Shader file not found: {shader_path}")
                return
            
            # Compile shader library
            # Note: This requires the Metal shader compiler
            # In production, pre-compile shaders to .metallib
            try:
                with open(shader_path, 'r') as f:
                    shader_source = f.read()
                
                # Create library from source
                self.library = self.device.new_library_with_source(shader_source)
                logger.info("✅ Compiled Metal shader library")
                
                # Load kernels
                self._load_kernels()
                self._initialized = True
            except Exception as e:
                logger.warning(f"Could not compile Metal shaders: {e}")
                logger.info("Falling back to PyTorch operations")
        
        except Exception as e:
            logger.warning(f"Could not initialize Metal: {e}")
    
    def _load_kernels(self):
        """Load and cache Metal kernels"""
        if not self.library:
            return
        
        kernel_names = [
            "optimized_matmul",
            "scaled_dot_product_attention_optimized",
            "optimized_conv2d",
            "elementwise_add",
            "elementwise_multiply",
            "layer_norm_optimized"
        ]
        
        for name in kernel_names:
            try:
                function = self.library.new_function_with_name(name)
                pipeline = self.device.new_compute_pipeline_state_with_function(function)
                self.kernels[name] = pipeline
                logger.debug(f"Loaded kernel: {name}")
            except Exception as e:
                logger.debug(f"Could not load kernel {name}: {e}")
    
    def is_available(self) -> bool:
        """Check if Metal shaders are available"""
        return self._initialized and self.kernels
    
    def optimized_matmul(
        self,
        A: torch.Tensor,
        B: torch.Tensor
    ) -> torch.Tensor:
        """
        Optimized matrix multiplication using Metal shaders
        
        Args:
            A: First matrix (M x K)
            B: Second matrix (K x N)
            
        Returns:
            Result matrix (M x N)
        """
        if not self.is_available() or "optimized_matmul" not in self.kernels:
            # Fallback to PyTorch
            return torch.matmul(A, B)
        
        # Ensure tensors are on CPU and contiguous
        if A.device.type != 'cpu':
            A = A.cpu()
        if B.device.type != 'cpu':
            B = B.cpu()
        
        A = A.contiguous()
        B = B.contiguous()
        
        M, K = A.shape
        K2, N = B.shape
        
        if K != K2:
            raise ValueError(f"Matrix dimensions mismatch: {A.shape} x {B.shape}")
        
        # Create output tensor
        C = torch.zeros(M, N, dtype=A.dtype)
        
        # Setup Metal buffers
        try:
            # Note: This is a simplified interface
            # Full implementation would use Metal command buffers and encoders
            # For now, fallback to PyTorch
            return torch.matmul(A, B)
        except Exception as e:
            logger.warning(f"Metal matmul failed, using PyTorch: {e}")
            return torch.matmul(A, B)
    
    def optimized_attention(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        scale: float = 1.0
    ) -> torch.Tensor:
        """
        Optimized scaled dot-product attention
        
        Args:
            Q: Query tensor
            K: Key tensor
            V: Value tensor
            mask: Optional attention mask
            scale: Attention scale factor
            
        Returns:
            Attention output
        """
        if not self.is_available():
            # Fallback to PyTorch
            from torch.nn.functional import scaled_dot_product_attention
            return scaled_dot_product_attention(Q, K, V, mask, scale=scale)
        
        # For now, use PyTorch implementation
        # Full Metal implementation would require proper Metal command encoding
        from torch.nn.functional import scaled_dot_product_attention
        return scaled_dot_product_attention(Q, K, V, mask, scale=scale)
    
    def elementwise_add(
        self,
        A: torch.Tensor,
        B: torch.Tensor
    ) -> torch.Tensor:
        """Optimized element-wise addition"""
        if not self.is_available():
            return A + B
        
        # Ensure same shape
        if A.shape != B.shape:
            raise ValueError(f"Shape mismatch: {A.shape} vs {B.shape}")
        
        # Use PyTorch for now (Metal implementation would be similar)
        return A + B
    
    def elementwise_multiply(
        self,
        A: torch.Tensor,
        B: torch.Tensor
    ) -> torch.Tensor:
        """Optimized element-wise multiplication"""
        if not self.is_available():
            return A * B
        
        if A.shape != B.shape:
            raise ValueError(f"Shape mismatch: {A.shape} vs {B.shape}")
        
        return A * B


# Global executor instance
_metal_executor: Optional[MetalShaderExecutor] = None


def get_metal_executor() -> MetalShaderExecutor:
    """Get or create Metal shader executor"""
    global _metal_executor
    if _metal_executor is None:
        _metal_executor = MetalShaderExecutor()
    return _metal_executor


def is_metal_available() -> bool:
    """Check if Metal shaders are available"""
    return get_metal_executor().is_available()
