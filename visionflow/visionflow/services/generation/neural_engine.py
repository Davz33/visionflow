"""
Neural Engine Integration for M4 Chip
Leverages Apple's Neural Engine for ML operations
"""

import os
import platform
from typing import Optional, Dict, Any, List
import numpy as np
import torch

try:
    import coremltools as ct
    from coremltools import models
    _COREML_AVAILABLE = True
except ImportError:
    _COREML_AVAILABLE = False

try:
    from vision import framework as vision_framework
    _VISION_AVAILABLE = True
except ImportError:
    _VISION_AVAILABLE = False

from ...shared.monitoring import get_logger

logger = get_logger("neural_engine")


class NeuralEngineManager:
    """
    Manager for Apple Neural Engine operations
    
    Offloads ML operations to the Neural Engine for optimal performance
    """
    
    def __init__(self):
        self.is_available = self._check_neural_engine()
        self.coreml_models = {}
        self.device_type = "neural_engine" if self.is_available else "cpu"
        
        if self.is_available:
            logger.info("🍎 Neural Engine available - ML operations can be accelerated")
        else:
            logger.warning("Neural Engine not available - using CPU fallback")
    
    def _check_neural_engine(self) -> bool:
        """Check if Neural Engine is available"""
        if platform.system() != "Darwin":
            return False
        
        # Check for Core ML availability
        if not _COREML_AVAILABLE:
            logger.debug("Core ML not available")
            return False
        
        # Check for Apple Silicon
        import platform as pf
        if pf.machine() != "arm64":
            return False
        
        # Neural Engine is available on Apple Silicon
        return True
    
    def convert_pytorch_to_coreml(
        self,
        model: torch.nn.Module,
        input_shape: tuple,
        output_path: Optional[str] = None
    ) -> Optional[Any]:
        """
        Convert PyTorch model to Core ML for Neural Engine execution
        
        Args:
            model: PyTorch model to convert
            input_shape: Input shape tuple
            output_path: Optional path to save Core ML model
            
        Returns:
            Core ML model if successful, None otherwise
        """
        if not self.is_available or not _COREML_AVAILABLE:
            logger.warning("Neural Engine not available - cannot convert model")
            return None
        
        try:
            # Set model to evaluation mode
            model.eval()
            
            # Create example input
            example_input = torch.randn(1, *input_shape)
            
            # Trace the model
            traced_model = torch.jit.trace(model, example_input)
            
            # Convert to Core ML
            coreml_model = ct.convert(
                traced_model,
                inputs=[ct.TensorType(name="input", shape=example_input.shape)],
                compute_units=ct.ComputeUnit.ALL  # Use Neural Engine + CPU + GPU
            )
            
            # Save if path provided
            if output_path:
                coreml_model.save(output_path)
                logger.info(f"✅ Saved Core ML model to {output_path}")
            
            logger.info("✅ Converted PyTorch model to Core ML")
            return coreml_model
        
        except Exception as e:
            logger.error(f"Failed to convert PyTorch model to Core ML: {e}")
            return None
    
    def run_on_neural_engine(
        self,
        model_path: str,
        input_data: np.ndarray
    ) -> Optional[np.ndarray]:
        """
        Run inference on Neural Engine using Core ML model
        
        Args:
            model_path: Path to Core ML model
            input_data: Input data as numpy array
            
        Returns:
            Output data as numpy array, or None if failed
        """
        if not self.is_available:
            logger.warning("Neural Engine not available")
            return None
        
        try:
            # Load Core ML model
            if model_path not in self.coreml_models:
                coreml_model = ct.models.MLModel(model_path)
                self.coreml_models[model_path] = coreml_model
            else:
                coreml_model = self.coreml_models[model_path]
            
            # Prepare input
            input_dict = {"input": input_data}
            
            # Run prediction
            prediction = coreml_model.predict(input_dict)
            
            # Extract output
            if isinstance(prediction, dict):
                # Get first output value
                output = list(prediction.values())[0]
            else:
                output = prediction
            
            return output
        
        except Exception as e:
            logger.error(f"Neural Engine inference failed: {e}")
            return None
    
    def optimize_for_neural_engine(
        self,
        model: torch.nn.Module,
        input_shape: tuple
    ) -> Optional[Any]:
        """
        Optimize model for Neural Engine execution
        
        Args:
            model: PyTorch model
            input_shape: Input shape
            
        Returns:
            Optimized Core ML model
        """
        if not self.is_available:
            return None
        
        # Convert to Core ML
        coreml_model = self.convert_pytorch_to_coreml(model, input_shape)
        
        if coreml_model is None:
            return None
        
        try:
            # Optimize for Neural Engine
            # Core ML automatically optimizes for Neural Engine when using ComputeUnit.ALL
            logger.info("✅ Model optimized for Neural Engine")
            return coreml_model
        
        except Exception as e:
            logger.error(f"Failed to optimize model: {e}")
            return None
    
    def get_neural_engine_info(self) -> Dict[str, Any]:
        """Get information about Neural Engine capabilities"""
        info = {
            "available": self.is_available,
            "device_type": self.device_type,
            "coreml_available": _COREML_AVAILABLE,
            "loaded_models": len(self.coreml_models),
        }
        
        if self.is_available:
            # Try to get Core ML compute units info
            try:
                # On Apple Silicon, Neural Engine is available
                info["compute_units"] = "ALL (Neural Engine + CPU + GPU)"
            except Exception:
                pass
        
        return info


class NeuralEngineOffloader:
    """
    Intelligent offloader for Neural Engine operations
    
    Automatically decides when to use Neural Engine vs CPU/GPU
    """
    
    def __init__(self):
        self.neural_engine = NeuralEngineManager()
        self.offload_threshold = 0.1  # Operations larger than this benefit from Neural Engine
    
    def should_offload(self, operation_size: int, operation_type: str) -> bool:
        """
        Determine if operation should be offloaded to Neural Engine
        
        Args:
            operation_size: Size/complexity of operation
            operation_type: Type of operation (e.g., 'conv', 'matmul', 'attention')
            
        Returns:
            True if should offload, False otherwise
        """
        if not self.neural_engine.is_available:
            return False
        
        # Offload large operations
        if operation_size > self.offload_threshold:
            return True
        
        # Always offload certain operation types
        neural_engine_optimal = ['conv', 'depthwise_conv', 'grouped_conv']
        if operation_type in neural_engine_optimal:
            return True
        
        return False
    
    def offload_operation(
        self,
        model: torch.nn.Module,
        input_data: np.ndarray,
        operation_type: str = "inference"
    ) -> Optional[np.ndarray]:
        """
        Offload operation to Neural Engine if beneficial
        
        Args:
            model: PyTorch model
            input_data: Input data
            operation_type: Type of operation
            
        Returns:
            Output data or None if offload not possible
        """
        operation_size = input_data.size * input_data.itemsize / (1024**2)  # Size in MB
        
        if not self.should_offload(operation_size, operation_type):
            return None
        
        # Convert model if needed
        model_hash = hash(str(model))
        cache_key = f"model_{model_hash}"
        
        if cache_key not in self.neural_engine.coreml_models:
            # Convert model
            input_shape = input_data.shape[1:]  # Remove batch dimension
            coreml_model = self.neural_engine.convert_pytorch_to_coreml(
                model, input_shape
            )
            
            if coreml_model is None:
                return None
            
            # Cache model
            self.neural_engine.coreml_models[cache_key] = coreml_model
        
        # Run on Neural Engine
        return self.neural_engine.run_on_neural_engine(cache_key, input_data)


# Global instances
_neural_engine_manager: Optional[NeuralEngineManager] = None
_neural_engine_offloader: Optional[NeuralEngineOffloader] = None


def get_neural_engine_manager() -> NeuralEngineManager:
    """Get or create Neural Engine manager"""
    global _neural_engine_manager
    if _neural_engine_manager is None:
        _neural_engine_manager = NeuralEngineManager()
    return _neural_engine_manager


def get_neural_engine_offloader() -> NeuralEngineOffloader:
    """Get or create Neural Engine offloader"""
    global _neural_engine_offloader
    if _neural_engine_offloader is None:
        _neural_engine_offloader = NeuralEngineOffloader()
    return _neural_engine_offloader
