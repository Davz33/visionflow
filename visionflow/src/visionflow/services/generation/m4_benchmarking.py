"""
Performance Profiling and Benchmarking Suite for M4 Optimizations
"""

import time
import statistics
import json
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Any
from pathlib import Path
from contextlib import contextmanager
import psutil
import torch

from ...shared.monitoring import get_logger

logger = get_logger("m4_benchmarking")


@dataclass
class BenchmarkResult:
    """Result of a single benchmark run"""
    name: str
    duration_ms: float
    memory_peak_mb: float
    memory_delta_mb: float
    cpu_percent: float
    device: str
    metadata: Dict[str, Any]


@dataclass
class BenchmarkSuite:
    """Collection of benchmark results"""
    suite_name: str
    timestamp: str
    platform: str
    results: List[BenchmarkResult]
    summary: Dict[str, float]


class M4PerformanceProfiler:
    """
    Performance profiler for M4 optimizations
    
    Profiles operations and generates detailed performance reports
    """
    
    def __init__(self, output_dir: Optional[Path] = None):
        self.output_dir = output_dir or Path("benchmarks")
        self.output_dir.mkdir(exist_ok=True)
        self.benchmarks: List[BenchmarkResult] = []
        self.process = psutil.Process()
    
    @contextmanager
    def profile(self, name: str, device: str = "cpu", metadata: Optional[Dict] = None):
        """
        Context manager for profiling operations
        
        Usage:
            with profiler.profile("matrix_multiply", device="mps"):
                result = torch.matmul(A, B)
        """
        # Record initial state
        initial_memory = self.process.memory_info().rss / (1024**2)
        initial_cpu = self.process.cpu_percent()
        start_time = time.perf_counter()
        
        # Track peak memory
        peak_memory = initial_memory
        
        try:
            # Sync GPU if using CUDA/MPS
            if device == "cuda" and torch.cuda.is_available():
                torch.cuda.synchronize()
            elif device == "mps" and torch.backends.mps.is_available():
                torch.mps.synchronize()
            
            yield
            
            # Sync again after operation
            if device == "cuda" and torch.cuda.is_available():
                torch.cuda.synchronize()
            elif device == "mps" and torch.backends.mps.is_available():
                torch.mps.synchronize()
            
            # Update peak memory
            peak_memory = max(peak_memory, self.process.memory_info().rss / (1024**2))
        
        finally:
            # Record final state
            end_time = time.perf_counter()
            final_memory = self.process.memory_info().rss / (1024**2)
            final_cpu = self.process.cpu_percent()
            
            duration_ms = (end_time - start_time) * 1000
            memory_delta = final_memory - initial_memory
            peak_memory_delta = peak_memory - initial_memory
            
            result = BenchmarkResult(
                name=name,
                duration_ms=duration_ms,
                memory_peak_mb=peak_memory_delta,
                memory_delta_mb=memory_delta,
                cpu_percent=final_cpu - initial_cpu,
                device=device,
                metadata=metadata or {}
            )
            
            self.benchmarks.append(result)
            logger.debug(f"Profiled {name}: {duration_ms:.2f}ms, {peak_memory_delta:.2f}MB peak")
    
    def benchmark_tensor_operations(self, device: str = "cpu", iterations: int = 10):
        """Benchmark common tensor operations"""
        logger.info(f"Running tensor operations benchmark on {device} ({iterations} iterations)")
        
        # Create test tensors
        size = 1024
        if device == "cuda" and torch.cuda.is_available():
            A = torch.randn(size, size, device="cuda")
            B = torch.randn(size, size, device="cuda")
        elif device == "mps" and torch.backends.mps.is_available():
            A = torch.randn(size, size, device="mps")
            B = torch.randn(size, size, device="mps")
        else:
            A = torch.randn(size, size)
            B = torch.randn(size, size)
            device = "cpu"
        
        # Benchmark matrix multiplication
        for i in range(iterations):
            with self.profile(f"matmul_{device}", device=device, metadata={"size": size, "iteration": i}):
                _ = torch.matmul(A, B)
        
        # Benchmark element-wise operations
        for i in range(iterations):
            with self.profile(f"add_{device}", device=device, metadata={"size": size, "iteration": i}):
                _ = A + B
        
        for i in range(iterations):
            with self.profile(f"multiply_{device}", device=device, metadata={"size": size, "iteration": i}):
                _ = A * B
        
        # Benchmark normalization
        for i in range(iterations):
            with self.profile(f"normalize_{device}", device=device, metadata={"size": size, "iteration": i}):
                _ = torch.nn.functional.layer_norm(A, (size,))
    
    def benchmark_video_generation(
        self,
        pipeline,
        prompt: str,
        iterations: int = 3,
        device: str = "cpu"
    ):
        """Benchmark video generation pipeline"""
        logger.info(f"Running video generation benchmark ({iterations} iterations)")
        
        for i in range(iterations):
            with self.profile(
                f"video_generation_{device}",
                device=device,
                metadata={"prompt": prompt[:50], "iteration": i}
            ):
                with torch.inference_mode():
                    result = pipeline(prompt, num_inference_steps=10)
    
    def generate_report(self, suite_name: str = "benchmark") -> BenchmarkSuite:
        """Generate comprehensive benchmark report"""
        if not self.benchmarks:
            logger.warning("No benchmarks recorded")
            return BenchmarkSuite(
                suite_name=suite_name,
                timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
                platform=self._get_platform_info(),
                results=[],
                summary={}
            )
        
        # Group by operation name
        grouped = {}
        for result in self.benchmarks:
            base_name = result.name.rsplit('_', 1)[0] if '_' in result.name else result.name
            if base_name not in grouped:
                grouped[base_name] = []
            grouped[base_name].append(result)
        
        # Calculate summary statistics
        summary = {}
        for op_name, results in grouped.items():
            durations = [r.duration_ms for r in results]
            memory_peaks = [r.memory_peak_mb for r in results]
            
            summary[f"{op_name}_mean_ms"] = statistics.mean(durations)
            summary[f"{op_name}_median_ms"] = statistics.median(durations)
            summary[f"{op_name}_std_ms"] = statistics.stdev(durations) if len(durations) > 1 else 0
            summary[f"{op_name}_min_ms"] = min(durations)
            summary[f"{op_name}_max_ms"] = max(durations)
            summary[f"{op_name}_mean_memory_mb"] = statistics.mean(memory_peaks)
        
        suite = BenchmarkSuite(
            suite_name=suite_name,
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
            platform=self._get_platform_info(),
            results=self.benchmarks,
            summary=summary
        )
        
        # Save report
        self._save_report(suite)
        
        return suite
    
    def _get_platform_info(self) -> str:
        """Get platform information"""
        import platform
        info = {
            "system": platform.system(),
            "machine": platform.machine(),
            "processor": platform.processor(),
        }
        
        # Try to get CPU info on macOS
        if platform.system() == "Darwin":
            try:
                import subprocess
                cpu_info = subprocess.check_output(['sysctl', '-n', 'machdep.cpu.brand_string']).decode().strip()
                info["cpu"] = cpu_info
            except Exception:
                pass
        
        return json.dumps(info)
    
    def _save_report(self, suite: BenchmarkSuite):
        """Save benchmark report to file"""
        filename = f"{suite.suite_name}_{suite.timestamp.replace(':', '-').replace(' ', '_')}.json"
        filepath = self.output_dir / filename
        
        with open(filepath, 'w') as f:
            json.dump(asdict(suite), f, indent=2)
        
        logger.info(f"Saved benchmark report: {filepath}")
        
        # Also print summary
        print("\n" + "="*60)
        print(f"Benchmark Report: {suite.suite_name}")
        print("="*60)
        print(f"Platform: {suite.platform}")
        print(f"Timestamp: {suite.timestamp}")
        print(f"Total benchmarks: {len(suite.results)}")
        print("\nSummary Statistics:")
        for key, value in suite.summary.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.2f}")
            else:
                print(f"  {key}: {value}")
        print("="*60 + "\n")


class M4BenchmarkRunner:
    """
    Comprehensive benchmark runner for M4 optimizations
    """
    
    def __init__(self):
        self.profiler = M4PerformanceProfiler()
    
    def run_full_suite(self):
        """Run full benchmark suite"""
        logger.info("Starting full M4 benchmark suite")
        
        # Test different devices
        devices = []
        if torch.cuda.is_available():
            devices.append("cuda")
        if torch.backends.mps.is_available():
            devices.append("mps")
        devices.append("cpu")
        
        # Benchmark tensor operations on each device
        for device in devices:
            self.profiler.benchmark_tensor_operations(device=device, iterations=10)
        
        # Generate report
        suite = self.profiler.generate_report("m4_full_suite")
        
        return suite
    
    def compare_devices(self, operation_name: str, operation_func, *args, **kwargs):
        """Compare performance across different devices"""
        results = {}
        
        devices = []
        if torch.cuda.is_available():
            devices.append("cuda")
        if torch.backends.mps.is_available():
            devices.append("mps")
        devices.append("cpu")
        
        for device in devices:
            durations = []
            for _ in range(5):
                with self.profiler.profile(f"{operation_name}_{device}", device=device):
                    operation_func(*args, device=device, **kwargs)
                durations.append(self.profiler.benchmarks[-1].duration_ms)
            
            results[device] = {
                "mean": statistics.mean(durations),
                "std": statistics.stdev(durations) if len(durations) > 1 else 0,
                "min": min(durations),
                "max": max(durations)
            }
        
        return results


# Global profiler instance
_profiler: Optional[M4PerformanceProfiler] = None


def get_profiler() -> M4PerformanceProfiler:
    """Get or create profiler instance"""
    global _profiler
    if _profiler is None:
        _profiler = M4PerformanceProfiler()
    return _profiler


def benchmark_operation(name: str, operation, device: str = "cpu", *args, **kwargs):
    """Convenience function to benchmark an operation"""
    profiler = get_profiler()
    with profiler.profile(name, device=device):
        return operation(*args, **kwargs)
