"""
Memory profiling and optimization utilities for normalizing flows.

This module provides tools for monitoring memory usage, detecting memory leaks,
and optimizing memory consumption during flow training and inference.
"""

import torch
import torch.nn as nn
import gc
import psutil
import time
from typing import Dict, List, Optional, Tuple, Any, Callable, Union
from contextlib import contextmanager
from collections import defaultdict, deque
import threading
import warnings

from ..flow.flow import Flow


class MemoryProfiler:
    """
    A comprehensive memory profiler for normalizing flows.
    
    This class provides detailed memory usage tracking including GPU memory,
    system memory, and memory allocation patterns.
    """
    
    def __init__(self, track_allocations: bool = True, max_snapshots: int = 100):
        self.track_allocations = track_allocations
        self.max_snapshots = max_snapshots
        
        # Memory tracking data
        self.snapshots = deque(maxlen=max_snapshots)
        self.allocation_history = []
        self.peak_memory = {
            'gpu': 0,
            'cpu': 0,
            'system': 0
        }
        
        # Profiling state
        self.is_profiling = False
        self.start_time = None
        
        # Thread for continuous monitoring
        self._monitor_thread = None
        self._stop_monitoring = threading.Event()
    
    def start_profiling(self, continuous: bool = False, interval: float = 0.1):
        """
        Start memory profiling.
        
        Args:
            continuous: Whether to continuously monitor memory usage
            interval: Monitoring interval in seconds (for continuous mode)
        """
        if self.is_profiling:
            warnings.warn("Profiling is already active", UserWarning)
            return
        
        self.is_profiling = True
        self.start_time = time.time()
        
        # Reset tracking data
        self.snapshots.clear()
        self.allocation_history.clear()
        self.peak_memory = {'gpu': 0, 'cpu': 0, 'system': 0}
        
        # Take initial snapshot
        self._take_snapshot()
        
        if continuous:
            self._start_continuous_monitoring(interval)
    
    def stop_profiling(self) -> Dict[str, Any]:
        """
        Stop memory profiling and return summary.
        
        Returns:
            Dictionary containing profiling results
        """
        if not self.is_profiling:
            warnings.warn("Profiling is not active", UserWarning)
            return {}
        
        self.is_profiling = False
        
        # Stop continuous monitoring if active
        if self._monitor_thread is not None:
            self._stop_monitoring.set()
            self._monitor_thread.join()
            self._monitor_thread = None
            self._stop_monitoring.clear()
        
        # Take final snapshot
        self._take_snapshot()
        
        return self.get_summary()
    
    def _start_continuous_monitoring(self, interval: float):
        """Start continuous memory monitoring in a separate thread."""
        def monitor():
            while not self._stop_monitoring.wait(interval):
                if self.is_profiling:
                    self._take_snapshot()
        
        self._monitor_thread = threading.Thread(target=monitor, daemon=True)
        self._monitor_thread.start()
    
    def _take_snapshot(self):
        """Take a memory usage snapshot."""
        timestamp = time.time()
        
        # GPU memory
        gpu_memory = 0
        gpu_allocated = 0
        gpu_cached = 0
        
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.memory_allocated()
            gpu_allocated = torch.cuda.memory_allocated()
            gpu_cached = torch.cuda.memory_reserved()
        
        # CPU memory (PyTorch tensors)
        cpu_memory = 0
        for obj in gc.get_objects():
            if isinstance(obj, torch.Tensor) and not obj.is_cuda:
                cpu_memory += obj.element_size() * obj.nelement()
        
        # System memory
        process = psutil.Process()
        system_memory = process.memory_info().rss
        
        snapshot = {
            'timestamp': timestamp,
            'gpu_allocated': gpu_allocated,
            'gpu_cached': gpu_cached,
            'cpu_memory': cpu_memory,
            'system_memory': system_memory,
            'total_memory': gpu_allocated + cpu_memory
        }
        
        self.snapshots.append(snapshot)
        
        # Update peak memory
        self.peak_memory['gpu'] = max(self.peak_memory['gpu'], gpu_allocated)
        self.peak_memory['cpu'] = max(self.peak_memory['cpu'], cpu_memory)
        self.peak_memory['system'] = max(self.peak_memory['system'], system_memory)
    
    @contextmanager
    def profile_block(self, name: str):
        """Context manager for profiling a specific code block."""
        start_snapshot = self._get_current_memory()
        start_time = time.time()
        
        try:
            yield
        finally:
            end_time = time.time()
            end_snapshot = self._get_current_memory()
            
            memory_delta = {
                key: end_snapshot[key] - start_snapshot[key]
                for key in start_snapshot.keys()
            }
            
            self.allocation_history.append({
                'name': name,
                'start_time': start_time,
                'end_time': end_time,
                'duration': end_time - start_time,
                'memory_delta': memory_delta,
                'peak_memory_during': self._get_peak_memory_since(start_time)
            })
    
    def _get_current_memory(self) -> Dict[str, int]:
        """Get current memory usage."""
        gpu_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
        
        cpu_memory = 0
        for obj in gc.get_objects():
            if isinstance(obj, torch.Tensor) and not obj.is_cuda:
                cpu_memory += obj.element_size() * obj.nelement()
        
        system_memory = psutil.Process().memory_info().rss
        
        return {
            'gpu_allocated': gpu_memory,
            'cpu_memory': cpu_memory,
            'system_memory': system_memory
        }
    
    def _get_peak_memory_since(self, timestamp: float) -> Dict[str, int]:
        """Get peak memory usage since a given timestamp."""
        relevant_snapshots = [
            s for s in self.snapshots if s['timestamp'] >= timestamp
        ]
        
        if not relevant_snapshots:
            return {'gpu_allocated': 0, 'cpu_memory': 0, 'system_memory': 0}
        
        return {
            'gpu_allocated': max(s['gpu_allocated'] for s in relevant_snapshots),
            'cpu_memory': max(s['cpu_memory'] for s in relevant_snapshots),
            'system_memory': max(s['system_memory'] for s in relevant_snapshots)
        }
    
    def get_summary(self) -> Dict[str, Any]:
        """Get comprehensive profiling summary."""
        if not self.snapshots:
            return {'error': 'No profiling data available'}
        
        first_snapshot = self.snapshots[0]
        last_snapshot = self.snapshots[-1]
        
        total_duration = last_snapshot['timestamp'] - first_snapshot['timestamp']
        
        # Memory usage statistics
        gpu_usage = [s['gpu_allocated'] for s in self.snapshots]
        cpu_usage = [s['cpu_memory'] for s in self.snapshots]
        system_usage = [s['system_memory'] for s in self.snapshots]
        
        summary = {
            'profiling_duration': total_duration,
            'num_snapshots': len(self.snapshots),
            'peak_memory': self.peak_memory.copy(),
            'memory_stats': {
                'gpu': {
                    'initial': first_snapshot['gpu_allocated'],
                    'final': last_snapshot['gpu_allocated'],
                    'peak': max(gpu_usage),
                    'average': sum(gpu_usage) / len(gpu_usage),
                    'delta': last_snapshot['gpu_allocated'] - first_snapshot['gpu_allocated']
                },
                'cpu': {
                    'initial': first_snapshot['cpu_memory'],
                    'final': last_snapshot['cpu_memory'],
                    'peak': max(cpu_usage),
                    'average': sum(cpu_usage) / len(cpu_usage),
                    'delta': last_snapshot['cpu_memory'] - first_snapshot['cpu_memory']
                },
                'system': {
                    'initial': first_snapshot['system_memory'],
                    'final': last_snapshot['system_memory'],
                    'peak': max(system_usage),
                    'average': sum(system_usage) / len(system_usage),
                    'delta': last_snapshot['system_memory'] - first_snapshot['system_memory']
                }
            },
            'allocation_history': self.allocation_history.copy()
        }
        
        return summary
    
    def get_memory_timeline(self) -> List[Dict[str, Any]]:
        """Get memory usage timeline."""
        return list(self.snapshots)
    
    def format_memory_size(self, size_bytes: int) -> str:
        """Format memory size in human-readable format."""
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size_bytes < 1024.0:
                return f"{size_bytes:.2f} {unit}"
            size_bytes /= 1024.0
        return f"{size_bytes:.2f} TB"


class MemoryOptimizer:
    """
    Memory optimization utilities for normalizing flows.
    
    This class provides automatic memory optimization suggestions and
    implementations for reducing memory usage.
    """
    
    def __init__(self):
        self.optimization_history = []
    
    def analyze_flow_memory(self, flow: Flow, input_shape: Tuple[int, ...]) -> Dict[str, Any]:
        """
        Analyze memory usage patterns of a flow.
        
        Args:
            flow: The flow to analyze
            input_shape: Shape of input tensors
            
        Returns:
            Dictionary containing memory analysis results
        """
        profiler = MemoryProfiler()
        
        # Create sample input
        device = next(flow.parameters()).device
        sample_input = torch.randn(input_shape, device=device)
        
        analysis = {
            'parameter_memory': self._calculate_parameter_memory(flow),
            'activation_memory': {},
            'gradient_memory': {},
            'optimization_suggestions': []
        }
        
        # Profile forward pass
        profiler.start_profiling()
        
        with profiler.profile_block('forward_pass'):
            flow.train()
            output, log_det = flow.forward(sample_input)
        
        # Profile backward pass
        with profiler.profile_block('backward_pass'):
            loss = output.sum() + log_det.sum()
            loss.backward()
        
        profiling_results = profiler.stop_profiling()
        
        # Extract memory usage for different phases
        for block in profiling_results.get('allocation_history', []):
            if block['name'] == 'forward_pass':
                analysis['activation_memory'] = block['memory_delta']
            elif block['name'] == 'backward_pass':
                analysis['gradient_memory'] = block['memory_delta']
        
        # Generate optimization suggestions
        analysis['optimization_suggestions'] = self._generate_optimization_suggestions(
            analysis, profiling_results
        )
        
        return analysis
    
    def _calculate_parameter_memory(self, flow: Flow) -> Dict[str, int]:
        """Calculate memory usage of model parameters."""
        param_memory = 0
        buffer_memory = 0
        
        for param in flow.parameters():
            param_memory += param.element_size() * param.nelement()
        
        for buffer in flow.buffers():
            buffer_memory += buffer.element_size() * buffer.nelement()
        
        return {
            'parameters': param_memory,
            'buffers': buffer_memory,
            'total': param_memory + buffer_memory
        }
    
    def _generate_optimization_suggestions(
        self, 
        analysis: Dict[str, Any], 
        profiling_results: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Generate memory optimization suggestions."""
        suggestions = []
        
        # Check for high activation memory
        activation_memory = analysis.get('activation_memory', {}).get('gpu_allocated', 0)
        if activation_memory > 100 * 1024 * 1024:  # > 100MB
            suggestions.append({
                'type': 'gradient_checkpointing',
                'priority': 'high',
                'description': 'High activation memory detected. Consider using gradient checkpointing.',
                'estimated_savings': '30-70% activation memory',
                'implementation': 'Use CheckpointedFlow or CheckpointedSequentialFlow'
            })
        
        # Check for large parameter memory
        param_memory = analysis.get('parameter_memory', {}).get('total', 0)
        if param_memory > 500 * 1024 * 1024:  # > 500MB
            suggestions.append({
                'type': 'mixed_precision',
                'priority': 'medium',
                'description': 'Large model detected. Mixed precision training can reduce memory usage.',
                'estimated_savings': '40-50% parameter and activation memory',
                'implementation': 'Use MixedPrecisionFlow wrapper'
            })
        
        # Check for memory leaks
        memory_delta = profiling_results.get('memory_stats', {}).get('gpu', {}).get('delta', 0)
        if memory_delta > 10 * 1024 * 1024:  # > 10MB increase
            suggestions.append({
                'type': 'memory_leak',
                'priority': 'high',
                'description': 'Potential memory leak detected. Memory usage increased significantly.',
                'estimated_savings': 'Prevent unbounded memory growth',
                'implementation': 'Check for unreleased tensors and use proper context managers'
            })
        
        # Check for inefficient operations
        if len(profiling_results.get('allocation_history', [])) > 10:
            suggestions.append({
                'type': 'operation_fusion',
                'priority': 'low',
                'description': 'Many memory allocations detected. Consider fusing operations.',
                'estimated_savings': '10-20% memory overhead',
                'implementation': 'Use in-place operations where possible'
            })
        
        return suggestions
    
    def apply_optimizations(
        self, 
        flow: Flow, 
        optimizations: List[str],
        **kwargs
    ) -> Flow:
        """
        Apply memory optimizations to a flow.
        
        Args:
            flow: The flow to optimize
            optimizations: List of optimization types to apply
            **kwargs: Additional arguments for specific optimizations
            
        Returns:
            Optimized flow
        """
        optimized_flow = flow
        
        for opt_type in optimizations:
            if opt_type == 'gradient_checkpointing':
                from ..optimization.gradient_checkpointing import apply_gradient_checkpointing
                optimized_flow = apply_gradient_checkpointing(
                    optimized_flow,
                    checkpoint_segments=kwargs.get('checkpoint_segments', None)
                )
                
            elif opt_type == 'mixed_precision':
                from ..optimization.mixed_precision import apply_mixed_precision
                optimized_flow = apply_mixed_precision(
                    optimized_flow,
                    enabled=kwargs.get('mixed_precision_enabled', True)
                )
                
            elif opt_type == 'inplace_operations':
                optimized_flow = self._apply_inplace_optimizations(optimized_flow)
        
        # Record optimization history
        self.optimization_history.append({
            'timestamp': time.time(),
            'optimizations': optimizations,
            'kwargs': kwargs
        })
        
        return optimized_flow
    
    def _apply_inplace_optimizations(self, flow: Flow) -> Flow:
        """Apply in-place operation optimizations where safe."""
        # This is a placeholder for more sophisticated in-place optimizations
        # In practice, this would require careful analysis of the flow structure
        # to ensure mathematical correctness is preserved
        
        warnings.warn(
            "In-place optimizations are not yet implemented. "
            "This requires careful analysis to ensure correctness.",
            UserWarning
        )
        
        return flow


def track_memory_usage(func: Callable) -> Callable:
    """
    Decorator for tracking memory usage of a function.
    
    Args:
        func: Function to track
        
    Returns:
        Wrapped function that tracks memory usage
    """
    def wrapper(*args, **kwargs):
        profiler = MemoryProfiler()
        profiler.start_profiling()
        
        try:
            with profiler.profile_block(func.__name__):
                result = func(*args, **kwargs)
        finally:
            summary = profiler.stop_profiling()
            
            # Store memory usage info in function attribute
            if not hasattr(wrapper, 'memory_usage_history'):
                wrapper.memory_usage_history = []
            
            wrapper.memory_usage_history.append(summary)
        
        return result
    
    return wrapper


def detect_memory_leaks(
    flow: Flow,
    input_shape: Tuple[int, ...],
    num_iterations: int = 10,
    threshold_mb: float = 10.0
) -> Dict[str, Any]:
    """
    Detect potential memory leaks in a flow.
    
    Args:
        flow: The flow to test
        input_shape: Shape of input tensors
        num_iterations: Number of iterations to run
        threshold_mb: Memory increase threshold in MB to flag as potential leak
        
    Returns:
        Dictionary containing leak detection results
    """
    profiler = MemoryProfiler()
    device = next(flow.parameters()).device
    
    memory_snapshots = []
    
    profiler.start_profiling()
    
    for i in range(num_iterations):
        # Create fresh input for each iteration
        sample_input = torch.randn(input_shape, device=device)
        
        with profiler.profile_block(f'iteration_{i}'):
            # Forward pass
            output, log_det = flow.forward(sample_input)
            
            # Backward pass
            loss = output.sum() + log_det.sum()
            loss.backward()
            
            # Clear gradients
            flow.zero_grad()
            
            # Force garbage collection
            del sample_input, output, log_det, loss
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # Take memory snapshot
        current_memory = profiler._get_current_memory()
        memory_snapshots.append(current_memory)
    
    profiling_results = profiler.stop_profiling()
    
    # Analyze memory trend
    gpu_memory_trend = [s['gpu_allocated'] for s in memory_snapshots]
    cpu_memory_trend = [s['cpu_memory'] for s in memory_snapshots]
    
    # Calculate memory increase
    initial_gpu = gpu_memory_trend[0] if gpu_memory_trend else 0
    final_gpu = gpu_memory_trend[-1] if gpu_memory_trend else 0
    gpu_increase_mb = (final_gpu - initial_gpu) / (1024 * 1024)
    
    initial_cpu = cpu_memory_trend[0] if cpu_memory_trend else 0
    final_cpu = cpu_memory_trend[-1] if cpu_memory_trend else 0
    cpu_increase_mb = (final_cpu - initial_cpu) / (1024 * 1024)
    
    # Determine if leak is detected
    gpu_leak_detected = gpu_increase_mb > threshold_mb
    cpu_leak_detected = cpu_increase_mb > threshold_mb
    
    return {
        'leak_detected': gpu_leak_detected or cpu_leak_detected,
        'gpu_leak': gpu_leak_detected,
        'cpu_leak': cpu_leak_detected,
        'gpu_increase_mb': gpu_increase_mb,
        'cpu_increase_mb': cpu_increase_mb,
        'memory_snapshots': memory_snapshots,
        'profiling_results': profiling_results,
        'recommendations': _generate_leak_recommendations(
            gpu_leak_detected, cpu_leak_detected, gpu_increase_mb, cpu_increase_mb
        )
    }


def _generate_leak_recommendations(
    gpu_leak: bool,
    cpu_leak: bool, 
    gpu_increase: float,
    cpu_increase: float
) -> List[str]:
    """Generate recommendations for addressing memory leaks."""
    recommendations = []
    
    if gpu_leak:
        recommendations.append(
            f"GPU memory leak detected (+{gpu_increase:.1f}MB). "
            "Check for unreleased CUDA tensors and use proper context managers."
        )
    
    if cpu_leak:
        recommendations.append(
            f"CPU memory leak detected (+{cpu_increase:.1f}MB). "
            "Check for unreleased CPU tensors and circular references."
        )
    
    if gpu_leak or cpu_leak:
        recommendations.extend([
            "Use torch.no_grad() for inference to prevent gradient accumulation.",
            "Explicitly delete large tensors when no longer needed.",
            "Call gc.collect() and torch.cuda.empty_cache() periodically.",
            "Check for tensors being stored in global variables or closures."
        ])
    
    return recommendations


def get_memory_summary() -> Dict[str, Any]:
    """
    Get current memory usage summary.
    
    Returns:
        Dictionary containing current memory usage information
    """
    summary = {
        'timestamp': time.time(),
        'gpu_available': torch.cuda.is_available(),
        'system_memory': {}
    }
    
    # System memory info
    process = psutil.Process()
    memory_info = process.memory_info()
    
    summary['system_memory'] = {
        'rss': memory_info.rss,
        'vms': memory_info.vms,
        'percent': process.memory_percent(),
        'available': psutil.virtual_memory().available
    }
    
    # GPU memory info
    if torch.cuda.is_available():
        summary['gpu_memory'] = {}
        for i in range(torch.cuda.device_count()):
            device_props = torch.cuda.get_device_properties(i)
            summary['gpu_memory'][f'device_{i}'] = {
                'name': device_props.name,
                'total_memory': device_props.total_memory,
                'allocated': torch.cuda.memory_allocated(i),
                'reserved': torch.cuda.memory_reserved(i),
                'free': device_props.total_memory - torch.cuda.memory_reserved(i)
            }
    
    # PyTorch tensor memory
    cpu_tensor_memory = 0
    gpu_tensor_memory = 0
    
    for obj in gc.get_objects():
        if isinstance(obj, torch.Tensor):
            tensor_memory = obj.element_size() * obj.nelement()
            if obj.is_cuda:
                gpu_tensor_memory += tensor_memory
            else:
                cpu_tensor_memory += tensor_memory
    
    summary['tensor_memory'] = {
        'cpu': cpu_tensor_memory,
        'gpu': gpu_tensor_memory,
        'total': cpu_tensor_memory + gpu_tensor_memory
    }
    
    return summary