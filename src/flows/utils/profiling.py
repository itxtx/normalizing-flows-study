"""
Performance profiling and benchmarking utilities for normalizing flows.

This module provides comprehensive tools for measuring and analyzing the
performance characteristics of normalizing flows, including timing,
throughput, and computational efficiency metrics.
"""

import torch
import torch.nn as nn
import time
import statistics
from typing import Dict, List, Optional, Tuple, Any, Callable, Union
from contextlib import contextmanager
from collections import defaultdict
import json
import numpy as np
from dataclasses import dataclass, asdict

from ..flow.flow import Flow


@dataclass
class PerformanceMetrics:
    """Data class for storing performance metrics."""
    forward_time: float
    inverse_time: float
    forward_throughput: float  # samples/second
    inverse_throughput: float  # samples/second
    memory_usage: Dict[str, int]
    flops: Optional[int] = None
    parameters: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


class FlowProfiler:
    """
    Comprehensive profiler for normalizing flows.
    
    This class provides detailed performance analysis including timing,
    throughput, memory usage, and computational complexity metrics.
    """
    
    def __init__(self, warmup_iterations: int = 5, measurement_iterations: int = 20):
        self.warmup_iterations = warmup_iterations
        self.measurement_iterations = measurement_iterations
        
        # Profiling results storage
        self.results = {}
        self.detailed_timings = defaultdict(list)
        
        # CUDA events for precise GPU timing
        if torch.cuda.is_available():
            self.start_event = torch.cuda.Event(enable_timing=True)
            self.end_event = torch.cuda.Event(enable_timing=True)
        else:
            self.start_event = None
            self.end_event = None
    
    def profile_flow(
        self,
        flow: Flow,
        input_shape: Tuple[int, ...],
        batch_sizes: List[int] = [1, 8, 32, 64],
        device: str = 'cpu',
        include_backward: bool = True
    ) -> Dict[str, PerformanceMetrics]:
        """
        Profile a flow across different batch sizes.
        
        Args:
            flow: The flow to profile
            input_shape: Shape of input tensors (excluding batch dimension)
            batch_sizes: List of batch sizes to test
            device: Device to run profiling on
            include_backward: Whether to include backward pass timing
            
        Returns:
            Dictionary mapping batch sizes to performance metrics
        """
        flow = flow.to(device)
        flow.eval()
        
        results = {}
        
        for batch_size in batch_sizes:
            print(f"Profiling batch size {batch_size}...")
            
            full_input_shape = (batch_size,) + input_shape
            metrics = self._profile_single_batch_size(
                flow, full_input_shape, device, include_backward
            )
            
            results[batch_size] = metrics
        
        self.results.update(results)
        return results
    
    def _profile_single_batch_size(
        self,
        flow: Flow,
        input_shape: Tuple[int, ...],
        device: str,
        include_backward: bool
    ) -> PerformanceMetrics:
        """Profile flow for a single batch size."""
        batch_size = input_shape[0]
        
        # Warmup
        for _ in range(self.warmup_iterations):
            x = torch.randn(input_shape, device=device)
            with torch.no_grad():
                flow.forward(x)
                flow.inverse(x)
        
        # Clear cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        # Measure forward pass
        forward_times = []
        for _ in range(self.measurement_iterations):
            x = torch.randn(input_shape, device=device)
            
            start_time = self._get_time()
            with torch.no_grad():
                output, log_det = flow.forward(x)
            self._synchronize()
            end_time = self._get_time()
            
            forward_times.append(end_time - start_time)
        
        # Measure inverse pass
        inverse_times = []
        for _ in range(self.measurement_iterations):
            x = torch.randn(input_shape, device=device)
            
            start_time = self._get_time()
            with torch.no_grad():
                z, log_det = flow.inverse(x)
            self._synchronize()
            end_time = self._get_time()
            
            inverse_times.append(end_time - start_time)
        
        # Calculate statistics
        avg_forward_time = statistics.mean(forward_times)
        avg_inverse_time = statistics.mean(inverse_times)
        
        forward_throughput = batch_size / avg_forward_time
        inverse_throughput = batch_size / avg_inverse_time
        
        # Memory usage
        memory_usage = self._measure_memory_usage(flow, input_shape, device)
        
        # Parameter count
        param_count = sum(p.numel() for p in flow.parameters())
        
        # Store detailed timings
        self.detailed_timings[f'forward_{batch_size}'] = forward_times
        self.detailed_timings[f'inverse_{batch_size}'] = inverse_times
        
        return PerformanceMetrics(
            forward_time=avg_forward_time,
            inverse_time=avg_inverse_time,
            forward_throughput=forward_throughput,
            inverse_throughput=inverse_throughput,
            memory_usage=memory_usage,
            parameters=param_count
        )
    
    def _get_time(self) -> float:
        """Get current time with appropriate precision."""
        if torch.cuda.is_available() and self.start_event is not None:
            self.start_event.record()
            return 0.0  # Will be calculated in _synchronize
        else:
            return time.perf_counter()
    
    def _synchronize(self):
        """Synchronize timing measurements."""
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    
    def _measure_memory_usage(
        self, 
        flow: Flow, 
        input_shape: Tuple[int, ...], 
        device: str
    ) -> Dict[str, int]:
        """Measure memory usage during forward and inverse passes."""
        if not torch.cuda.is_available() or device == 'cpu':
            return {'gpu_allocated': 0, 'gpu_reserved': 0}
        
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        x = torch.randn(input_shape, device=device)
        
        # Forward pass
        output, log_det = flow.forward(x)
        forward_memory = torch.cuda.max_memory_allocated()
        
        # Inverse pass
        z, log_det_inv = flow.inverse(x)
        total_memory = torch.cuda.max_memory_allocated()
        
        return {
            'gpu_allocated': total_memory,
            'gpu_reserved': torch.cuda.max_memory_reserved(),
            'forward_peak': forward_memory
        }
    
    def get_timing_statistics(self, batch_size: int) -> Dict[str, Dict[str, float]]:
        """Get detailed timing statistics for a specific batch size."""
        forward_key = f'forward_{batch_size}'
        inverse_key = f'inverse_{batch_size}'
        
        stats = {}
        
        if forward_key in self.detailed_timings:
            times = self.detailed_timings[forward_key]
            stats['forward'] = {
                'mean': statistics.mean(times),
                'median': statistics.median(times),
                'std': statistics.stdev(times) if len(times) > 1 else 0.0,
                'min': min(times),
                'max': max(times),
                'p95': np.percentile(times, 95),
                'p99': np.percentile(times, 99)
            }
        
        if inverse_key in self.detailed_timings:
            times = self.detailed_timings[inverse_key]
            stats['inverse'] = {
                'mean': statistics.mean(times),
                'median': statistics.median(times),
                'std': statistics.stdev(times) if len(times) > 1 else 0.0,
                'min': min(times),
                'max': max(times),
                'p95': np.percentile(times, 95),
                'p99': np.percentile(times, 99)
            }
        
        return stats
    
    def export_results(self, filename: str):
        """Export profiling results to JSON file."""
        export_data = {
            'results': {
                str(k): v.to_dict() for k, v in self.results.items()
            },
            'detailed_timings': {
                k: list(v) for k, v in self.detailed_timings.items()
            },
            'profiler_config': {
                'warmup_iterations': self.warmup_iterations,
                'measurement_iterations': self.measurement_iterations
            }
        }
        
        with open(filename, 'w') as f:
            json.dump(export_data, f, indent=2)


class BenchmarkSuite:
    """
    Comprehensive benchmark suite for comparing multiple flows.
    
    This class provides standardized benchmarks for comparing the performance
    of different normalizing flow architectures.
    """
    
    def __init__(self):
        self.benchmark_results = {}
        self.profiler = FlowProfiler()
    
    def add_flow(self, name: str, flow: Flow):
        """Add a flow to the benchmark suite."""
        self.flows = getattr(self, 'flows', {})
        self.flows[name] = flow
    
    def run_benchmark(
        self,
        input_shape: Tuple[int, ...],
        batch_sizes: List[int] = [1, 8, 32, 64],
        device: str = 'cpu',
        include_backward: bool = False
    ) -> Dict[str, Dict[str, PerformanceMetrics]]:
        """
        Run benchmark on all registered flows.
        
        Args:
            input_shape: Shape of input tensors (excluding batch dimension)
            batch_sizes: List of batch sizes to test
            device: Device to run benchmarks on
            include_backward: Whether to include backward pass timing
            
        Returns:
            Dictionary mapping flow names to their performance metrics
        """
        if not hasattr(self, 'flows') or not self.flows:
            raise ValueError("No flows registered. Use add_flow() to add flows.")
        
        results = {}
        
        for flow_name, flow in self.flows.items():
            print(f"\nBenchmarking {flow_name}...")
            
            try:
                flow_results = self.profiler.profile_flow(
                    flow, input_shape, batch_sizes, device, include_backward
                )
                results[flow_name] = flow_results
                
            except Exception as e:
                print(f"Error benchmarking {flow_name}: {e}")
                results[flow_name] = {}
        
        self.benchmark_results = results
        return results
    
    def compare_flows(
        self, 
        metric: str = 'forward_throughput',
        batch_size: int = 32
    ) -> Dict[str, float]:
        """
        Compare flows based on a specific metric.
        
        Args:
            metric: Metric to compare ('forward_time', 'inverse_time', 
                   'forward_throughput', 'inverse_throughput', 'memory_usage')
            batch_size: Batch size to compare
            
        Returns:
            Dictionary mapping flow names to metric values
        """
        if not self.benchmark_results:
            raise ValueError("No benchmark results available. Run run_benchmark() first.")
        
        comparison = {}
        
        for flow_name, results in self.benchmark_results.items():
            if batch_size in results:
                metrics = results[batch_size]
                
                if metric == 'memory_usage':
                    # For memory usage, return total GPU memory
                    value = metrics.memory_usage.get('gpu_allocated', 0)
                else:
                    value = getattr(metrics, metric, 0)
                
                comparison[flow_name] = value
        
        return comparison
    
    def get_ranking(
        self, 
        metric: str = 'forward_throughput',
        batch_size: int = 32,
        ascending: bool = False
    ) -> List[Tuple[str, float]]:
        """
        Get ranking of flows based on a metric.
        
        Args:
            metric: Metric to rank by
            batch_size: Batch size to use for ranking
            ascending: Whether to sort in ascending order (lower is better)
            
        Returns:
            List of (flow_name, metric_value) tuples sorted by performance
        """
        comparison = self.compare_flows(metric, batch_size)
        
        return sorted(
            comparison.items(),
            key=lambda x: x[1],
            reverse=not ascending
        )
    
    def generate_report(self) -> str:
        """Generate a comprehensive benchmark report."""
        if not self.benchmark_results:
            return "No benchmark results available."
        
        report = ["Normalizing Flow Benchmark Report", "=" * 40, ""]
        
        # Summary table
        report.append("Performance Summary (batch_size=32):")
        report.append("-" * 50)
        
        batch_size = 32
        if any(batch_size in results for results in self.benchmark_results.values()):
            
            # Forward throughput ranking
            throughput_ranking = self.get_ranking('forward_throughput', batch_size)
            report.append("\nForward Throughput (samples/sec):")
            for i, (name, value) in enumerate(throughput_ranking, 1):
                report.append(f"{i:2d}. {name:20s}: {value:8.1f}")
            
            # Memory usage ranking
            memory_ranking = self.get_ranking('memory_usage', batch_size, ascending=True)
            report.append("\nMemory Usage (bytes):")
            for i, (name, value) in enumerate(memory_ranking, 1):
                report.append(f"{i:2d}. {name:20s}: {value:12,d}")
            
            # Parameter count
            report.append("\nParameter Count:")
            for name, results in self.benchmark_results.items():
                if batch_size in results:
                    params = results[batch_size].parameters
                    report.append(f"    {name:20s}: {params:12,d}")
        
        return "\n".join(report)
    
    def export_benchmark(self, filename: str):
        """Export benchmark results to JSON file."""
        export_data = {
            'benchmark_results': {
                flow_name: {
                    str(batch_size): metrics.to_dict()
                    for batch_size, metrics in results.items()
                }
                for flow_name, results in self.benchmark_results.items()
            },
            'timestamp': time.time()
        }
        
        with open(filename, 'w') as f:
            json.dump(export_data, f, indent=2)


def profile_flow_performance(
    flow: Flow,
    input_shape: Tuple[int, ...],
    batch_size: int = 32,
    device: str = 'cpu',
    num_iterations: int = 100
) -> Dict[str, Any]:
    """
    Quick performance profiling function for a single flow.
    
    Args:
        flow: The flow to profile
        input_shape: Shape of input tensors (excluding batch dimension)
        batch_size: Batch size to use
        device: Device to run on
        num_iterations: Number of iterations for timing
        
    Returns:
        Dictionary containing performance metrics
    """
    profiler = FlowProfiler(
        warmup_iterations=10,
        measurement_iterations=num_iterations
    )
    
    full_input_shape = (batch_size,) + input_shape
    results = profiler._profile_single_batch_size(
        flow, full_input_shape, device, include_backward=False
    )
    
    # Add timing statistics
    timing_stats = profiler.get_timing_statistics(batch_size)
    
    return {
        'performance_metrics': results.to_dict(),
        'timing_statistics': timing_stats,
        'profiler_config': {
            'batch_size': batch_size,
            'input_shape': input_shape,
            'device': device,
            'num_iterations': num_iterations
        }
    }


def compare_flow_performance(
    flows: Dict[str, Flow],
    input_shape: Tuple[int, ...],
    batch_sizes: List[int] = [1, 8, 32, 64],
    device: str = 'cpu'
) -> Dict[str, Any]:
    """
    Compare performance of multiple flows.
    
    Args:
        flows: Dictionary mapping flow names to Flow objects
        input_shape: Shape of input tensors (excluding batch dimension)
        batch_sizes: List of batch sizes to test
        device: Device to run on
        
    Returns:
        Dictionary containing comparison results
    """
    benchmark = BenchmarkSuite()
    
    # Add flows to benchmark
    for name, flow in flows.items():
        benchmark.add_flow(name, flow)
    
    # Run benchmark
    results = benchmark.run_benchmark(input_shape, batch_sizes, device)
    
    # Generate comparison report
    report = benchmark.generate_report()
    
    return {
        'benchmark_results': results,
        'comparison_report': report,
        'rankings': {
            'forward_throughput': benchmark.get_ranking('forward_throughput', 32),
            'memory_usage': benchmark.get_ranking('memory_usage', 32, ascending=True)
        }
    }


@contextmanager
def profile_context(name: str = "operation"):
    """
    Context manager for profiling code blocks.
    
    Args:
        name: Name of the operation being profiled
        
    Yields:
        Dictionary that will be populated with timing results
    """
    results = {}
    
    # GPU timing setup
    if torch.cuda.is_available():
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        torch.cuda.synchronize()
        start_event.record()
        start_time = time.perf_counter()
    else:
        start_time = time.perf_counter()
        start_event = None
        end_event = None
    
    try:
        yield results
    finally:
        if torch.cuda.is_available() and start_event is not None:
            end_event.record()
            torch.cuda.synchronize()
            gpu_time = start_event.elapsed_time(end_event) / 1000.0  # Convert to seconds
            results['gpu_time'] = gpu_time
        
        end_time = time.perf_counter()
        results['cpu_time'] = end_time - start_time
        results['operation_name'] = name
        
        print(f"Profile [{name}]: CPU time = {results['cpu_time']:.4f}s", end="")
        if 'gpu_time' in results:
            print(f", GPU time = {results['gpu_time']:.4f}s")
        else:
            print()


class PerformanceRegression:
    """
    Tool for detecting performance regressions in flows.
    
    This class helps track performance over time and detect when
    performance degrades significantly.
    """
    
    def __init__(self, baseline_file: Optional[str] = None):
        self.baseline_file = baseline_file
        self.baseline_results = {}
        
        if baseline_file:
            self.load_baseline(baseline_file)
    
    def load_baseline(self, filename: str):
        """Load baseline performance results."""
        try:
            with open(filename, 'r') as f:
                self.baseline_results = json.load(f)
        except FileNotFoundError:
            print(f"Baseline file {filename} not found. Will create new baseline.")
    
    def save_baseline(self, results: Dict[str, Any], filename: str):
        """Save current results as baseline."""
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)
        
        self.baseline_results = results
        print(f"Baseline saved to {filename}")
    
    def check_regression(
        self,
        current_results: Dict[str, Any],
        threshold: float = 0.1
    ) -> Dict[str, Any]:
        """
        Check for performance regressions.
        
        Args:
            current_results: Current performance results
            threshold: Regression threshold (e.g., 0.1 = 10% slower)
            
        Returns:
            Dictionary containing regression analysis
        """
        if not self.baseline_results:
            return {'error': 'No baseline results available'}
        
        regressions = []
        improvements = []
        
        # Compare throughput metrics
        for flow_name in current_results.get('benchmark_results', {}):
            if flow_name not in self.baseline_results.get('benchmark_results', {}):
                continue
            
            current_flow = current_results['benchmark_results'][flow_name]
            baseline_flow = self.baseline_results['benchmark_results'][flow_name]
            
            for batch_size in current_flow:
                if batch_size not in baseline_flow:
                    continue
                
                current_metrics = current_flow[batch_size]
                baseline_metrics = baseline_flow[batch_size]
                
                # Check forward throughput
                current_throughput = current_metrics.get('forward_throughput', 0)
                baseline_throughput = baseline_metrics.get('forward_throughput', 0)
                
                if baseline_throughput > 0:
                    change_ratio = (current_throughput - baseline_throughput) / baseline_throughput
                    
                    if change_ratio < -threshold:  # Significant slowdown
                        regressions.append({
                            'flow': flow_name,
                            'batch_size': batch_size,
                            'metric': 'forward_throughput',
                            'baseline': baseline_throughput,
                            'current': current_throughput,
                            'change_percent': change_ratio * 100
                        })
                    elif change_ratio > threshold:  # Significant improvement
                        improvements.append({
                            'flow': flow_name,
                            'batch_size': batch_size,
                            'metric': 'forward_throughput',
                            'baseline': baseline_throughput,
                            'current': current_throughput,
                            'change_percent': change_ratio * 100
                        })
        
        return {
            'regressions_detected': len(regressions) > 0,
            'regressions': regressions,
            'improvements': improvements,
            'threshold_percent': threshold * 100
        }