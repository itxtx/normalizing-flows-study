"""
Numerical Stability & Performance Profiling Tests

This module performs comprehensive numerical stability testing and performance benchmarking
for all normalizing flow implementations. It detects NaN/Inf values, exploding gradients,
and performance regressions while logging all incidents as high-priority issues.
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
import json
import os
import time
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

from src.flows import (
    CouplingLayer,
    MaskedAutoregressiveFlow, 
    InverseAutoregressiveFlow,
    SplineCouplingLayer,
    ContinuousFlow
)
from src.models import RealNVP, RealNVPSpline, NormalizingFlowModel

# Global configuration
REPORTS_DIR = Path("reports/stability")
BASELINE_FILE = REPORTS_DIR / "benchmark_baseline.json"
GRADIENT_EXPLOSION_THRESHOLD = 1e3
PERFORMANCE_REGRESSION_MULTIPLIER = 2.0

# Ensure reports directory exists
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def create_mask(dim: int, mask_type: str = "alternating") -> torch.Tensor:
    """Helper function to create masks for coupling layers."""
    mask = torch.zeros(dim)
    if mask_type == "alternating":
        mask[::2] = 1
    elif mask_type == "half":
        mask[:dim//2] = 1
    return mask


def get_all_flows_for_testing() -> List[Tuple[str, nn.Module]]:
    """Get all flow classes with their names for testing."""
    dim = 4
    hidden_dim = 16
    
    # Individual flow layers
    flows = [
        ("CouplingLayer_alternating", CouplingLayer(dim, hidden_dim, create_mask(dim, "alternating"))),
        ("CouplingLayer_half", CouplingLayer(dim, hidden_dim, create_mask(dim, "half"))),
        ("SplineCouplingLayer_alternating", SplineCouplingLayer(dim, hidden_dim, create_mask(dim, "alternating"))),  
        ("SplineCouplingLayer_half", SplineCouplingLayer(dim, hidden_dim, create_mask(dim, "half"))),
        ("MaskedAutoregressiveFlow", MaskedAutoregressiveFlow(dim, hidden_dim)),
        ("InverseAutoregressiveFlow", InverseAutoregressiveFlow(dim, hidden_dim)),
        ("ContinuousFlow", ContinuousFlow(dim, hidden_dim)),
        # Composite models
        ("RealNVP", RealNVP(dim, n_layers=2, hidden_dim=hidden_dim)),
        ("RealNVPSpline", RealNVPSpline(dim, n_layers=2, hidden_dim=hidden_dim)),
        ("NormalizingFlowModel", NormalizingFlowModel([
            CouplingLayer(dim, hidden_dim, create_mask(dim, "alternating")),
            MaskedAutoregressiveFlow(dim, hidden_dim)
        ]))
    ]
    
    return flows


class StressInputGenerator:
    """Generates various stress test inputs for numerical stability testing."""
    
    @staticmethod
    def generate_small_scalars(batch_size: int = 8, dim: int = 4) -> torch.Tensor:
        """Generate small scalar stress inputs (±1e-6)."""
        signs = torch.randint(0, 2, (batch_size, dim)) * 2 - 1  # Random ±1
        return signs * 1e-6
    
    @staticmethod
    def generate_large_scalars(batch_size: int = 8, dim: int = 4) -> torch.Tensor:
        """Generate large scalar stress inputs (±1e3)."""
        signs = torch.randint(0, 2, (batch_size, dim)) * 2 - 1  # Random ±1
        return signs * 1e3
    
    @staticmethod
    def generate_high_dim_tensors(batch_size: int = 8, dim: int = 128) -> torch.Tensor:
        """Generate random high-dimensional tensors."""
        return torch.randn(batch_size, dim)
    
    @staticmethod
    def generate_mixed_precision_batch(batch_size: int = 8, dim: int = 4) -> torch.Tensor:
        """Generate mixed-precision batch using torch.float16."""
        return torch.randn(batch_size, dim, dtype=torch.float16)
    
    @staticmethod
    def generate_edge_cases(batch_size: int = 8, dim: int = 4) -> List[Tuple[str, torch.Tensor]]:
        """Generate various edge case inputs."""
        edge_cases = []
        
        # Zero inputs
        edge_cases.append(("zeros", torch.zeros(batch_size, dim)))
        
        # Very large inputs  
        edge_cases.append(("very_large", torch.full((batch_size, dim), 1e10)))
        
        # Very small inputs
        edge_cases.append(("very_small", torch.full((batch_size, dim), 1e-10)))
        
        # Mixed scale inputs
        mixed = torch.randn(batch_size, dim)
        mixed[:, 0] *= 1e6  # First dimension very large
        mixed[:, 1] *= 1e-6  # Second dimension very small
        edge_cases.append(("mixed_scale", mixed))
        
        # NaN inputs (for testing robustness)
        nan_input = torch.randn(batch_size, dim)
        nan_input[0, 0] = float('nan')
        edge_cases.append(("with_nan", nan_input))
        
        # Inf inputs
        inf_input = torch.randn(batch_size, dim)
        inf_input[0, 0] = float('inf')
        edge_cases.append(("with_inf", inf_input))
        
        return edge_cases


class StabilityChecker:
    """Utilities for checking numerical stability."""
    
    @staticmethod
    def check_finite_tensor(tensor: torch.Tensor, name: str) -> List[str]:
        """Check if tensor contains only finite values."""
        issues = []
        
        if not torch.isfinite(tensor).all():
            nan_count = torch.isnan(tensor).sum().item()
            inf_count = torch.isinf(tensor).sum().item()
            
            if nan_count > 0:
                issues.append(f"**high-priority perf/stability issue** {name} contains {nan_count} NaN values")
            if inf_count > 0:
                issues.append(f"**high-priority perf/stability issue** {name} contains {inf_count} Inf values")
                
        return issues
    
    @staticmethod  
    def check_gradient_explosion(tensor: torch.Tensor, name: str) -> List[str]:
        """Check for exploding gradients."""
        issues = []
        
        if tensor.requires_grad and tensor.grad is not None:
            grad_norm = torch.norm(tensor.grad).item()
            if grad_norm > GRADIENT_EXPLOSION_THRESHOLD:
                issues.append(f"**high-priority perf/stability issue** Exploding gradient in {name}: norm = {grad_norm:.2e}")
                
        return issues
    
    @staticmethod
    def compute_gradient_norm(tensor: torch.Tensor) -> float:
        """Compute gradient norm for a tensor.""" 
        if tensor.requires_grad and tensor.grad is not None:
            return torch.norm(tensor.grad).item()
        return 0.0


class PerformanceBenchmarker:
    """Handles performance benchmarking and regression detection."""
    
    def __init__(self):
        self.baseline_data = self._load_baseline()
        
    def _load_baseline(self) -> Dict[str, float]:
        """Load baseline performance data."""
        if BASELINE_FILE.exists():
            with open(BASELINE_FILE, 'r') as f:
                return json.load(f)
        return {}
    
    def _save_baseline(self, data: Dict[str, float]):
        """Save baseline performance data."""
        with open(BASELINE_FILE, 'w') as f:
            json.dump(data, f, indent=2)
    
    def benchmark_function(self, func, *args, **kwargs) -> Tuple[Any, float]:
        """Benchmark a function call and return result + execution time."""
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        start_time = time.perf_counter()
        
        result = func(*args, **kwargs)
        
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        end_time = time.perf_counter()
        
        return result, end_time - start_time
    
    def check_performance_regression(self, flow_name: str, operation: str, current_time: float) -> List[str]:
        """Check for performance regression against baseline."""
        issues = []
        key = f"{flow_name}_{operation}"
        
        if key in self.baseline_data:
            baseline_time = self.baseline_data[key]
            if current_time > baseline_time * PERFORMANCE_REGRESSION_MULTIPLIER:
                issues.append(
                    f"**high-priority perf/stability issue** Performance regression in {flow_name}.{operation}: "
                    f"{current_time:.4f}s vs baseline {baseline_time:.4f}s "
                    f"({current_time/baseline_time:.1f}x slower)"
                )
        else:
            # New benchmark - save as baseline
            self.baseline_data[key] = current_time
            self._save_baseline(self.baseline_data)
            
        return issues


class StabilityReporter:
    """Handles logging and reporting of stability issues."""
    
    @staticmethod
    def save_report(flow_name: str, report_data: Dict[str, Any]):
        """Save stability report for a flow."""
        report_file = REPORTS_DIR / f"{flow_name}.json"
        
        # Load existing data if present
        existing_data = {}
        if report_file.exists():
            with open(report_file, 'r') as f:
                existing_data = json.load(f)
        
        # Update with new data
        existing_data.update(report_data)
        existing_data['last_updated'] = time.strftime('%Y-%m-%d %H:%M:%S')
        
        # Save updated data
        with open(report_file, 'w') as f:
            json.dump(existing_data, f, indent=2)


@pytest.mark.parametrize("flow_name,flow", get_all_flows_for_testing())
class TestNumericalStability:
    """Comprehensive numerical stability tests."""
    
    def setup_method(self):
        """Setup for each test method."""
        self.stability_checker = StabilityChecker()
        self.benchmarker = PerformanceBenchmarker()
        self.reporter = StabilityReporter()
        self.all_issues = []
        
    def teardown_method(self, flow_name: str = None):
        """Teardown and report issues."""
        if self.all_issues and flow_name:
            # Log all issues found during testing
            for issue in self.all_issues:
                print(f"\n{issue}")
                
            # Save summary report
            report_data = {
                'total_issues': len(self.all_issues),
                'issues': self.all_issues
            }
            self.reporter.save_report(flow_name, report_data)
    
    def _run_flow_operations(self, flow: nn.Module, x: torch.Tensor, 
                           operation_name: str, flow_name: str) -> Tuple[List[str], Dict[str, float]]:
        """Run flow operations and collect stability/performance data."""
        issues = []
        timings = {}
        
        try:
            # Enable gradient computation for stability checks
            x = x.clone().detach().requires_grad_(True)
            
            if isinstance(flow, ContinuousFlow):
                # ContinuousFlow returns (x, log_det_J) like other flows
                if operation_name == "forward":
                    result, exec_time = self.benchmarker.benchmark_function(flow.forward, x)
                    output, log_det = result
                elif operation_name == "inverse":
                    result, exec_time = self.benchmarker.benchmark_function(flow.inverse, x)
                    output, log_det = result
                else:
                    return issues, timings
            else:
                # Standard flow interface with log_det
                if operation_name == "forward":
                    result, exec_time = self.benchmarker.benchmark_function(flow.forward, x)
                    output, log_det = result
                elif operation_name == "inverse":
                    result, exec_time = self.benchmarker.benchmark_function(flow.inverse, x)
                    output, log_det = result
                else:
                    return issues, timings
            
            timings[operation_name] = exec_time
            
            # Check output stability
            issues.extend(self.stability_checker.check_finite_tensor(output, f"{operation_name}_output"))
            
            # Check log determinant stability (if present)
            if log_det is not None:
                issues.extend(self.stability_checker.check_finite_tensor(log_det, f"{operation_name}_log_det"))
            
            # Compute gradients and check for explosion
            if x.requires_grad:
                # Create a scalar loss to enable backward pass
                if log_det is not None:
                    loss = output.sum() + log_det.sum()
                else:
                    loss = output.sum()
                    
                loss.backward(retain_graph=True)
                
                # Check input gradients
                issues.extend(self.stability_checker.check_gradient_explosion(x, f"{operation_name}_input_grad"))
                
                # Check gradients of all parameters
                for name, param in flow.named_parameters():
                    if param.grad is not None:
                        grad_norm = torch.norm(param.grad).item()
                        if grad_norm > GRADIENT_EXPLOSION_THRESHOLD:
                            issues.append(f"**high-priority perf/stability issue** Exploding gradient in {name}: norm = {grad_norm:.2e}")
            
            # Check for performance regression
            perf_issues = self.benchmarker.check_performance_regression(flow_name, operation_name, exec_time)
            issues.extend(perf_issues)
            
        except Exception as e:
            issues.append(f"**high-priority perf/stability issue** Exception in {operation_name}: {str(e)}")
            
        return issues, timings
    
    def test_small_scalars_stability(self, flow_name: str, flow: nn.Module):
        """Test stability with small scalar inputs (±1e-6)."""
        x = StressInputGenerator.generate_small_scalars()
        
        # Test forward operation
        issues_fwd, timings_fwd = self._run_flow_operations(flow, x, "forward", flow_name)
        self.all_issues.extend(issues_fwd)
        
        # Test inverse operation  
        issues_inv, timings_inv = self._run_flow_operations(flow, x, "inverse", flow_name)
        self.all_issues.extend(issues_inv)
        
        # Save test-specific report
        report_data = {
            'small_scalars_test': {
                'forward_issues': len(issues_fwd),
                'inverse_issues': len(issues_inv),
                'forward_time': timings_fwd.get('forward', 0),
                'inverse_time': timings_inv.get('inverse', 0),
                'issues': issues_fwd + issues_inv
            }
        }
        self.reporter.save_report(flow_name, report_data)
        
        self.teardown_method(flow_name)
    
    def test_large_scalars_stability(self, flow_name: str, flow: nn.Module):
        """Test stability with large scalar inputs (±1e3)."""
        x = StressInputGenerator.generate_large_scalars()
        
        # Test forward operation
        issues_fwd, timings_fwd = self._run_flow_operations(flow, x, "forward", flow_name)
        self.all_issues.extend(issues_fwd)
        
        # Test inverse operation
        issues_inv, timings_inv = self._run_flow_operations(flow, x, "inverse", flow_name)
        self.all_issues.extend(issues_inv)
        
        # Save test-specific report
        report_data = {
            'large_scalars_test': {
                'forward_issues': len(issues_fwd),
                'inverse_issues': len(issues_inv),
                'forward_time': timings_fwd.get('forward', 0),
                'inverse_time': timings_inv.get('inverse', 0),
                'issues': issues_fwd + issues_inv
            }
        }
        self.reporter.save_report(flow_name, report_data)
        
        self.teardown_method(flow_name)
    
    def test_high_dim_tensors_stability(self, flow_name: str, flow: nn.Module):
        """Test stability with high-dimensional tensors (128D).""" 
        # Need to create a higher-dimensional version of the flow for this test
        if hasattr(flow, 'dim') or hasattr(flow, 'data_dim'):
            pytest.skip("Skipping high-dim test for flow that has fixed dimensionality")
            
        # For flows that can handle variable dimensions, create 128D version
        dim = 128
        hidden_dim = 64
        
        try:
            if isinstance(flow, CouplingLayer):
                high_dim_flow = CouplingLayer(dim, hidden_dim, create_mask(dim, "alternating"))
            elif isinstance(flow, SplineCouplingLayer):
                high_dim_flow = SplineCouplingLayer(dim, hidden_dim, create_mask(dim, "alternating"))
            elif isinstance(flow, MaskedAutoregressiveFlow):
                high_dim_flow = MaskedAutoregressiveFlow(dim, hidden_dim)
            elif isinstance(flow, InverseAutoregressiveFlow):
                high_dim_flow = InverseAutoregressiveFlow(dim, hidden_dim)
            elif isinstance(flow, ContinuousFlow):
                high_dim_flow = ContinuousFlow(dim, hidden_dim)
            else:
                pytest.skip(f"High-dim test not implemented for {type(flow).__name__}")
        except Exception:
            pytest.skip(f"Cannot create high-dim version of {type(flow).__name__}")
        
        x = StressInputGenerator.generate_high_dim_tensors(dim=dim)
        
        # Test forward operation
        issues_fwd, timings_fwd = self._run_flow_operations(high_dim_flow, x, "forward", flow_name)
        self.all_issues.extend(issues_fwd)
        
        # Test inverse operation
        issues_inv, timings_inv = self._run_flow_operations(high_dim_flow, x, "inverse", flow_name)
        self.all_issues.extend(issues_inv)
        
        # Save test-specific report
        report_data = {
            'high_dim_tensors_test': {
                'dimension': dim,
                'forward_issues': len(issues_fwd),
                'inverse_issues': len(issues_inv),
                'forward_time': timings_fwd.get('forward', 0),
                'inverse_time': timings_inv.get('inverse', 0),
                'issues': issues_fwd + issues_inv
            }
        }
        self.reporter.save_report(flow_name, report_data)
        
        self.teardown_method(flow_name)
    
    def test_mixed_precision_stability(self, flow_name: str, flow: nn.Module):
        """Test stability with mixed-precision (torch.float16) inputs."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available for mixed precision testing")
            
        # Move flow to GPU for mixed precision
        flow = flow.cuda()
        
        # Generate mixed precision input
        x = StressInputGenerator.generate_mixed_precision_batch().cuda()
        
        # Use autocast for mixed precision
        with torch.cuda.amp.autocast():
            # Test forward operation
            issues_fwd, timings_fwd = self._run_flow_operations(flow, x, "forward", flow_name)
            self.all_issues.extend(issues_fwd)
            
            # Test inverse operation  
            issues_inv, timings_inv = self._run_flow_operations(flow, x, "inverse", flow_name)
            self.all_issues.extend(issues_inv)
        
        # Save test-specific report
        report_data = {
            'mixed_precision_test': {
                'forward_issues': len(issues_fwd),
                'inverse_issues': len(issues_inv),
                'forward_time': timings_fwd.get('forward', 0),
                'inverse_time': timings_inv.get('inverse', 0),
                'issues': issues_fwd + issues_inv
            }
        }
        self.reporter.save_report(flow_name, report_data)
        
        self.teardown_method(flow_name)
    
    def test_edge_cases_stability(self, flow_name: str, flow: nn.Module):
        """Test stability with various edge case inputs."""
        edge_cases = StressInputGenerator.generate_edge_cases()
        
        all_edge_issues = []
        edge_timings = {}
        
        for case_name, x in edge_cases:
            # Skip NaN and Inf inputs as they are expected to cause issues
            if case_name in ["with_nan", "with_inf"]:
                continue
                
            # Test forward operation
            issues_fwd, timings_fwd = self._run_flow_operations(flow, x, "forward", flow_name)
            all_edge_issues.extend([f"[{case_name}] {issue}" for issue in issues_fwd])
            
            # Test inverse operation
            issues_inv, timings_inv = self._run_flow_operations(flow, x, "inverse", flow_name)
            all_edge_issues.extend([f"[{case_name}] {issue}" for issue in issues_inv])
            
            # Store timings
            edge_timings[f"{case_name}_forward"] = timings_fwd.get('forward', 0)
            edge_timings[f"{case_name}_inverse"] = timings_inv.get('inverse', 0)
        
        self.all_issues.extend(all_edge_issues)
        
        # Save test-specific report
        report_data = {
            'edge_cases_test': {
                'total_issues': len(all_edge_issues),
                'timings': edge_timings,
                'issues': all_edge_issues
            }
        }
        self.reporter.save_report(flow_name, report_data)
        
        self.teardown_method(flow_name)


@pytest.mark.benchmark(group="flow_performance")
@pytest.mark.parametrize("flow_name,flow", get_all_flows_for_testing())
class TestPerformanceBenchmarking:
    """Performance benchmarking tests using pytest-benchmark."""
    
    def test_forward_performance(self, flow_name: str, flow: nn.Module, benchmark):
        """Benchmark forward pass performance."""
        x = torch.randn(32, 4, requires_grad=True)
        
        def run_forward():
            if isinstance(flow, ContinuousFlow):
                return flow.forward(x)
            else:
                return flow.forward(x)
        
        result = benchmark(run_forward)
        
        # Save benchmark result
        reporter = StabilityReporter()
        report_data = {
            'forward_benchmark': {
                'mean_time': benchmark.stats.mean,
                'std_time': benchmark.stats.stddev,
                'min_time': benchmark.stats.min,
                'max_time': benchmark.stats.max,
                'iterations': benchmark.stats.rounds
            }
        }
        reporter.save_report(flow_name, report_data)
    
    def test_inverse_performance(self, flow_name: str, flow: nn.Module, benchmark):
        """Benchmark inverse pass performance."""
        x = torch.randn(32, 4, requires_grad=True)
        
        def run_inverse():
            if isinstance(flow, ContinuousFlow):
                return flow.inverse(x)
            else:
                return flow.inverse(x)
        
        result = benchmark(run_inverse)
        
        # Save benchmark result
        reporter = StabilityReporter()
        report_data = {
            'inverse_benchmark': {
                'mean_time': benchmark.stats.mean,
                'std_time': benchmark.stats.stddev,
                'min_time': benchmark.stats.min,
                'max_time': benchmark.stats.max,
                'iterations': benchmark.stats.rounds
            }
        }
        reporter.save_report(flow_name, report_data)


if __name__ == "__main__":
    # Run the tests
    pytest.main([__file__, "-v", "--benchmark-save=baseline"])
