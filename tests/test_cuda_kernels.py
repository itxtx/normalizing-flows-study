"""
Tests for CUDA kernel optimizations.

Note: These tests focus on interface testing and fallback behavior
since CUDA compilation may not be available in all test environments.
"""

import torch
import pytest
from unittest.mock import patch, MagicMock

from src.flows.optimization.cuda_kernels import (
    CUDASplineEvaluator,
    CUDAAutoregressiveSampler,
    CUDALogDetComputer,
    CUDAFlowOptimizer,
    get_cuda_optimizer,
    benchmark_cuda_kernels
)


class TestCUDASplineEvaluator:
    """Test CUDASplineEvaluator class."""
    
    def test_initialization_without_cuda(self):
        """Test initialization when CUDA is not available."""
        with patch('torch.cuda.is_available', return_value=False):
            evaluator = CUDASplineEvaluator(use_cuda=True)
            assert not evaluator.use_cuda
            assert evaluator.cuda_module is None
    
    def test_initialization_with_cuda_disabled(self):
        """Test initialization with CUDA explicitly disabled."""
        evaluator = CUDASplineEvaluator(use_cuda=False)
        assert not evaluator.use_cuda
        assert evaluator.cuda_module is None
    
    def test_pytorch_fallback_evaluation(self):
        """Test PyTorch fallback implementation."""
        evaluator = CUDASplineEvaluator(use_cuda=False)
        
        batch_size, input_dim, num_bins = 8, 10, 4
        inputs = torch.randn(batch_size, input_dim)
        knots = torch.randn(batch_size, input_dim, num_bins + 1)
        coefficients = torch.randn(batch_size, input_dim, num_bins, 4)
        
        outputs, log_derivs = evaluator.evaluate_splines(
            inputs, knots, coefficients, -1.0, 1.0
        )
        
        assert outputs.shape == inputs.shape
        assert log_derivs.shape == inputs.shape
        assert not torch.isnan(outputs).any()
        assert not torch.isnan(log_derivs).any()
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda_evaluation_fallback(self):
        """Test CUDA evaluation with fallback to PyTorch."""
        # Mock CUDA module to simulate compilation failure
        with patch('src.flows.optimization.cuda_kernels._get_cuda_module', return_value=None):
            evaluator = CUDASplineEvaluator(use_cuda=True)
            assert not evaluator.use_cuda
            
            batch_size, input_dim, num_bins = 4, 5, 3
            inputs = torch.randn(batch_size, input_dim)
            knots = torch.randn(batch_size, input_dim, num_bins + 1)
            coefficients = torch.randn(batch_size, input_dim, num_bins, 4)
            
            outputs, log_derivs = evaluator.evaluate_splines(
                inputs, knots, coefficients
            )
            
            assert outputs.shape == inputs.shape
            assert log_derivs.shape == inputs.shape


class TestCUDAAutoregressiveSampler:
    """Test CUDAAutoregressiveSampler class."""
    
    def test_initialization_without_cuda(self):
        """Test initialization when CUDA is not available."""
        with patch('torch.cuda.is_available', return_value=False):
            sampler = CUDAAutoregressiveSampler(use_cuda=True)
            assert not sampler.use_cuda
            assert sampler.cuda_module is None
    
    def test_pytorch_fallback_sampling(self):
        """Test PyTorch fallback implementation."""
        sampler = CUDAAutoregressiveSampler(use_cuda=False)
        
        batch_size, input_dim = 8, 10
        noise = torch.randn(batch_size, input_dim)
        conditioner_outputs = torch.randn(batch_size, input_dim, 2)
        
        samples = sampler.sample(noise, conditioner_outputs, hidden_dim=64)
        
        assert samples.shape == noise.shape
        assert not torch.isnan(samples).any()
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda_sampling_fallback(self):
        """Test CUDA sampling with fallback to PyTorch."""
        with patch('src.flows.optimization.cuda_kernels._get_cuda_module', return_value=None):
            sampler = CUDAAutoregressiveSampler(use_cuda=True)
            assert not sampler.use_cuda
            
            batch_size, input_dim = 4, 5
            noise = torch.randn(batch_size, input_dim)
            conditioner_outputs = torch.randn(batch_size, input_dim, 2)
            
            samples = sampler.sample(noise, conditioner_outputs)
            
            assert samples.shape == noise.shape
            assert not torch.isnan(samples).any()


class TestCUDALogDetComputer:
    """Test CUDALogDetComputer class."""
    
    def test_initialization_without_cuda(self):
        """Test initialization when CUDA is not available."""
        with patch('torch.cuda.is_available', return_value=False):
            computer = CUDALogDetComputer(use_cuda=True)
            assert not computer.use_cuda
            assert computer.cuda_module is None
    
    def test_pytorch_fallback_computation(self):
        """Test PyTorch fallback implementation."""
        computer = CUDALogDetComputer(use_cuda=False)
        
        batch_size, dim = 8, 10
        jacobian_diag = torch.randn(batch_size, dim)
        
        logdet = computer.compute_logdet(jacobian_diag)
        
        assert logdet.shape == (batch_size,)
        assert not torch.isnan(logdet).any()
        
        # Compare with expected PyTorch computation
        expected = torch.sum(torch.log(torch.abs(jacobian_diag) + 1e-8), dim=1)
        assert torch.allclose(logdet, expected, atol=1e-6)
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda_computation_fallback(self):
        """Test CUDA computation with fallback to PyTorch."""
        with patch('src.flows.optimization.cuda_kernels._get_cuda_module', return_value=None):
            computer = CUDALogDetComputer(use_cuda=True)
            assert not computer.use_cuda
            
            batch_size, dim = 4, 5
            jacobian_diag = torch.randn(batch_size, dim)
            
            logdet = computer.compute_logdet(jacobian_diag)
            
            assert logdet.shape == (batch_size,)
            assert not torch.isnan(logdet).any()


class TestCUDAFlowOptimizer:
    """Test CUDAFlowOptimizer class."""
    
    def test_initialization(self):
        """Test optimizer initialization."""
        optimizer = CUDAFlowOptimizer(
            enable_spline_opt=True,
            enable_autoregressive_opt=True
        )
        
        assert hasattr(optimizer, 'spline_evaluator')
        assert hasattr(optimizer, 'autoregressive_sampler')
        assert hasattr(optimizer, 'logdet_computer')
        assert hasattr(optimizer, 'stats')
        
        # Check initial stats
        stats = optimizer.get_stats()
        assert stats['spline_evaluations'] == 0
        assert stats['autoregressive_samples'] == 0
        assert stats['logdet_computations'] == 0
    
    def test_spline_evaluation_interface(self):
        """Test spline evaluation interface."""
        optimizer = CUDAFlowOptimizer(enable_spline_opt=False)
        
        batch_size, input_dim, num_bins = 4, 5, 3
        inputs = torch.randn(batch_size, input_dim)
        knots = torch.randn(batch_size, input_dim, num_bins + 1)
        coefficients = torch.randn(batch_size, input_dim, num_bins, 4)
        
        outputs, log_derivs = optimizer.evaluate_splines(
            inputs, knots, coefficients
        )
        
        assert outputs.shape == inputs.shape
        assert log_derivs.shape == inputs.shape
        assert optimizer.get_stats()['spline_evaluations'] == 1
    
    def test_autoregressive_sampling_interface(self):
        """Test autoregressive sampling interface."""
        optimizer = CUDAFlowOptimizer(enable_autoregressive_opt=False)
        
        batch_size, input_dim = 4, 5
        noise = torch.randn(batch_size, input_dim)
        conditioner_outputs = torch.randn(batch_size, input_dim, 2)
        
        samples = optimizer.sample_autoregressive(noise, conditioner_outputs)
        
        assert samples.shape == noise.shape
        assert optimizer.get_stats()['autoregressive_samples'] == 1
    
    def test_logdet_computation_interface(self):
        """Test log-determinant computation interface."""
        optimizer = CUDAFlowOptimizer()
        
        batch_size, dim = 4, 5
        jacobian_diag = torch.randn(batch_size, dim)
        
        logdet = optimizer.compute_logdet(jacobian_diag)
        
        assert logdet.shape == (batch_size,)
        assert optimizer.get_stats()['logdet_computations'] == 1
    
    def test_optimization_status(self):
        """Test optimization status reporting."""
        optimizer = CUDAFlowOptimizer()
        
        status = optimizer.get_optimization_status()
        
        assert 'spline_evaluation' in status
        assert 'autoregressive_sampling' in status
        assert 'logdet_computation' in status
        assert 'cuda_available' in status
        
        assert isinstance(status['spline_evaluation'], bool)
        assert isinstance(status['autoregressive_sampling'], bool)
        assert isinstance(status['logdet_computation'], bool)
        assert isinstance(status['cuda_available'], bool)
    
    def test_cuda_availability_check(self):
        """Test CUDA availability check."""
        optimizer = CUDAFlowOptimizer()
        
        is_available = optimizer.is_cuda_available()
        assert isinstance(is_available, bool)
        
        # Should be consistent with individual component availability
        status = optimizer.get_optimization_status()
        expected = (
            status['spline_evaluation'] or
            status['autoregressive_sampling'] or
            status['logdet_computation']
        )
        assert is_available == expected


class TestGlobalCUDAOptimizer:
    """Test global CUDA optimizer functionality."""
    
    def test_get_cuda_optimizer(self):
        """Test getting global CUDA optimizer instance."""
        optimizer1 = get_cuda_optimizer()
        optimizer2 = get_cuda_optimizer()
        
        # Should return the same instance
        assert optimizer1 is optimizer2
        assert isinstance(optimizer1, CUDAFlowOptimizer)
    
    def test_global_optimizer_persistence(self):
        """Test that global optimizer persists state."""
        optimizer = get_cuda_optimizer()
        
        # Use the optimizer
        batch_size, dim = 4, 5
        jacobian_diag = torch.randn(batch_size, dim)
        optimizer.compute_logdet(jacobian_diag)
        
        # Get optimizer again and check stats
        optimizer2 = get_cuda_optimizer()
        stats = optimizer2.get_stats()
        
        assert stats['logdet_computations'] >= 1


class TestBenchmarkCUDAKernels:
    """Test CUDA kernel benchmarking functionality."""
    
    def test_benchmark_without_cuda(self):
        """Test benchmarking when CUDA is not available."""
        with patch('torch.cuda.is_available', return_value=False):
            result = benchmark_cuda_kernels()
            
            assert 'error' in result
            assert 'CUDA not available' in result['error']
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_benchmark_with_cuda(self):
        """Test benchmarking when CUDA is available."""
        # Mock CUDA module to avoid compilation issues
        with patch('src.flows.optimization.cuda_kernels._get_cuda_module', return_value=None):
            result = benchmark_cuda_kernels(
                batch_size=4,
                input_dim=5,
                num_bins=3,
                num_iterations=2
            )
            
            assert 'optimization_status' in result
            assert 'benchmark_config' in result
            
            config = result['benchmark_config']
            assert config['batch_size'] == 4
            assert config['input_dim'] == 5
            assert config['num_bins'] == 3
            assert config['num_iterations'] == 2
    
    def test_benchmark_parameters(self):
        """Test benchmark parameter validation."""
        # Test with different parameter combinations
        params_list = [
            {'batch_size': 8, 'input_dim': 10},
            {'batch_size': 16, 'input_dim': 5, 'num_bins': 4},
            {'batch_size': 4, 'input_dim': 3, 'num_bins': 2, 'num_iterations': 5}
        ]
        
        for params in params_list:
            if torch.cuda.is_available():
                with patch('src.flows.optimization.cuda_kernels._get_cuda_module', return_value=None):
                    result = benchmark_cuda_kernels(**params)
                    
                    assert 'benchmark_config' in result
                    config = result['benchmark_config']
                    
                    for key, value in params.items():
                        assert config[key] == value
            else:
                result = benchmark_cuda_kernels(**params)
                assert 'error' in result


class TestCUDAKernelIntegration:
    """Test integration of CUDA kernels with flow operations."""
    
    def test_spline_evaluation_integration(self):
        """Test integration of spline evaluation with flow operations."""
        optimizer = CUDAFlowOptimizer(enable_spline_opt=False)  # Force PyTorch fallback
        
        # Simulate neural spline flow operation
        batch_size, input_dim = 8, 10
        inputs = torch.randn(batch_size, input_dim)
        
        # Create mock spline parameters
        num_bins = 4
        knots = torch.linspace(-1, 1, num_bins + 1).unsqueeze(0).unsqueeze(0)
        knots = knots.expand(batch_size, input_dim, -1)
        coefficients = torch.randn(batch_size, input_dim, num_bins, 4)
        
        outputs, log_derivs = optimizer.evaluate_splines(inputs, knots, coefficients)
        
        # Check that outputs are reasonable
        assert outputs.shape == inputs.shape
        assert log_derivs.shape == inputs.shape
        assert torch.all(torch.isfinite(outputs))
        assert torch.all(torch.isfinite(log_derivs))
    
    def test_autoregressive_sampling_integration(self):
        """Test integration of autoregressive sampling."""
        optimizer = CUDAFlowOptimizer(enable_autoregressive_opt=False)
        
        # Simulate autoregressive flow sampling
        batch_size, input_dim = 8, 10
        noise = torch.randn(batch_size, input_dim)
        
        # Mock conditioner outputs (mean and log_std for each dimension)
        conditioner_outputs = torch.randn(batch_size, input_dim, 2)
        
        samples = optimizer.sample_autoregressive(noise, conditioner_outputs)
        
        # Check that samples are reasonable
        assert samples.shape == noise.shape
        assert torch.all(torch.isfinite(samples))
    
    def test_logdet_computation_integration(self):
        """Test integration of log-determinant computation."""
        optimizer = CUDAFlowOptimizer()
        
        # Simulate Jacobian diagonal computation
        batch_size, dim = 8, 10
        jacobian_diag = torch.exp(torch.randn(batch_size, dim))  # Ensure positive
        
        logdet = optimizer.compute_logdet(jacobian_diag)
        
        # Check that log-determinant is reasonable
        assert logdet.shape == (batch_size,)
        assert torch.all(torch.isfinite(logdet))
        
        # Compare with manual computation
        expected = torch.sum(torch.log(jacobian_diag), dim=1)
        assert torch.allclose(logdet, expected, atol=1e-6)


if __name__ == "__main__":
    pytest.main([__file__])