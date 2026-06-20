"""
Tests for Jacobian analysis functionality.
"""

import torch
import pytest
import numpy as np
import matplotlib.pyplot as plt
from src.flows.coupling.coupling_layer import CouplingLayer
from src.flows.flow.sequential_flow import SequentialFlow
from src.visualization.jacobian_analyzer import JacobianAnalyzer


class TestJacobianAnalyzer:
    """Test cases for JacobianAnalyzer."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.device = 'cpu'
        self.analyzer = JacobianAnalyzer(device=self.device)
        
        # Create a simple flow for testing
        mask = torch.tensor([1., 0.])
        self.flow = CouplingLayer(2, 16, mask)
        
        # Create test data
        self.x = torch.randn(10, 2)
    
    def test_analyzer_initialization(self):
        """Test JacobianAnalyzer initialization."""
        analyzer = JacobianAnalyzer()
        assert analyzer.device == 'cpu'
        
        analyzer_cuda = JacobianAnalyzer(device='cuda')
        assert analyzer_cuda.device == 'cuda'
    
    def test_compute_jacobian(self):
        """Test Jacobian computation."""
        jacobian = self.analyzer.compute_jacobian(self.flow, self.x)
        
        # Check shape
        batch_size, dim = self.x.shape
        assert jacobian.shape == (batch_size, dim, dim)
        
        # Check that Jacobian is not all zeros
        assert not torch.allclose(jacobian, torch.zeros_like(jacobian))
        
        # Check that Jacobian contains finite values
        assert torch.all(torch.isfinite(jacobian))
    
    def test_compute_eigenvalue_spectrum(self):
        """Test eigenvalue spectrum computation."""
        # Test without eigenvectors
        eigenvals, eigenvecs = self.analyzer.compute_eigenvalue_spectrum(
            self.flow, self.x, return_eigenvectors=False
        )
        
        batch_size, dim = self.x.shape
        assert eigenvals.shape == (batch_size, dim)
        assert eigenvecs is None
        
        # Test with eigenvectors
        eigenvals, eigenvecs = self.analyzer.compute_eigenvalue_spectrum(
            self.flow, self.x, return_eigenvectors=True
        )
        
        assert eigenvals.shape == (batch_size, dim)
        assert eigenvecs.shape == (batch_size, dim, dim)
        
        # Check that eigenvalues are finite
        assert torch.all(torch.isfinite(eigenvals))
    
    def test_compute_condition_numbers(self):
        """Test condition number computation."""
        condition_numbers = self.analyzer.compute_condition_numbers(self.flow, self.x)
        
        batch_size = self.x.shape[0]
        assert condition_numbers.shape == (batch_size,)
        
        # Condition numbers should be positive and finite
        assert torch.all(condition_numbers > 0)
        assert torch.all(torch.isfinite(condition_numbers))
    
    def test_plot_eigenvalue_spectrum(self):
        """Test eigenvalue spectrum plotting."""
        fig = self.analyzer.plot_eigenvalue_spectrum(
            self.flow, self.x, show_unit_circle=True
        )
        
        assert fig is not None
        assert len(fig.axes) == 2  # Complex plane and magnitude distribution
        
        # Clean up
        plt.close(fig)
    
    def test_plot_condition_number_evolution(self):
        """Test condition number evolution plotting."""
        # Create multiple layers
        layers = []
        for i in range(3):
            mask = torch.tensor([1., 0.]) if i % 2 == 0 else torch.tensor([0., 1.])
            layer = CouplingLayer(2, 16, mask)
            layers.append(layer)
        
        fig = self.analyzer.plot_condition_number_evolution(layers, self.x)
        
        assert fig is not None
        assert len(fig.axes) == 2  # Box plot and line plot
        
        # Clean up
        plt.close(fig)
    
    def test_analyze_gradient_flow(self):
        """Test gradient flow analysis."""
        results = self.analyzer.analyze_gradient_flow(self.flow, self.x)
        
        # Check that all expected keys are present
        expected_keys = [
            'gradients', 'magnitudes', 'directions',
            'mean_magnitude', 'std_magnitude', 'max_magnitude', 'min_magnitude'
        ]
        
        for key in expected_keys:
            assert key in results
        
        # Check shapes
        batch_size, dim = self.x.shape
        assert results['gradients'].shape == (batch_size, dim)
        assert results['magnitudes'].shape == (batch_size,)
        assert results['directions'].shape == (batch_size, dim)
        
        # Check that statistics are scalars
        assert isinstance(results['mean_magnitude'], torch.Tensor)
        assert results['mean_magnitude'].numel() == 1
    
    def test_plot_gradient_flow_analysis(self):
        """Test gradient flow analysis plotting."""
        fig = self.analyzer.plot_gradient_flow_analysis(self.flow, self.x)
        
        assert fig is not None
        assert len(fig.axes) == 3  # Magnitude hist, direction field, magnitude by position
        
        # Clean up
        plt.close(fig)
    
    def test_compute_jacobian_determinant_accuracy(self):
        """Test Jacobian determinant accuracy computation."""
        results = self.analyzer.compute_jacobian_determinant_accuracy(
            self.flow, self.x, tolerance=1e-4
        )
        
        # Check that all expected keys are present
        expected_keys = [
            'mean_abs_error', 'max_abs_error', 'mean_rel_error', 
            'max_rel_error', 'accuracy_within_tolerance'
        ]
        
        for key in expected_keys:
            assert key in results
            assert isinstance(results[key], float)
        
        # Check that accuracy metrics are reasonable
        assert 0 <= results['accuracy_within_tolerance'] <= 1
        assert results['mean_abs_error'] >= 0
        assert results['max_abs_error'] >= 0
    
    def test_sequential_flow_analysis(self):
        """Test analysis with SequentialFlow."""
        # Create a sequential flow
        layers = []
        for i in range(2):
            mask = torch.tensor([1., 0.]) if i % 2 == 0 else torch.tensor([0., 1.])
            layer = CouplingLayer(2, 16, mask)
            layers.append(layer)
        
        sequential_flow = SequentialFlow(layers)
        
        # Test Jacobian computation
        jacobian = self.analyzer.compute_jacobian(sequential_flow, self.x)
        batch_size, dim = self.x.shape
        assert jacobian.shape == (batch_size, dim, dim)
        
        # Test eigenvalue spectrum
        eigenvals, _ = self.analyzer.compute_eigenvalue_spectrum(sequential_flow, self.x)
        assert eigenvals.shape == (batch_size, dim)
        
        # Test condition numbers
        condition_numbers = self.analyzer.compute_condition_numbers(sequential_flow, self.x)
        assert condition_numbers.shape == (batch_size,)
    
    def test_higher_dimensional_flow(self):
        """Test analysis with higher-dimensional flows."""
        # Create 4D flow
        mask_4d = torch.tensor([1., 0., 1., 0.])
        flow_4d = CouplingLayer(4, 32, mask_4d)
        x_4d = torch.randn(5, 4)
        
        # Test Jacobian computation
        jacobian = self.analyzer.compute_jacobian(flow_4d, x_4d)
        assert jacobian.shape == (5, 4, 4)
        
        # Test eigenvalue spectrum
        eigenvals, _ = self.analyzer.compute_eigenvalue_spectrum(flow_4d, x_4d)
        assert eigenvals.shape == (5, 4)
        
        # Test condition numbers
        condition_numbers = self.analyzer.compute_condition_numbers(flow_4d, x_4d)
        assert condition_numbers.shape == (5,)
    
    def test_numerical_stability(self):
        """Test numerical stability with extreme values."""
        # Test with extreme input values
        x_extreme = torch.randn(5, 2) * 10  # Large values
        
        try:
            jacobian = self.analyzer.compute_jacobian(self.flow, x_extreme)
            assert torch.all(torch.isfinite(jacobian))
            
            condition_numbers = self.analyzer.compute_condition_numbers(self.flow, x_extreme)
            # Condition numbers might be large but should be finite
            assert torch.all(torch.isfinite(condition_numbers))
            
        except RuntimeError:
            # Some numerical instability is expected with extreme values
            pass
    
    def test_error_handling(self):
        """Test error handling for edge cases."""
        # Test with very small batch
        x_small = torch.randn(1, 2)
        
        jacobian = self.analyzer.compute_jacobian(self.flow, x_small)
        assert jacobian.shape == (1, 2, 2)
        
        # Test with singular matrices (might occur in some flows)
        # This is more of a robustness test
        try:
            condition_numbers = self.analyzer.compute_condition_numbers(self.flow, x_small)
            assert condition_numbers.shape == (1,)
        except RuntimeError:
            # Some flows might produce singular Jacobians
            pass


if __name__ == "__main__":
    # Run basic tests
    test_analyzer = TestJacobianAnalyzer()
    test_analyzer.setup_method()
    
    print("Running JacobianAnalyzer tests...")
    
    test_analyzer.test_analyzer_initialization()
    print("✓ Initialization test passed")
    
    test_analyzer.test_compute_jacobian()
    print("✓ Jacobian computation test passed")
    
    test_analyzer.test_compute_eigenvalue_spectrum()
    print("✓ Eigenvalue spectrum test passed")
    
    test_analyzer.test_compute_condition_numbers()
    print("✓ Condition number test passed")
    
    test_analyzer.test_plot_eigenvalue_spectrum()
    print("✓ Eigenvalue spectrum plotting test passed")
    
    test_analyzer.test_analyze_gradient_flow()
    print("✓ Gradient flow analysis test passed")
    
    test_analyzer.test_compute_jacobian_determinant_accuracy()
    print("✓ Jacobian determinant accuracy test passed")
    
    test_analyzer.test_sequential_flow_analysis()
    print("✓ Sequential flow analysis test passed")
    
    print("All JacobianAnalyzer tests passed!")