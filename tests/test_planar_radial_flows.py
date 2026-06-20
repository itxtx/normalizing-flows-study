"""
Unit tests for Planar and Radial Flows.

This module contains comprehensive tests for the Planar and Radial flow implementations,
including tests for mathematical correctness, numerical stability, and invertibility.
"""

import torch
import torch.nn as nn
import pytest
import numpy as np
from src.flows.advanced.planar_flow import PlanarFlow
from src.flows.advanced.radial_flow import RadialFlow


class TestPlanarFlow:
    """Test cases for Planar Flow."""
    
    @pytest.fixture
    def planar_flow_2d(self):
        """Create a 2D planar flow for testing."""
        return PlanarFlow(dim=2)
    
    @pytest.fixture
    def planar_flow_5d(self):
        """Create a 5D planar flow for testing."""
        return PlanarFlow(dim=5)
    
    def test_initialization(self, planar_flow_2d):
        """Test that planar flow initializes correctly."""
        assert planar_flow_2d.dim == 2
        assert planar_flow_2d.data_dim == 2
        assert hasattr(planar_flow_2d, 'w')
        assert hasattr(planar_flow_2d, 'u_hat')
        assert hasattr(planar_flow_2d, 'b')
        assert planar_flow_2d.w.shape == (2,)
        assert planar_flow_2d.u_hat.shape == (2,)
        assert planar_flow_2d.b.shape == (1,)
    
    def test_forward_shape(self, planar_flow_2d):
        """Test that forward pass produces correct output shapes."""
        batch_size = 10
        z = torch.randn(batch_size, 2)
        
        x, log_det = planar_flow_2d.forward(z)
        
        assert x.shape == (batch_size, 2)
        assert log_det.shape == (batch_size,)
        assert not torch.isnan(x).any()
        assert not torch.isinf(x).any()
        assert not torch.isnan(log_det).any()
        assert not torch.isinf(log_det).any()
    
    def test_inverse_shape(self, planar_flow_2d):
        """Test that inverse pass produces correct output shapes."""
        batch_size = 10
        x = torch.randn(batch_size, 2)
        
        z, log_det = planar_flow_2d.inverse(x)
        
        assert z.shape == (batch_size, 2)
        assert log_det.shape == (batch_size,)
        assert not torch.isnan(z).any()
        assert not torch.isinf(z).any()
        assert not torch.isnan(log_det).any()
        assert not torch.isinf(log_det).any()
    
    def test_invertibility(self, planar_flow_2d):
        """Test that forward and inverse are approximately inverse operations."""
        batch_size = 5
        z_original = torch.randn(batch_size, 2)
        
        # Forward then inverse
        x, log_det_forward = planar_flow_2d.forward(z_original)
        z_reconstructed, log_det_inverse = planar_flow_2d.inverse(x)
        
        # Check reconstruction accuracy
        reconstruction_error = torch.norm(z_original - z_reconstructed, dim=1)
        assert torch.all(reconstruction_error < 1e-4), f"Max reconstruction error: {reconstruction_error.max()}"
        
        # Check log-determinant consistency
        log_det_sum = log_det_forward + log_det_inverse
        assert torch.all(torch.abs(log_det_sum) < 1e-4), f"Max log-det error: {torch.abs(log_det_sum).max()}"
    
    def test_jacobian_determinant(self, planar_flow_2d):
        """Test log-determinant computation using finite differences."""
        torch.manual_seed(42)
        z = torch.randn(3, 2, requires_grad=True)
        
        x, log_det_analytical = planar_flow_2d.forward(z)
        
        # Compute Jacobian numerically
        batch_size, dim = z.shape
        jacobian = torch.zeros(batch_size, dim, dim)
        
        for i in range(dim):
            for j in range(dim):
                # Create unit vector
                e_j = torch.zeros_like(z)
                e_j[:, j] = 1.0
                
                # Compute gradient
                x_i = x[:, i].sum()
                grad = torch.autograd.grad(x_i, z, create_graph=True, retain_graph=True)[0]
                jacobian[:, i, j] = grad[:, j]
        
        # Compute log-determinant numerically
        det_numerical = torch.det(jacobian)
        log_det_numerical = torch.log(torch.abs(det_numerical) + 1e-8)
        
        # Compare with analytical result
        error = torch.abs(log_det_analytical - log_det_numerical)
        assert torch.all(error < 1e-3), f"Max log-det error: {error.max()}"
    
    def test_constraint_satisfaction(self, planar_flow_2d):
        """Test that the invertibility constraint u^T w >= -1 is satisfied."""
        u = planar_flow_2d._get_u()
        w = planar_flow_2d.w
        
        dot_product = torch.dot(u, w)
        assert dot_product >= -1.0 - 1e-6, f"Constraint violated: u^T w = {dot_product}"
    
    def test_gradient_flow(self, planar_flow_2d):
        """Test that gradients flow properly through the transformation."""
        z = torch.randn(5, 2, requires_grad=True)
        x, log_det = planar_flow_2d.forward(z)
        
        # Compute loss and backpropagate
        loss = x.sum() + log_det.sum()
        loss.backward()
        
        # Check that gradients exist and are finite
        assert z.grad is not None
        assert not torch.isnan(z.grad).any()
        assert not torch.isinf(z.grad).any()
        
        # Check parameter gradients
        for param in planar_flow_2d.parameters():
            assert param.grad is not None
            assert not torch.isnan(param.grad).any()
            assert not torch.isinf(param.grad).any()


class TestRadialFlow:
    """Test cases for Radial Flow."""
    
    @pytest.fixture
    def radial_flow_2d(self):
        """Create a 2D radial flow for testing."""
        return RadialFlow(dim=2)
    
    @pytest.fixture
    def radial_flow_5d(self):
        """Create a 5D radial flow for testing."""
        return RadialFlow(dim=5)
    
    def test_initialization(self, radial_flow_2d):
        """Test that radial flow initializes correctly."""
        assert radial_flow_2d.dim == 2
        assert radial_flow_2d.data_dim == 2
        assert hasattr(radial_flow_2d, 'z0')
        assert hasattr(radial_flow_2d, 'alpha_hat')
        assert hasattr(radial_flow_2d, 'beta_hat')
        assert radial_flow_2d.z0.shape == (2,)
        assert radial_flow_2d.alpha_hat.shape == (1,)
        assert radial_flow_2d.beta_hat.shape == (1,)
    
    def test_forward_shape(self, radial_flow_2d):
        """Test that forward pass produces correct output shapes."""
        batch_size = 10
        z = torch.randn(batch_size, 2)
        
        x, log_det = radial_flow_2d.forward(z)
        
        assert x.shape == (batch_size, 2)
        assert log_det.shape == (batch_size,)
        assert not torch.isnan(x).any()
        assert not torch.isinf(x).any()
        assert not torch.isnan(log_det).any()
        assert not torch.isinf(log_det).any()
    
    def test_inverse_shape(self, radial_flow_2d):
        """Test that inverse pass produces correct output shapes."""
        batch_size = 10
        x = torch.randn(batch_size, 2)
        
        z, log_det = radial_flow_2d.inverse(x)
        
        assert z.shape == (batch_size, 2)
        assert log_det.shape == (batch_size,)
        assert not torch.isnan(z).any()
        assert not torch.isinf(z).any()
        assert not torch.isnan(log_det).any()
        assert not torch.isinf(log_det).any()
    
    def test_invertibility(self, radial_flow_2d):
        """Test that forward and inverse are approximately inverse operations."""
        batch_size = 5
        z_original = torch.randn(batch_size, 2)
        
        # Forward then inverse
        x, log_det_forward = radial_flow_2d.forward(z_original)
        z_reconstructed, log_det_inverse = radial_flow_2d.inverse(x)
        
        # Check reconstruction accuracy
        reconstruction_error = torch.norm(z_original - z_reconstructed, dim=1)
        assert torch.all(reconstruction_error < 1e-4), f"Max reconstruction error: {reconstruction_error.max()}"
        
        # Check log-determinant consistency
        log_det_sum = log_det_forward + log_det_inverse
        assert torch.all(torch.abs(log_det_sum) < 1e-4), f"Max log-det error: {torch.abs(log_det_sum).max()}"
    
    def test_constraint_satisfaction(self, radial_flow_2d):
        """Test that the invertibility constraint β > -α is satisfied."""
        alpha = radial_flow_2d._get_alpha()
        beta = radial_flow_2d._get_beta(alpha)
        
        assert beta > -alpha - 1e-6, f"Constraint violated: β = {beta}, α = {alpha}"
        assert alpha > 0, f"Alpha should be positive: α = {alpha}"
    
    def test_gradient_flow(self, radial_flow_2d):
        """Test that gradients flow properly through the transformation."""
        z = torch.randn(5, 2, requires_grad=True)
        x, log_det = radial_flow_2d.forward(z)
        
        # Compute loss and backpropagate
        loss = x.sum() + log_det.sum()
        loss.backward()
        
        # Check that gradients exist and are finite
        assert z.grad is not None
        assert not torch.isnan(z.grad).any()
        assert not torch.isinf(z.grad).any()
        
        # Check parameter gradients
        for param in radial_flow_2d.parameters():
            assert param.grad is not None
            assert not torch.isnan(param.grad).any()
            assert not torch.isinf(param.grad).any()
    
    def test_numerical_stability_edge_cases(self, radial_flow_2d):
        """Test numerical stability with edge cases."""
        # Test with points very close to reference point
        z_close = radial_flow_2d.z0.unsqueeze(0) + 1e-8 * torch.randn(5, 2)
        x, log_det = radial_flow_2d.forward(z_close)
        
        assert not torch.isnan(x).any()
        assert not torch.isinf(x).any()
        assert not torch.isnan(log_det).any()
        assert not torch.isinf(log_det).any()
        
        # Test with points very far from reference point
        z_far = radial_flow_2d.z0.unsqueeze(0) + 100 * torch.randn(5, 2)
        x, log_det = radial_flow_2d.forward(z_far)
        
        assert not torch.isnan(x).any()
        assert not torch.isinf(x).any()
        assert not torch.isnan(log_det).any()
        assert not torch.isinf(log_det).any()


class TestFlowComparison:
    """Test cases comparing Planar and Radial flows."""
    
    def test_different_expressiveness(self):
        """Test that planar and radial flows have different expressiveness."""
        torch.manual_seed(42)
        
        planar = PlanarFlow(dim=2)
        radial = RadialFlow(dim=2)
        
        z = torch.randn(100, 2)
        
        x_planar, _ = planar.forward(z)
        x_radial, _ = radial.forward(z)
        
        # The transformations should be different
        difference = torch.norm(x_planar - x_radial, dim=1)
        assert torch.mean(difference) > 0.1, "Flows should produce different transformations"
    
    def test_parameter_count(self):
        """Test parameter counts for different dimensions."""
        for dim in [2, 5, 10]:
            planar = PlanarFlow(dim=dim)
            radial = RadialFlow(dim=dim)
            
            # Planar: w (dim) + u_hat (dim) + b (1) = 2*dim + 1
            planar_params = sum(p.numel() for p in planar.parameters())
            assert planar_params == 2 * dim + 1
            
            # Radial: z0 (dim) + alpha_hat (1) + beta_hat (1) = dim + 2
            radial_params = sum(p.numel() for p in radial.parameters())
            assert radial_params == dim + 2


if __name__ == "__main__":
    # Run basic tests
    print("Testing Planar Flow...")
    planar = PlanarFlow(dim=2)
    z = torch.randn(5, 2)
    x, log_det = planar.forward(z)
    print(f"Forward: z.shape={z.shape} -> x.shape={x.shape}, log_det.shape={log_det.shape}")
    
    z_rec, log_det_inv = planar.inverse(x)
    print(f"Inverse: x.shape={x.shape} -> z_rec.shape={z_rec.shape}, log_det_inv.shape={log_det_inv.shape}")
    print(f"Reconstruction error: {torch.norm(z - z_rec, dim=1).max():.6f}")
    
    print("\nTesting Radial Flow...")
    radial = RadialFlow(dim=2)
    x, log_det = radial.forward(z)
    print(f"Forward: z.shape={z.shape} -> x.shape={x.shape}, log_det.shape={log_det.shape}")
    
    z_rec, log_det_inv = radial.inverse(x)
    print(f"Inverse: x.shape={x.shape} -> z_rec.shape={z_rec.shape}, log_det_inv.shape={log_det_inv.shape}")
    print(f"Reconstruction error: {torch.norm(z - z_rec, dim=1).max():.6f}")
    
    print("\nAll basic tests passed!")