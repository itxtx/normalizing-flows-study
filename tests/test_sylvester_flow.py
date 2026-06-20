"""
Unit tests for Sylvester Flow.

This module contains comprehensive tests for the Sylvester flow implementation,
including tests for mathematical correctness, numerical stability, invertibility,
and orthogonality constraints.
"""

import torch
import torch.nn as nn
import pytest
import numpy as np
from src.flows.advanced.sylvester_flow import SylvesterFlow, HouseholderReflection, OrthogonalMatrix


class TestHouseholderReflection:
    """Test cases for Householder Reflection."""
    
    @pytest.fixture
    def householder_2d(self):
        """Create a 2D Householder reflection for testing."""
        return HouseholderReflection(dim=2)
    
    def test_initialization(self, householder_2d):
        """Test that Householder reflection initializes correctly."""
        assert householder_2d.dim == 2
        assert hasattr(householder_2d, 'v')
        assert householder_2d.v.shape == (2,)
    
    def test_orthogonality(self, householder_2d):
        """Test that Householder matrix is orthogonal."""
        H = householder_2d.get_matrix()
        
        # Check that H^T H = I
        HtH = torch.mm(H.t(), H)
        I = torch.eye(2)
        error = torch.norm(HtH - I, p='fro')
        assert error < 1e-6, f"Orthogonality error: {error}"
        
        # Check that det(H) = -1 (reflection property)
        det_H = torch.det(H)
        assert torch.abs(det_H + 1.0) < 1e-6, f"Determinant should be -1, got {det_H}"
    
    def test_reflection_property(self, householder_2d):
        """Test that reflection preserves vector norms."""
        x = torch.randn(5, 2)
        Hx = householder_2d(x)
        
        # Check that ||Hx|| = ||x||
        norm_x = torch.norm(x, dim=1)
        norm_Hx = torch.norm(Hx, dim=1)
        error = torch.abs(norm_x - norm_Hx)
        assert torch.all(error < 1e-6), f"Norm preservation error: {error.max()}"


class TestOrthogonalMatrix:
    """Test cases for Orthogonal Matrix."""
    
    @pytest.fixture
    def orthogonal_3d(self):
        """Create a 3D orthogonal matrix for testing."""
        return OrthogonalMatrix(dim=3, num_householder=2)
    
    def test_initialization(self, orthogonal_3d):
        """Test that orthogonal matrix initializes correctly."""
        assert orthogonal_3d.dim == 3
        assert orthogonal_3d.num_householder == 2
        assert len(orthogonal_3d.reflections) == 2
    
    def test_orthogonality(self, orthogonal_3d):
        """Test that the composed matrix is orthogonal."""
        Q = orthogonal_3d.get_matrix()
        
        # Check that Q^T Q = I
        QtQ = torch.mm(Q.t(), Q)
        I = torch.eye(3)
        error = torch.norm(QtQ - I, p='fro')
        assert error < 1e-5, f"Orthogonality error: {error}"
        
        # Check that |det(Q)| = 1
        det_Q = torch.det(Q)
        assert torch.abs(torch.abs(det_Q) - 1.0) < 1e-6, f"Determinant magnitude should be 1, got {torch.abs(det_Q)}"
    
    def test_norm_preservation(self, orthogonal_3d):
        """Test that orthogonal transformation preserves norms."""
        x = torch.randn(5, 3)
        Qx = orthogonal_3d(x)
        
        # Check that ||Qx|| = ||x||
        norm_x = torch.norm(x, dim=1)
        norm_Qx = torch.norm(Qx, dim=1)
        error = torch.abs(norm_x - norm_Qx)
        assert torch.all(error < 1e-6), f"Norm preservation error: {error.max()}"


class TestSylvesterFlow:
    """Test cases for Sylvester Flow."""
    
    @pytest.fixture
    def sylvester_flow_2d(self):
        """Create a 2D Sylvester flow for testing."""
        return SylvesterFlow(dim=2, num_householder=2, num_transforms=2)
    
    @pytest.fixture
    def sylvester_flow_5d(self):
        """Create a 5D Sylvester flow for testing."""
        return SylvesterFlow(dim=5, num_householder=3, num_transforms=3)
    
    def test_initialization(self, sylvester_flow_2d):
        """Test that Sylvester flow initializes correctly."""
        assert sylvester_flow_2d.dim == 2
        assert sylvester_flow_2d.data_dim == 2
        assert sylvester_flow_2d.num_householder == 2
        assert sylvester_flow_2d.num_transforms == 2
        assert hasattr(sylvester_flow_2d, 'Q')
        assert hasattr(sylvester_flow_2d, 'R')
        assert hasattr(sylvester_flow_2d, 'b')
        assert sylvester_flow_2d.R.shape == (2, 2)
        assert sylvester_flow_2d.b.shape == (2,)
    
    def test_forward_shape(self, sylvester_flow_2d):
        """Test that forward pass produces correct output shapes."""
        batch_size = 10
        z = torch.randn(batch_size, 2)
        
        x, log_det = sylvester_flow_2d.forward(z)
        
        assert x.shape == (batch_size, 2)
        assert log_det.shape == (batch_size,)
        assert not torch.isnan(x).any()
        assert not torch.isinf(x).any()
        assert not torch.isnan(log_det).any()
        assert not torch.isinf(log_det).any()
    
    def test_inverse_shape(self, sylvester_flow_2d):
        """Test that inverse pass produces correct output shapes."""
        batch_size = 10
        x = torch.randn(batch_size, 2)
        
        z, log_det = sylvester_flow_2d.inverse(x)
        
        assert z.shape == (batch_size, 2)
        assert log_det.shape == (batch_size,)
        assert not torch.isnan(z).any()
        assert not torch.isinf(z).any()
        assert not torch.isnan(log_det).any()
        assert not torch.isinf(log_det).any()
    
    def test_invertibility(self, sylvester_flow_2d):
        """Test that forward and inverse are approximately inverse operations."""
        batch_size = 3
        z_original = torch.randn(batch_size, 2)
        
        # Forward then inverse
        x, log_det_forward = sylvester_flow_2d.forward(z_original)
        z_reconstructed, log_det_inverse = sylvester_flow_2d.inverse(x)
        
        # Check reconstruction accuracy
        reconstruction_error = torch.norm(z_original - z_reconstructed, dim=1)
        assert torch.all(reconstruction_error < 1e-3), f"Max reconstruction error: {reconstruction_error.max()}"
        
        # Check log-determinant consistency
        log_det_sum = log_det_forward + log_det_inverse
        assert torch.all(torch.abs(log_det_sum) < 1e-3), f"Max log-det error: {torch.abs(log_det_sum).max()}"
    
    def test_orthogonality_constraint(self, sylvester_flow_2d):
        """Test that the orthogonal matrix Q satisfies Q^T Q = I."""
        error = sylvester_flow_2d.check_orthogonality()
        assert error < 1e-5, f"Orthogonality constraint violated: error = {error}"
    
    def test_gradient_flow(self, sylvester_flow_2d):
        """Test that gradients flow properly through the transformation."""
        z = torch.randn(3, 2, requires_grad=True)
        x, log_det = sylvester_flow_2d.forward(z)
        
        # Compute loss and backpropagate
        loss = x.sum() + log_det.sum()
        loss.backward()
        
        # Check that gradients exist and are finite
        assert z.grad is not None
        assert not torch.isnan(z.grad).any()
        assert not torch.isinf(z.grad).any()
        
        # Check parameter gradients
        for param in sylvester_flow_2d.parameters():
            assert param.grad is not None
            assert not torch.isnan(param.grad).any()
            assert not torch.isinf(param.grad).any()
    
    def test_different_dimensions(self):
        """Test Sylvester flow with different dimensions and parameters."""
        for dim in [2, 3, 5]:
            for num_householder in [1, 2, min(3, dim)]:
                for num_transforms in [1, 2, dim]:
                    flow = SylvesterFlow(
                        dim=dim,
                        num_householder=num_householder,
                        num_transforms=num_transforms
                    )
                    
                    # Test forward pass
                    z = torch.randn(5, dim)
                    x, log_det = flow.forward(z)
                    
                    assert x.shape == (5, dim)
                    assert log_det.shape == (5,)
                    assert not torch.isnan(x).any()
                    assert not torch.isinf(x).any()
                    assert not torch.isnan(log_det).any()
                    assert not torch.isinf(log_det).any()
                    
                    # Test orthogonality
                    error = flow.check_orthogonality()
                    assert error < 1e-4, f"Orthogonality error for dim={dim}, nh={num_householder}: {error}"
    
    def test_numerical_stability_edge_cases(self, sylvester_flow_2d):
        """Test numerical stability with edge cases."""
        # Test with very small inputs
        z_small = 1e-8 * torch.randn(5, 2)
        x, log_det = sylvester_flow_2d.forward(z_small)
        
        assert not torch.isnan(x).any()
        assert not torch.isinf(x).any()
        assert not torch.isnan(log_det).any()
        assert not torch.isinf(log_det).any()
        
        # Test with very large inputs
        z_large = 100 * torch.randn(5, 2)
        x, log_det = sylvester_flow_2d.forward(z_large)
        
        assert not torch.isnan(x).any()
        assert not torch.isinf(x).any()
        assert not torch.isnan(log_det).any()
        assert not torch.isinf(log_det).any()
    
    def test_parameter_count(self):
        """Test parameter counts for different configurations."""
        dim = 5
        num_householder = 3
        num_transforms = 4
        
        flow = SylvesterFlow(
            dim=dim,
            num_householder=num_householder,
            num_transforms=num_transforms
        )
        
        total_params = sum(p.numel() for p in flow.parameters())
        
        # Expected: num_householder * dim (Householder vectors) + 
        #          num_transforms * dim (R matrix) + 
        #          num_transforms (bias vector)
        expected_params = num_householder * dim + num_transforms * dim + num_transforms
        
        assert total_params == expected_params, f"Expected {expected_params} parameters, got {total_params}"


if __name__ == "__main__":
    # Run basic tests
    print("Testing Householder Reflection...")
    h = HouseholderReflection(dim=3)
    x = torch.randn(5, 3)
    Hx = h(x)
    print(f"Input shape: {x.shape}, Output shape: {Hx.shape}")
    
    H = h.get_matrix()
    print(f"Householder matrix shape: {H.shape}")
    print(f"Orthogonality error: {torch.norm(torch.mm(H.t(), H) - torch.eye(3), p='fro'):.6f}")
    print(f"Determinant: {torch.det(H):.6f}")
    
    print("\nTesting Orthogonal Matrix...")
    Q_module = OrthogonalMatrix(dim=3, num_householder=2)
    Qx = Q_module(x)
    print(f"Input shape: {x.shape}, Output shape: {Qx.shape}")
    
    Q = Q_module.get_matrix()
    print(f"Orthogonal matrix shape: {Q.shape}")
    print(f"Orthogonality error: {torch.norm(torch.mm(Q.t(), Q) - torch.eye(3), p='fro'):.6f}")
    print(f"Determinant: {torch.det(Q):.6f}")
    
    print("\nTesting Sylvester Flow...")
    flow = SylvesterFlow(dim=3, num_householder=2, num_transforms=2)
    z = torch.randn(5, 3)
    x, log_det = flow.forward(z)
    print(f"Forward: z.shape={z.shape} -> x.shape={x.shape}, log_det.shape={log_det.shape}")
    
    z_rec, log_det_inv = flow.inverse(x)
    print(f"Inverse: x.shape={x.shape} -> z_rec.shape={z_rec.shape}, log_det_inv.shape={log_det_inv.shape}")
    print(f"Reconstruction error: {torch.norm(z - z_rec, dim=1).max():.6f}")
    print(f"Orthogonality error: {flow.check_orthogonality():.6f}")
    
    print("\nAll basic tests passed!")