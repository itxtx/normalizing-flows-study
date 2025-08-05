"""
Unit tests for Residual Flow with Lipschitz constraints.
"""

import torch
import torch.nn as nn
import pytest
import numpy as np
from src.flows.advanced.residual_flow import ResidualFlow, SpectralNorm, ResidualBlock


class TestSpectralNorm:
    """Test cases for SpectralNorm layer."""
    
    def test_initialization(self):
        """Test SpectralNorm initialization."""
        linear = nn.Linear(10, 5)
        spec_norm = SpectralNorm(linear, lipschitz_constant=0.9)
        
        assert spec_norm.lipschitz_constant == 0.9
        assert spec_norm.u.shape == (5,)
        assert spec_norm.v.shape == (10,)
    
    def test_forward_pass(self):
        """Test forward pass through SpectralNorm."""
        linear = nn.Linear(10, 5)
        spec_norm = SpectralNorm(linear, lipschitz_constant=0.9)
        
        x = torch.randn(32, 10)
        output = spec_norm(x)
        
        assert output.shape == (32, 5)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()
    
    def test_lipschitz_constraint(self):
        """Test that spectral normalization enforces Lipschitz constraint."""
        # Create a linear layer with large weights
        linear = nn.Linear(5, 5)
        nn.init.normal_(linear.weight, mean=0, std=10)  # Large weights
        
        # Check original spectral norm is large
        u, s, v = torch.svd(linear.weight)
        original_spectral_norm = s[0].item()
        assert original_spectral_norm > 1.0  # Should be large
        
        spec_norm = SpectralNorm(linear, lipschitz_constant=0.5)
        
        # Test that the effective spectral norm is constrained during forward pass
        # We'll test this by checking the Lipschitz property of the function
        x1 = torch.randn(16, 5)
        x2 = torch.randn(16, 5)
        
        y1 = spec_norm(x1)
        y2 = spec_norm(x2)
        
        # Check Lipschitz constraint: ||f(x1) - f(x2)|| <= L * ||x1 - x2||
        y_diff = torch.norm(y1 - y2, dim=1)
        x_diff = torch.norm(x1 - x2, dim=1)
        
        # Compute Lipschitz ratios
        lipschitz_ratios = y_diff / (x_diff + 1e-8)  # Add epsilon to avoid division by zero
        
        # Should be approximately bounded by the Lipschitz constant
        assert torch.all(lipschitz_ratios <= 0.7)  # Allow some tolerance


class TestResidualBlock:
    """Test cases for ResidualBlock."""
    
    def test_initialization(self):
        """Test ResidualBlock initialization."""
        block = ResidualBlock(dim=4, hidden_dim=16, lipschitz_constant=0.9)
        
        assert block.dim == 4
        assert block.hidden_dim == 16
        assert block.lipschitz_constant == 0.9
    
    def test_forward_pass(self):
        """Test forward pass through ResidualBlock."""
        block = ResidualBlock(dim=4, hidden_dim=16)
        
        x = torch.randn(32, 4)
        output = block(x)
        
        assert output.shape == (32, 4)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()
    
    def test_residual_connection(self):
        """Test that the block implements residual connection."""
        block = ResidualBlock(dim=4, hidden_dim=16)
        
        x = torch.randn(32, 4)
        output = block(x)
        
        # Output should be x + f(x), so it should be different from x
        assert not torch.allclose(output, x)
        
        # But the difference should be bounded due to Lipschitz constraint
        diff = torch.norm(output - x, dim=1)
        assert torch.all(diff < 10)  # Reasonable bound
    
    def test_different_activations(self):
        """Test different activation functions."""
        activations = ["relu", "elu", "leaky_relu", "tanh"]
        x = torch.randn(16, 4)
        
        for activation in activations:
            block = ResidualBlock(dim=4, hidden_dim=16, activation=activation)
            output = block(x)
            
            assert output.shape == (16, 4)
            assert not torch.isnan(output).any()


class TestResidualFlow:
    """Test cases for ResidualFlow."""
    
    def test_initialization(self):
        """Test ResidualFlow initialization."""
        flow = ResidualFlow(dim=4, hidden_dim=16, lipschitz_constant=0.9)
        
        assert flow.data_dim == 4
        assert flow.dim == 4
        assert flow.hidden_dim == 16
        assert flow.lipschitz_constant == 0.9
    
    def test_forward_pass(self):
        """Test forward transformation."""
        flow = ResidualFlow(dim=4, hidden_dim=16)
        
        z = torch.randn(32, 4)
        x, log_det = flow.forward(z)
        
        assert x.shape == (32, 4)
        assert log_det.shape == (32,)
        assert not torch.isnan(x).any()
        assert not torch.isnan(log_det).any()
    
    def test_inverse_pass(self):
        """Test inverse transformation."""
        flow = ResidualFlow(dim=4, hidden_dim=16, max_iter=50)
        
        x = torch.randn(32, 4)
        z, log_det = flow.inverse(x)
        
        assert z.shape == (32, 4)
        assert log_det.shape == (32,)
        assert not torch.isnan(z).any()
        assert not torch.isnan(log_det).any()
    
    def test_forward_inverse_consistency(self):
        """Test that forward and inverse are approximately consistent."""
        flow = ResidualFlow(dim=3, hidden_dim=16, max_iter=100, tol=1e-6)
        
        # Test inverse -> forward
        x = torch.randn(16, 3)
        z, log_det_inv = flow.inverse(x)
        x_reconstructed, log_det_fwd = flow.forward(z)
        
        # Should be approximately consistent
        assert torch.allclose(x, x_reconstructed, atol=1e-3)
        assert torch.allclose(log_det_inv, -log_det_fwd, atol=1e-2)
        
        # Test forward -> inverse
        z = torch.randn(16, 3)
        x, log_det_fwd = flow.forward(z)
        z_reconstructed, log_det_inv = flow.inverse(x)
        
        # Should be approximately consistent
        assert torch.allclose(z, z_reconstructed, atol=1e-3)
        assert torch.allclose(log_det_fwd, -log_det_inv, atol=1e-2)
    
    def test_lipschitz_constraint_enforcement(self):
        """Test that the flow enforces Lipschitz constraints."""
        flow = ResidualFlow(dim=4, hidden_dim=16, lipschitz_constant=0.5)
        
        # Test that the residual function has bounded Lipschitz constant
        x1 = torch.randn(16, 4)
        x2 = torch.randn(16, 4)
        
        f1 = flow._residual_function(x1)
        f2 = flow._residual_function(x2)
        
        # Lipschitz constraint: ||f(x1) - f(x2)|| <= L * ||x1 - x2||
        f_diff = torch.norm(f1 - f2, dim=1)
        x_diff = torch.norm(x1 - x2, dim=1)
        
        # Allow some tolerance for the constraint
        lipschitz_ratios = f_diff / (x_diff + 1e-8)  # Add small epsilon to avoid division by zero
        assert torch.all(lipschitz_ratios <= 0.7)  # Slightly above 0.5 for tolerance
    
    def test_fixed_point_convergence(self):
        """Test that fixed-point iteration converges."""
        flow = ResidualFlow(dim=3, hidden_dim=8, max_iter=50, tol=1e-5)
        
        x = torch.randn(8, 3)
        z, _ = flow.inverse(x)
        
        # Verify that z is indeed a fixed point: z + f(z) = x
        reconstructed_x = z + flow._residual_function(z)
        assert torch.allclose(x, reconstructed_x, atol=1e-4)
    
    def test_jacobian_computation(self):
        """Test Jacobian computation."""
        flow = ResidualFlow(dim=2, hidden_dim=8)
        
        z = torch.randn(4, 2, requires_grad=True)
        jacobian = flow._compute_jacobian(z)
        
        assert jacobian.shape == (4, 2, 2)
        assert not torch.isnan(jacobian).any()
        assert not torch.isinf(jacobian).any()
    
    def test_neumann_series_approximation(self):
        """Test Neumann series log-determinant approximation."""
        flow = ResidualFlow(dim=3, hidden_dim=8, neumann_terms=3)
        
        z = torch.randn(8, 3)
        log_det = flow._compute_log_det_neumann(z)
        
        assert log_det.shape == (8,)
        assert not torch.isnan(log_det).any()
        assert not torch.isinf(log_det).any()
    
    def test_gradient_flow(self):
        """Test gradient flow through the network."""
        flow = ResidualFlow(dim=3, hidden_dim=16)
        
        z = torch.randn(16, 3, requires_grad=True)
        x, log_det = flow.forward(z)
        
        loss = x.sum() + log_det.sum()
        loss.backward()
        
        # Check that gradients exist and are finite
        assert z.grad is not None
        assert torch.isfinite(z.grad).all()
        
        # Check that model parameters have gradients
        for param in flow.parameters():
            if param.requires_grad:
                assert param.grad is not None
                assert torch.isfinite(param.grad).all()
    
    def test_numerical_stability(self):
        """Test numerical stability with extreme inputs."""
        flow = ResidualFlow(dim=4, hidden_dim=16)
        
        # Test with large values
        z_large = torch.randn(16, 4) * 10
        x, log_det = flow.forward(z_large)
        
        assert not torch.isnan(x).any()
        assert not torch.isinf(x).any()
        assert not torch.isnan(log_det).any()
        assert not torch.isinf(log_det).any()
        
        # Test with small values
        z_small = torch.randn(16, 4) * 0.01
        x, log_det = flow.forward(z_small)
        
        assert not torch.isnan(x).any()
        assert not torch.isinf(x).any()
        assert not torch.isnan(log_det).any()
        assert not torch.isinf(log_det).any()
    
    def test_sampling_and_log_prob(self):
        """Test sampling and log probability computation."""
        flow = ResidualFlow(dim=3, hidden_dim=16)
        
        # Test sampling
        base_dist = torch.distributions.MultivariateNormal(
            torch.zeros(3), torch.eye(3)
        )
        
        samples = flow.sample(100, base_dist)
        assert samples.shape == (100, 3)
        assert not torch.isnan(samples).any()
        
        # Test log probability
        x = torch.randn(50, 3)
        log_probs = flow.log_prob(x, base_dist)
        assert log_probs.shape == (50,)
        assert not torch.isnan(log_probs).any()
    
    def test_different_configurations(self):
        """Test different ResidualFlow configurations."""
        configs = [
            {"hidden_dim": 8, "lipschitz_constant": 0.5},
            {"hidden_dim": 32, "lipschitz_constant": 0.8},
            {"hidden_dim": 16, "activation": "tanh"},
            {"hidden_dim": 16, "max_iter": 20, "tol": 1e-4},
            {"hidden_dim": 16, "neumann_terms": 5},
        ]
        
        dim = 4
        batch_size = 16
        z = torch.randn(batch_size, dim)
        
        for config in configs:
            flow = ResidualFlow(dim=dim, **config)
            
            x, log_det = flow.forward(z)
            assert x.shape == (batch_size, dim)
            assert log_det.shape == (batch_size,)
            assert not torch.isnan(x).any()
            assert not torch.isnan(log_det).any()
    
    def test_device_compatibility(self):
        """Test device compatibility."""
        if torch.cuda.is_available():
            dim = 4
            batch_size = 16
            
            flow = ResidualFlow(dim=dim, hidden_dim=16)
            flow = flow.cuda()
            
            z = torch.randn(batch_size, dim).cuda()
            x, log_det = flow.forward(z)
            
            assert x.device.type == 'cuda'
            assert log_det.device.type == 'cuda'
            assert not torch.isnan(x).any()
            assert not torch.isnan(log_det).any()


if __name__ == "__main__":
    pytest.main([__file__])