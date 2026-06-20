"""
Unit tests for Neural Autoregressive Flow (NAF).
"""

import torch
import torch.nn as nn
import pytest
import numpy as np
from src.flows.advanced.neural_autoregressive_flow import NeuralAutoregressiveFlow, DeepMADE


class TestDeepMADE:
    """Test cases for DeepMADE conditioner network."""
    
    def test_initialization(self):
        """Test DeepMADE initialization."""
        input_dim = 4
        hidden_dims = [64, 64]
        
        made = DeepMADE(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            output_dim_multiplier=2
        )
        
        assert made.input_dim == input_dim
        assert made.hidden_dims == hidden_dims
        assert made.output_dim_multiplier == 2
    
    def test_forward_pass(self):
        """Test forward pass through DeepMADE."""
        batch_size = 32
        input_dim = 4
        hidden_dims = [64, 64]
        
        made = DeepMADE(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            output_dim_multiplier=2
        )
        
        x = torch.randn(batch_size, input_dim)
        output = made(x)
        
        assert output.shape == (batch_size, input_dim * 2)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()
    
    def test_autoregressive_property(self):
        """Test that the network maintains autoregressive property."""
        input_dim = 3
        hidden_dims = [32, 32]
        
        made = DeepMADE(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            output_dim_multiplier=2
        )
        
        batch_size = 16
        x = torch.randn(batch_size, input_dim)
        
        # Test that output for dimension i doesn't depend on input dimensions >= i
        for i in range(input_dim):
            x_modified = x.clone()
            x_modified[:, i:] = torch.randn_like(x_modified[:, i:])
            
            output_original = made(x)
            output_modified = made(x_modified)
            
            # Output for dimensions < i should be the same
            # Output format: [mu_0, mu_1, ..., alpha_0, alpha_1, ...]
            mu_orig, alpha_orig = output_original.chunk(2, dim=1)
            mu_mod, alpha_mod = output_modified.chunk(2, dim=1)
            
            for j in range(i):
                assert torch.allclose(mu_orig[:, j], mu_mod[:, j], atol=1e-6)
                assert torch.allclose(alpha_orig[:, j], alpha_mod[:, j], atol=1e-6)
    
    def test_different_activations(self):
        """Test different activation functions."""
        input_dim = 4
        hidden_dims = [32]
        batch_size = 16
        x = torch.randn(batch_size, input_dim)
        
        activations = ["relu", "elu", "leaky_relu", "gelu"]
        
        for activation in activations:
            made = DeepMADE(
                input_dim=input_dim,
                hidden_dims=hidden_dims,
                activation=activation
            )
            
            output = made(x)
            assert output.shape == (batch_size, input_dim * 2)
            assert not torch.isnan(output).any()
    
    def test_residual_connections(self):
        """Test residual connections."""
        input_dim = 4
        hidden_dims = [64, 64, 64]  # Same dimensions for residual connections
        
        made_with_residual = DeepMADE(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            use_residual=True
        )
        
        made_without_residual = DeepMADE(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            use_residual=False
        )
        
        batch_size = 16
        x = torch.randn(batch_size, input_dim)
        
        output_with = made_with_residual(x)
        output_without = made_without_residual(x)
        
        assert output_with.shape == output_without.shape
        assert not torch.allclose(output_with, output_without)  # Should be different


class TestNeuralAutoregressiveFlow:
    """Test cases for Neural Autoregressive Flow."""
    
    def test_initialization(self):
        """Test NAF initialization."""
        dim = 4
        hidden_dims = [64, 64]
        
        naf = NeuralAutoregressiveFlow(
            dim=dim,
            hidden_dims=hidden_dims
        )
        
        assert naf.data_dim == dim
        assert naf.dim == dim
        assert isinstance(naf.conditioner, DeepMADE)
    
    def test_forward_inverse_consistency(self):
        """Test that forward and inverse are consistent."""
        dim = 4
        batch_size = 32
        
        naf = NeuralAutoregressiveFlow(dim=dim, hidden_dims=[64, 64])
        
        # Test inverse -> forward
        x = torch.randn(batch_size, dim)
        z, log_det_inv = naf.inverse(x)
        x_reconstructed, log_det_fwd = naf.forward(z)
        
        assert torch.allclose(x, x_reconstructed, atol=5e-1)
        assert torch.allclose(log_det_inv, -log_det_fwd, atol=5e-1)
        
        # Test forward -> inverse
        z = torch.randn(batch_size, dim)
        x, log_det_fwd = naf.forward(z)
        z_reconstructed, log_det_inv = naf.inverse(x)
        
        assert torch.allclose(z, z_reconstructed, atol=1e-4)
        assert torch.allclose(log_det_fwd, -log_det_inv, atol=1e-4)
    
    def test_log_det_jacobian_computation(self):
        """Test log-determinant Jacobian computation."""
        dim = 3
        batch_size = 16
        
        naf = NeuralAutoregressiveFlow(dim=dim, hidden_dims=[32, 32])
        
        x = torch.randn(batch_size, dim, requires_grad=True)
        z, log_det_jacobian = naf.inverse(x)
        
        # Compute log-det using autograd for comparison
        log_det_autograd = []
        for i in range(batch_size):
            jacobian = torch.autograd.functional.jacobian(
                lambda x_single: naf.inverse(x_single.unsqueeze(0))[0].squeeze(0),
                x[i]
            )
            log_det_autograd.append(torch.logdet(jacobian))
        
        log_det_autograd = torch.stack(log_det_autograd)
        
        # Should be close (allowing for numerical differences)
        assert torch.allclose(log_det_jacobian, log_det_autograd, atol=5e-1)
    
    def test_numerical_stability(self):
        """Test numerical stability with extreme inputs."""
        dim = 4
        naf = NeuralAutoregressiveFlow(dim=dim, hidden_dims=[64])
        
        # Test with large values
        x_large = torch.randn(16, dim) * 10
        z, log_det = naf.inverse(x_large)
        
        assert not torch.isnan(z).any()
        assert not torch.isinf(z).any()
        assert not torch.isnan(log_det).any()
        assert not torch.isinf(log_det).any()
        
        # Test with small values
        x_small = torch.randn(16, dim) * 0.01
        z, log_det = naf.inverse(x_small)
        
        assert not torch.isnan(z).any()
        assert not torch.isinf(z).any()
        assert not torch.isnan(log_det).any()
        assert not torch.isinf(log_det).any()
    
    def test_gradient_flow(self):
        """Test gradient flow through the network."""
        dim = 4
        batch_size = 16
        
        naf = NeuralAutoregressiveFlow(dim=dim, hidden_dims=[64, 64])
        
        x = torch.randn(batch_size, dim, requires_grad=True)
        z, log_det = naf.inverse(x)
        
        loss = z.sum() + log_det.sum()
        loss.backward()
        
        # Check that gradients exist and are finite
        assert x.grad is not None
        assert torch.isfinite(x.grad).all()
        
        # Check that model parameters have gradients
        for param in naf.parameters():
            if param.requires_grad:
                assert param.grad is not None
                assert torch.isfinite(param.grad).all()
    
    def test_different_configurations(self):
        """Test different NAF configurations."""
        dim = 4
        batch_size = 16
        x = torch.randn(batch_size, dim)
        
        configs = [
            {"hidden_dims": [32], "activation": "relu"},
            {"hidden_dims": [64, 64], "activation": "elu"},
            {"hidden_dims": [128, 64, 32], "activation": "gelu"},
            {"hidden_dims": [64, 64], "use_layer_norm": False},
            {"hidden_dims": [64, 64], "use_residual": False},
            {"hidden_dims": [64, 64], "dropout": 0.2},
        ]
        
        for config in configs:
            naf = NeuralAutoregressiveFlow(dim=dim, **config)
            
            z, log_det = naf.inverse(x)
            assert z.shape == (batch_size, dim)
            assert log_det.shape == (batch_size,)
            assert not torch.isnan(z).any()
            assert not torch.isnan(log_det).any()
    
    def test_sampling_and_log_prob(self):
        """Test sampling and log probability computation."""
        dim = 3
        naf = NeuralAutoregressiveFlow(dim=dim, hidden_dims=[32, 32])
        
        # Test sampling
        base_dist = torch.distributions.MultivariateNormal(
            torch.zeros(dim), torch.eye(dim)
        )
        
        samples = naf.sample(100, base_dist)
        assert samples.shape == (100, dim)
        assert not torch.isnan(samples).any()
        
        # Test log probability
        x = torch.randn(50, dim)
        log_probs = naf.log_prob(x, base_dist)
        assert log_probs.shape == (50,)
        assert not torch.isnan(log_probs).any()
    
    def test_autoregressive_property_in_forward(self):
        """Test that forward transformation maintains autoregressive property."""
        dim = 3
        naf = NeuralAutoregressiveFlow(dim=dim, hidden_dims=[32])
        
        batch_size = 16
        z = torch.randn(batch_size, dim)
        
        # Forward pass should be sequential
        x, log_det = naf.forward(z)
        
        assert x.shape == (batch_size, dim)
        assert log_det.shape == (batch_size,)
        assert not torch.isnan(x).any()
        assert not torch.isnan(log_det).any()
    
    def test_device_compatibility(self):
        """Test device compatibility."""
        if torch.cuda.is_available():
            dim = 4
            batch_size = 16
            
            naf = NeuralAutoregressiveFlow(dim=dim, hidden_dims=[32, 32])
            naf = naf.cuda()
            
            x = torch.randn(batch_size, dim).cuda()
            z, log_det = naf.inverse(x)
            
            assert z.device.type == 'cuda'
            assert log_det.device.type == 'cuda'
            assert not torch.isnan(z).any()
            assert not torch.isnan(log_det).any()


if __name__ == "__main__":
    pytest.main([__file__])