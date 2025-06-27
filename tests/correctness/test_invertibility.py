"""
Test invertibility of all flow classes.

This module tests that for all flow classes, the forward and inverse operations
are truly inverses of each other, ensuring numerical accuracy and consistency.
"""

import pytest
import torch
import numpy as np
from src.flows import (
    CouplingLayer,
    MaskedAutoregressiveFlow,
    InverseAutoregressiveFlow,
    SplineCouplingLayer,
    ContinuousFlow
)
from src.models import RealNVP, RealNVPSpline, NormalizingFlowModel


def create_mask(dim, mask_type="alternating"):
    """Helper function to create masks for coupling layers."""
    mask = torch.zeros(dim)
    if mask_type == "alternating":
        mask[::2] = 1
    elif mask_type == "half":
        mask[:dim//2] = 1
    return mask


def get_all_flow_classes():
    """Get all flow classes to test."""
    dim = 4
    hidden_dim = 16
    
    # Individual flow layers
    flows = [
        CouplingLayer(dim, hidden_dim, create_mask(dim, "alternating")),
        CouplingLayer(dim, hidden_dim, create_mask(dim, "half")),
        SplineCouplingLayer(dim, hidden_dim, create_mask(dim, "alternating")),
        SplineCouplingLayer(dim, hidden_dim, create_mask(dim, "half")),
        MaskedAutoregressiveFlow(dim, hidden_dim),
        InverseAutoregressiveFlow(dim, hidden_dim),
    ]
    
    # Continuous flow (special case due to different interface)
    flows.append(ContinuousFlow(dim, hidden_dim))
    
    # Composite models
    flows.extend([
        RealNVP(dim, n_layers=2, hidden_dim=hidden_dim),
        RealNVPSpline(dim, n_layers=2, hidden_dim=hidden_dim),
        NormalizingFlowModel([
            CouplingLayer(dim, hidden_dim, create_mask(dim, "alternating")),
            MaskedAutoregressiveFlow(dim, hidden_dim)
        ])
    ])
    
    return flows


class TestInvertibility:
    """Test class for flow invertibility."""
    
    @pytest.mark.parametrize("flow", get_all_flow_classes())
    def test_forward_inverse_consistency(self, flow):
        """Test that forward(inverse(x)) == x for all flow classes."""
        torch.manual_seed(42)
        np.random.seed(42)
        
        batch_size = 16
        dim = 4
        
        # Generate random test data
        x_original = torch.randn(batch_size, dim, requires_grad=True)
        
        try:
            # Test forward-inverse consistency: x -> z -> x'
            if isinstance(flow, ContinuousFlow):
                # ContinuousFlow has different interface - no log_det return
                z = flow.inverse(x_original)
                x_reconstructed = flow.forward(z)
                
                # Check reconstruction accuracy
                if not torch.allclose(x_original, x_reconstructed, atol=1e-5):
                    pytest.fail(f"**critical-bug** Forward-inverse consistency failed for {type(flow).__name__}: "
                              f"max error = {torch.max(torch.abs(x_original - x_reconstructed)).item():.2e}")
                              
            else:
                # Standard flow interface with log_det
                z, log_det_inv = flow.inverse(x_original)
                x_reconstructed, log_det_fwd = flow.forward(z)
                
                # Check reconstruction accuracy
                if not torch.allclose(x_original, x_reconstructed, atol=1e-5):
                    pytest.fail(f"**critical-bug** Forward-inverse consistency failed for {type(flow).__name__}: "
                              f"max error = {torch.max(torch.abs(x_original - x_reconstructed)).item():.2e}")
                
                # Check log-determinant consistency: log_det_fwd + log_det_inv should ≈ 0
                log_det_sum = log_det_fwd + log_det_inv
                if torch.max(torch.abs(log_det_sum)).item() >= 1e-5:
                    pytest.fail(f"**critical-bug** Log-determinant consistency failed for {type(flow).__name__}: "
                              f"max |log_det_fwd + log_det_inv| = {torch.max(torch.abs(log_det_sum)).item():.2e}")
                              
        except Exception as e:
            pytest.fail(f"**critical-bug** Exception during invertibility test for {type(flow).__name__}: {str(e)}")
    
    @pytest.mark.parametrize("flow", get_all_flow_classes())
    def test_inverse_forward_consistency(self, flow):
        """Test that inverse(forward(z)) == z for all flow classes."""
        torch.manual_seed(123)
        np.random.seed(123)
        
        batch_size = 16
        dim = 4
        
        # Generate random latent samples
        z_original = torch.randn(batch_size, dim, requires_grad=True)
        
        try:
            # Test inverse-forward consistency: z -> x -> z'
            if isinstance(flow, ContinuousFlow):
                # ContinuousFlow has different interface
                x = flow.forward(z_original)
                z_reconstructed = flow.inverse(x)
                
                # Check reconstruction accuracy
                if not torch.allclose(z_original, z_reconstructed, atol=1e-5):
                    pytest.fail(f"**critical-bug** Inverse-forward consistency failed for {type(flow).__name__}: "
                              f"max error = {torch.max(torch.abs(z_original - z_reconstructed)).item():.2e}")
                              
            else:
                # Standard flow interface
                x, log_det_fwd = flow.forward(z_original)
                z_reconstructed, log_det_inv = flow.inverse(x)
                
                # Check reconstruction accuracy
                if not torch.allclose(z_original, z_reconstructed, atol=1e-5):
                    pytest.fail(f"**critical-bug** Inverse-forward consistency failed for {type(flow).__name__}: "
                              f"max error = {torch.max(torch.abs(z_original - z_reconstructed)).item():.2e}")
                
                # Check log-determinant consistency
                log_det_sum = log_det_fwd + log_det_inv
                if torch.max(torch.abs(log_det_sum)).item() >= 1e-5:
                    pytest.fail(f"**critical-bug** Log-determinant consistency failed for {type(flow).__name__}: "
                              f"max |log_det_fwd + log_det_inv| = {torch.max(torch.abs(log_det_sum)).item():.2e}")
                              
        except Exception as e:
            pytest.fail(f"**critical-bug** Exception during invertibility test for {type(flow).__name__}: {str(e)}")
    
    @pytest.mark.parametrize("flow", [f for f in get_all_flow_classes() if not isinstance(f, ContinuousFlow)])
    def test_log_determinant_symmetry(self, flow):
        """Test that log_det_jacobian relationships hold correctly."""
        torch.manual_seed(456)
        np.random.seed(456)
        
        batch_size = 8
        dim = 4
        
        x = torch.randn(batch_size, dim, requires_grad=True)
        
        try:
            # Forward pass: x -> z
            z, log_det_fwd = flow.inverse(x)  # Note: inverse gives us z from x
            
            # Backward pass: z -> x
            x_reconstructed, log_det_inv = flow.forward(z)  # Note: forward gives us x from z
            
            # The relationship should be: log_det_fwd + log_det_inv = 0
            # This is because: log|J_f| + log|J_f^-1| = log|J_f * J_f^-1| = log|I| = 0
            log_det_sum = log_det_fwd + log_det_inv
            
            if torch.max(torch.abs(log_det_sum)).item() >= 1e-5:
                pytest.fail(f"**critical-bug** Log-determinant symmetry failed for {type(flow).__name__}: "
                          f"Expected sum ≈ 0, got max |sum| = {torch.max(torch.abs(log_det_sum)).item():.2e}")
                          
        except Exception as e:
            pytest.fail(f"**critical-bug** Exception during log-determinant symmetry test for {type(flow).__name__}: {str(e)}")


if __name__ == "__main__":
    pytest.main([__file__])
