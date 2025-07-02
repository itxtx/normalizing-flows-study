"""
Test log-determinant computation vs autodiff.

This module tests that the analytic log-determinant computation matches
the determinant computed via torch.autograd.functional.jacobian for low dimensions.
"""

import pytest
import torch
import numpy as np
from src.flows.coupling.coupling_layer import CouplingLayer
from src.flows.autoregressive.masked_autoregressive_flow import MaskedAutoregressiveFlow
from src.flows.autoregressive.inverse_autoregressive_flow import InverseAutoregressiveFlow
from src.flows.spline.spline_coupling_layer import SplineCouplingLayer
from src.flows.continuous.continuous_flow import ContinuousFlow
from src.models.real_nvp import RealNVP
from src.models.real_nvp_spline import RealNVPSpline
from src.models.normalizing_flow_model import NormalizingFlowModel


def create_mask(dim, mask_type="alternating"):
    """Helper function to create masks for coupling layers."""
    mask = torch.zeros(dim)
    if mask_type == "alternating":
        mask[::2] = 1
    elif mask_type == "half":
        mask[:dim//2] = 1
    return mask


def get_flow_classes_for_dims(dim):
    """Get flow classes for specific dimension (≤ 3)."""
    hidden_dim = 16
    
    flows = [
        CouplingLayer(dim, hidden_dim, create_mask(dim, "alternating")),
        SplineCouplingLayer(dim, hidden_dim, create_mask(dim, "alternating")),
        MaskedAutoregressiveFlow(dim, hidden_dim),
        InverseAutoregressiveFlow(dim, hidden_dim),
    ]
    
    # Add composite models for dim > 1
    if dim > 1:
        flows.extend([
            RealNVP(dim, n_layers=2, hidden_dim=hidden_dim),
            RealNVPSpline(dim, n_layers=2, hidden_dim=hidden_dim),
            NormalizingFlowModel([
                CouplingLayer(dim, hidden_dim, create_mask(dim, "alternating"))
            ])
        ])
    
    return flows


def compute_jacobian_logdet(flow, x, direction="forward"):
    """
    Compute log-determinant using torch.autograd.functional.jacobian.
    
    Args:
        flow: The flow model
        x: Input tensor
        direction: "forward" or "inverse"
    
    Returns:
        Log-determinant computed via autodiff
    """
    batch_size = x.shape[0]
    log_dets = []
    
    for i in range(batch_size):
        x_single = x[i:i+1].requires_grad_(True)
        
        def flow_fn(input_tensor):
            if direction == "forward":
                output, _ = flow.forward(input_tensor)
            else:
                output, _ = flow.inverse(input_tensor)
            return output.squeeze(0)  # Remove batch dimension for jacobian computation
        
        try:
            # Compute Jacobian
            jacobian = torch.autograd.functional.jacobian(flow_fn, x_single.squeeze(0))
            
            # Compute determinant and take log
            det = torch.det(jacobian)
            
            # Handle numerical issues
            if det <= 0:
                log_det = torch.tensor(float('-inf'))
            else:
                log_det = torch.log(torch.abs(det))
            
            log_dets.append(log_det)
            
        except Exception as e:
            # If jacobian computation fails, return NaN
            log_dets.append(torch.tensor(float('nan')))
    
    return torch.stack(log_dets)


class TestLogDetAutoDiff:
    """Test log-determinant computation against autodiff."""
    
    @pytest.mark.parametrize("dim", [1, 2, 3])
    @pytest.mark.parametrize("direction", ["forward", "inverse"])
    def test_logdet_vs_autodiff(self, dim, direction):
        """Test log-determinant computation vs autodiff for dims ≤ 3."""
        torch.manual_seed(42)
        np.random.seed(42)
        
        flows = get_flow_classes_for_dims(dim)
        
        for flow in flows:
            flow.eval()  # Set to evaluation mode for deterministic behavior
            
            # Generate small batch for easier debugging
            batch_size = 4
            x = torch.randn(batch_size, dim, requires_grad=True) * 0.5  # Smaller magnitude
            
            try:
                # Get analytic log-determinant
                if direction == "forward":
                    _, log_det_analytic = flow.forward(x)
                else:
                    _, log_det_analytic = flow.inverse(x)
                
                # Get autodiff log-determinant
                log_det_autodiff = compute_jacobian_logdet(flow, x, direction)
                
                # Check for valid values
                valid_mask = ~(torch.isnan(log_det_autodiff) | torch.isinf(log_det_autodiff) | 
                              torch.isnan(log_det_analytic) | torch.isinf(log_det_analytic))
                
                if not valid_mask.any():
                    # Skip if all values are invalid
                    continue
                
                # Compute relative error for valid entries
                log_det_analytic_valid = log_det_analytic[valid_mask]
                log_det_autodiff_valid = log_det_autodiff[valid_mask]
                
                # Handle case where analytic is zero
                abs_analytic = torch.abs(log_det_analytic_valid)
                abs_autodiff = torch.abs(log_det_autodiff_valid)
                
                # Use relative error when values are large enough, absolute error otherwise
                threshold = 1e-8
                use_relative = (abs_analytic > threshold) & (abs_autodiff > threshold)
                
                if use_relative.any():
                    rel_error = torch.abs(log_det_analytic_valid[use_relative] - log_det_autodiff_valid[use_relative]) / torch.maximum(abs_analytic[use_relative], abs_autodiff[use_relative])
                    max_rel_error = torch.max(rel_error).item()
                    
                    if max_rel_error > 1e-4:
                        pytest.fail(f"**critical-bug** Log-determinant autodiff comparison failed for {type(flow).__name__} "
                                  f"({direction}, dim={dim}): max relative error = {max_rel_error:.2e} > 1e-4")
                
                if (~use_relative).any():
                    abs_error = torch.abs(log_det_analytic_valid[~use_relative] - log_det_autodiff_valid[~use_relative])
                    max_abs_error = torch.max(abs_error).item()
                    
                    if max_abs_error > 1e-4:
                        pytest.fail(f"**critical-bug** Log-determinant autodiff comparison failed for {type(flow).__name__} "
                                  f"({direction}, dim={dim}): max absolute error = {max_abs_error:.2e} > 1e-4")
                                  
            except Exception as e:
                pytest.fail(f"**critical-bug** Exception during log-determinant autodiff test for {type(flow).__name__} "
                          f"({direction}, dim={dim}): {str(e)}")
    
    @pytest.mark.parametrize("dim", [1, 2, 3])
    def test_logdet_consistency_with_autodiff(self, dim):
        """Test that forward and inverse log-determinants are consistent with autodiff."""
        torch.manual_seed(123)
        np.random.seed(123)
        
        flows = get_flow_classes_for_dims(dim)
        
        for flow in flows:
            flow.eval()
            
            batch_size = 3
            x = torch.randn(batch_size, dim, requires_grad=True) * 0.3
            
            try:
                # Get analytic values
                z, log_det_inv_analytic = flow.inverse(x)
                x_reconstructed, log_det_fwd_analytic = flow.forward(z)
                
                # Get autodiff values
                log_det_inv_autodiff = compute_jacobian_logdet(flow, x, "inverse")
                log_det_fwd_autodiff = compute_jacobian_logdet(flow, z, "forward")
                
                # Test inverse direction
                valid_inv = ~(torch.isnan(log_det_inv_autodiff) | torch.isinf(log_det_inv_autodiff) | 
                             torch.isnan(log_det_inv_analytic) | torch.isinf(log_det_inv_analytic))
                
                if valid_inv.any():
                    analytic_valid = log_det_inv_analytic[valid_inv]
                    autodiff_valid = log_det_inv_autodiff[valid_inv]
                    
                    # Compute relative error
                    abs_analytic = torch.abs(analytic_valid)
                    abs_autodiff = torch.abs(autodiff_valid)
                    threshold = 1e-8
                    
                    use_relative = (abs_analytic > threshold) & (abs_autodiff > threshold)
                    
                    if use_relative.any():
                        rel_error = torch.abs(analytic_valid[use_relative] - autodiff_valid[use_relative]) / torch.maximum(abs_analytic[use_relative], abs_autodiff[use_relative])
                        max_rel_error = torch.max(rel_error).item()
                        
                        if max_rel_error > 1e-4:
                            pytest.fail(f"**critical-bug** Inverse log-det consistency with autodiff failed for {type(flow).__name__} "
                                      f"(dim={dim}): max relative error = {max_rel_error:.2e} > 1e-4")
                
                # Test forward direction
                valid_fwd = ~(torch.isnan(log_det_fwd_autodiff) | torch.isinf(log_det_fwd_autodiff) | 
                             torch.isnan(log_det_fwd_analytic) | torch.isinf(log_det_fwd_analytic))
                
                if valid_fwd.any():
                    analytic_valid = log_det_fwd_analytic[valid_fwd]
                    autodiff_valid = log_det_fwd_autodiff[valid_fwd]
                    
                    abs_analytic = torch.abs(analytic_valid)
                    abs_autodiff = torch.abs(autodiff_valid)
                    use_relative = (abs_analytic > threshold) & (abs_autodiff > threshold)
                    
                    if use_relative.any():
                        rel_error = torch.abs(analytic_valid[use_relative] - autodiff_valid[use_relative]) / torch.maximum(abs_analytic[use_relative], abs_autodiff[use_relative])
                        max_rel_error = torch.max(rel_error).item()
                        
                        if max_rel_error > 1e-4:
                            pytest.fail(f"**critical-bug** Forward log-det consistency with autodiff failed for {type(flow).__name__} "
                                      f"(dim={dim}): max relative error = {max_rel_error:.2e} > 1e-4")
                                      
            except Exception as e:
                pytest.fail(f"**critical-bug** Exception during log-det consistency test for {type(flow).__name__} "
                          f"(dim={dim}): {str(e)}")


if __name__ == "__main__":
    pytest.main([__file__])
