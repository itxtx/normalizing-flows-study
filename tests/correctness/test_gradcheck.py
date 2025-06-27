"""
Test gradient computation using torch.autograd.gradcheck.

This module tests that the gradients computed by the flow implementations
are correct using PyTorch's gradient checking functionality.
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


def get_all_flow_classes_for_gradcheck():
    """Get all flow classes for gradient checking."""
    dim = 3  # Use smaller dimension for gradient checking
    hidden_dim = 8  # Smaller network for faster computation
    
    flows = [
        CouplingLayer(dim, hidden_dim, create_mask(dim, "alternating")),
        CouplingLayer(dim, hidden_dim, create_mask(dim, "half")),
        SplineCouplingLayer(dim, hidden_dim, create_mask(dim, "alternating"), num_bins=4),  # Fewer bins
        SplineCouplingLayer(dim, hidden_dim, create_mask(dim, "half"), num_bins=4),
        MaskedAutoregressiveFlow(dim, hidden_dim),
        InverseAutoregressiveFlow(dim, hidden_dim),
    ]
    
    # Add ContinuousFlow (special case)
    flows.append(ContinuousFlow(dim, hidden_dim))
    
    # Add composite models with smaller sizes
    flows.extend([
        RealNVP(dim, n_layers=2, hidden_dim=hidden_dim),
        RealNVPSpline(dim, n_layers=2, hidden_dim=hidden_dim),
        NormalizingFlowModel([
            CouplingLayer(dim, hidden_dim, create_mask(dim, "alternating"))
        ])
    ])
    
    return flows


class FlowForwardFunction(torch.autograd.Function):
    """Custom autograd function wrapper for flow forward pass."""
    
    @staticmethod
    def forward(ctx, input_tensor, flow):
        ctx.flow = flow
        with torch.no_grad():
            if isinstance(flow, ContinuousFlow):
                output = flow.forward(input_tensor)
                return output
            else:
                output, log_det = flow.forward(input_tensor)
                ctx.save_for_backward(input_tensor, log_det)
                return output
    
    @staticmethod
    def backward(ctx, grad_output):
        flow = ctx.flow
        if isinstance(flow, ContinuousFlow):
            # For ContinuousFlow, we need to use the regular backward pass
            return None, None
        else:
            # For regular flows, we can use the saved tensors
            return None, None


class FlowInverseFunction(torch.autograd.Function):
    """Custom autograd function wrapper for flow inverse pass."""
    
    @staticmethod
    def forward(ctx, input_tensor, flow):
        ctx.flow = flow
        with torch.no_grad():
            if isinstance(flow, ContinuousFlow):
                output = flow.inverse(input_tensor)
                return output
            else:
                output, log_det = flow.inverse(input_tensor)
                ctx.save_for_backward(input_tensor, log_det)
                return output
    
    @staticmethod
    def backward(ctx, grad_output):
        flow = ctx.flow
        return None, None


def create_flow_forward_function(flow):
    """Create a function for gradient checking forward pass."""
    def forward_fn(x):
        x = x.requires_grad_(True)
        if isinstance(flow, ContinuousFlow):
            return flow.forward(x)
        else:
            output, _ = flow.forward(x)
            return output
    return forward_fn


def create_flow_inverse_function(flow):
    """Create a function for gradient checking inverse pass."""
    def inverse_fn(x):
        x = x.requires_grad_(True)
        if isinstance(flow, ContinuousFlow):
            return flow.inverse(x)
        else:
            output, _ = flow.inverse(x)
            return output
    return inverse_fn


class TestGradCheck:
    """Test gradient computation using torch.autograd.gradcheck."""
    
    @pytest.mark.parametrize("flow", get_all_flow_classes_for_gradcheck())
    def test_forward_gradcheck(self, flow):
        """Test gradient correctness for forward pass using gradcheck."""
        torch.manual_seed(42)
        np.random.seed(42)
        
        # Set to double precision as required by gradcheck
        flow = flow.double()
        
        # Small input for gradient checking
        batch_size = 2
        dim = 3
        input_tensor = torch.randn(batch_size, dim, dtype=torch.float64, requires_grad=True)
        
        try:
            # Create function for gradient checking
            forward_fn = create_flow_forward_function(flow)
            
            # Run gradient check
            gradcheck_result = torch.autograd.gradcheck(
                forward_fn,
                input_tensor,
                eps=1e-6,
                atol=1e-4,
                rtol=1e-3,
                fast_mode=False
            )
            
            if not gradcheck_result:
                pytest.fail(f"**critical-bug** Forward gradient check failed for {type(flow).__name__}")
                
        except Exception as e:
            pytest.fail(f"**critical-bug** Exception during forward gradient check for {type(flow).__name__}: {str(e)}")
    
    @pytest.mark.parametrize("flow", get_all_flow_classes_for_gradcheck())
    def test_inverse_gradcheck(self, flow):
        """Test gradient correctness for inverse pass using gradcheck."""
        torch.manual_seed(123)
        np.random.seed(123)
        
        # Set to double precision as required by gradcheck
        flow = flow.double()
        
        # Small input for gradient checking
        batch_size = 2
        dim = 3
        input_tensor = torch.randn(batch_size, dim, dtype=torch.float64, requires_grad=True)
        
        try:
            # Create function for gradient checking
            inverse_fn = create_flow_inverse_function(flow)
            
            # Run gradient check
            gradcheck_result = torch.autograd.gradcheck(
                inverse_fn,
                input_tensor,
                eps=1e-6,
                atol=1e-4,
                rtol=1e-3,
                fast_mode=False
            )
            
            if not gradcheck_result:
                pytest.fail(f"**critical-bug** Inverse gradient check failed for {type(flow).__name__}")
                
        except Exception as e:
            pytest.fail(f"**critical-bug** Exception during inverse gradient check for {type(flow).__name__}: {str(e)}")
    
    @pytest.mark.parametrize("flow", [f for f in get_all_flow_classes_for_gradcheck() if not isinstance(f, ContinuousFlow)])
    def test_log_det_gradcheck(self, flow):
        """Test gradient correctness for log determinant computation."""
        torch.manual_seed(456)
        np.random.seed(456)
        
        # Set to double precision
        flow = flow.double()
        
        batch_size = 2
        dim = 3
        input_tensor = torch.randn(batch_size, dim, dtype=torch.float64, requires_grad=True)
        
        def log_det_forward_fn(x):
            x = x.requires_grad_(True)
            _, log_det = flow.forward(x)
            return log_det
        
        def log_det_inverse_fn(x):
            x = x.requires_grad_(True)
            _, log_det = flow.inverse(x)
            return log_det
        
        try:
            # Test forward log_det gradients
            gradcheck_fwd = torch.autograd.gradcheck(
                log_det_forward_fn,
                input_tensor,
                eps=1e-6,
                atol=1e-4,
                rtol=1e-3,
                fast_mode=False
            )
            
            if not gradcheck_fwd:
                pytest.fail(f"**critical-bug** Forward log-det gradient check failed for {type(flow).__name__}")
            
            # Test inverse log_det gradients
            gradcheck_inv = torch.autograd.gradcheck(
                log_det_inverse_fn,
                input_tensor,
                eps=1e-6,
                atol=1e-4,
                rtol=1e-3,
                fast_mode=False
            )
            
            if not gradcheck_inv:
                pytest.fail(f"**critical-bug** Inverse log-det gradient check failed for {type(flow).__name__}")
                
        except Exception as e:
            pytest.fail(f"**critical-bug** Exception during log-det gradient check for {type(flow).__name__}: {str(e)}")
    
    @pytest.mark.parametrize("flow", get_all_flow_classes_for_gradcheck())
    def test_parameter_gradients(self, flow):
        """Test that parameters receive gradients correctly."""
        torch.manual_seed(789)
        np.random.seed(789)
        
        # Set to double precision
        flow = flow.double()
        
        batch_size = 3
        dim = 3
        input_tensor = torch.randn(batch_size, dim, dtype=torch.float64, requires_grad=True)
        
        try:
            # Forward pass
            if isinstance(flow, ContinuousFlow):
                output = flow.forward(input_tensor)
                loss = torch.sum(output ** 2)
            else:
                output, log_det = flow.forward(input_tensor)
                loss = torch.sum(output ** 2) + torch.sum(log_det ** 2)
            
            # Backward pass
            loss.backward()
            
            # Check that parameters have gradients
            param_count = 0
            grad_count = 0
            
            for param in flow.parameters():
                param_count += 1
                if param.grad is not None:
                    grad_count += 1
                    # Check for NaN or infinite gradients
                    if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                        pytest.fail(f"**critical-bug** Invalid gradients (NaN/Inf) found in parameters for {type(flow).__name__}")
            
            if param_count > 0 and grad_count == 0:
                pytest.fail(f"**critical-bug** No parameter gradients computed for {type(flow).__name__}")
                
        except Exception as e:
            pytest.fail(f"**critical-bug** Exception during parameter gradient test for {type(flow).__name__}: {str(e)}")


if __name__ == "__main__":
    pytest.main([__file__])
