import pytest
import torch
import numpy as np
from src.flows import MaskedAutoregressiveFlow

# Try to import ARQS if available
try:
    from src.flows.spline import ARQS
    HAS_ARQS = True
except ImportError:
    HAS_ARQS = False

def is_lower_triangular(matrix, atol=1e-6):
    """Check if a matrix is lower-triangular (all elements above diagonal are close to zero)."""
    return torch.all(torch.abs(torch.triu(matrix, diagonal=1)) < atol)

@pytest.mark.parametrize("flow_class,flow_name", [
    (MaskedAutoregressiveFlow, "MAF"),
    *( [(ARQS, "ARQS")] if HAS_ARQS else [] )
])
@pytest.mark.parametrize("dim", [3, 4])
def test_autoregressive_jacobian_is_lower_triangular(flow_class, flow_name, dim):
    torch.manual_seed(42)
    np.random.seed(42)
    hidden_dim = 16
    batch_size = 1  # Jacobian is per-sample
    flow = flow_class(dim, hidden_dim)
    flow.eval()
    x = torch.randn(batch_size, dim, requires_grad=True)
    # Forward pass
    x_in = x.clone().detach().requires_grad_(True)
    y, _ = flow.forward(x_in)
    # Compute Jacobian for the first sample
    jac = torch.zeros(dim, dim)
    for i in range(dim):
        grad_outputs = torch.zeros_like(y)
        grad_outputs[0, i] = 1.0
        grads = torch.autograd.grad(y, x_in, grad_outputs=grad_outputs, retain_graph=True, create_graph=False)[0]
        jac[i, :] = grads[0]
    assert is_lower_triangular(jac), f"{flow_name} Jacobian is not lower-triangular for dim={dim}:\n{jac}" 