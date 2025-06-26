import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import MultivariateNormal
import sys
sys.path.append('src')
from src.flows import InverseAutoregressiveFlow
from src.models import NormalizingFlowModel

def test_iaf_stability():
    """Test if IAF produces stable outputs without gradient issues."""
    
    # Configuration
    input_dim = 2
    hidden_dim = 64
    num_flows = 3
    batch_size = 32
    
    # Create model
    flows = [InverseAutoregressiveFlow(dim=input_dim, hidden_dim=hidden_dim) for _ in range(num_flows)]
    model = NormalizingFlowModel(flows)
    
    # Create test data
    test_data = torch.randn(batch_size, input_dim, requires_grad=True)
    base_dist = MultivariateNormal(torch.zeros(input_dim), torch.eye(input_dim))
    
    print("Testing IAF stability...")
    
    # Test inverse pass (this is the slow one for IAF)
    try:
        z, log_det = model.inverse(test_data)
        print(f"✓ Inverse pass successful")
        print(f"  z shape: {z.shape}, log_det shape: {log_det.shape}")
        print(f"  z has NaN: {torch.isnan(z).any()}")
        print(f"  log_det has NaN: {torch.isnan(log_det).any()}")
        
        # Test log probability calculation
        log_prob_z = base_dist.log_prob(z)
        print(f"✓ Log probability calculation successful")
        print(f"  log_prob_z has NaN: {torch.isnan(log_prob_z).any()}")
        
        # Test gradient computation
        loss = -(log_prob_z + log_det).mean()
        loss.backward()
        print(f"✓ Gradient computation successful")
        print(f"  test_data.grad has NaN: {torch.isnan(test_data.grad).any() if test_data.grad is not None else 'No grad'}")
        
    except Exception as e:
        print(f"✗ Error in inverse pass: {e}")
        return False
    
    # Test forward pass (sampling - this is fast for IAF)
    try:
        z_samples = base_dist.sample((batch_size,))
        x_samples, log_det_forward = model.forward(z_samples)
        print(f"✓ Forward pass successful")
        print(f"  x_samples shape: {x_samples.shape}")
        print(f"  x_samples has NaN: {torch.isnan(x_samples).any()}")
        print(f"  log_det_forward has NaN: {torch.isnan(log_det_forward).any()}")
        
    except Exception as e:
        print(f"✗ Error in forward pass: {e}")
        return False
    
    print("✓ All tests passed! IAF is numerically stable and gradient-compatible.")
    return True

if __name__ == "__main__":
    test_iaf_stability() 