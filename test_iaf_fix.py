import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
import sys
sys.path.append('src')
from src.flows.autoregressive.inverse_autoregressive_flow import InverseAutoregressiveFlow
from src.models.normalizing_flow_model import NormalizingFlowModel

def test_iaf_stability_fix():
    """Test if the IAF stability fixes work correctly."""
    
    print("Testing IAF stability fixes...")
    
    # Configuration
    input_dim = 2
    hidden_dim = 64
    num_flows = 3
    batch_size = 100
    
    # Create model
    flows = [InverseAutoregressiveFlow(dim=input_dim, hidden_dim=hidden_dim) for _ in range(num_flows)]
    model = NormalizingFlowModel(flows)
    
    # Create base distribution
    base_dist = MultivariateNormal(torch.zeros(input_dim), torch.eye(input_dim))
    
    # Test 1: Forward pass (sampling)
    print("\n1. Testing forward pass (sampling)...")
    z_samples = base_dist.sample((batch_size,))
    
    with torch.no_grad():
        x_samples, log_det_fwd = model.forward(z_samples)
        
        print(f"   z range: [{z_samples.min():.3f}, {z_samples.max():.3f}]")
        print(f"   x range: [{x_samples.min():.3f}, {x_samples.max():.3f}]")
        print(f"   x mean: {x_samples.mean(dim=0)}")
        print(f"   x std: {x_samples.std(dim=0)}")
        print(f"   log_det range: [{log_det_fwd.min():.3f}, {log_det_fwd.max():.3f}]")
        print(f"   Has NaN: {torch.isnan(x_samples).any()}")
        print(f"   Has Inf: {torch.isinf(x_samples).any()}")
    
    # Test 2: Inverse pass (density estimation)
    print("\n2. Testing inverse pass (density estimation)...")
    data_subset = x_samples[:50]  # Use subset for efficiency
    
    with torch.no_grad():
        z_inv, log_det_inv = model.inverse(data_subset)
        
        print(f"   z range: [{z_inv.min():.3f}, {z_inv.max():.3f}]")
        print(f"   z mean: {z_inv.mean(dim=0)}")
        print(f"   z std: {z_inv.std(dim=0)}")
        print(f"   log_det range: [{log_det_inv.min():.3f}, {log_det_inv.max():.3f}]")
        print(f"   Has NaN: {torch.isnan(z_inv).any()}")
        print(f"   Has Inf: {torch.isinf(z_inv).any()}")
    
    # Test 3: Round-trip consistency
    print("\n3. Testing round-trip consistency...")
    with torch.no_grad():
        z_roundtrip, _ = model.inverse(x_samples[:50])
        roundtrip_error = torch.mean((z_samples[:50] - z_roundtrip) ** 2).item()
        print(f"   Round-trip error: {roundtrip_error:.6f}")
        
        if roundtrip_error < 1.0:
            print("   ✓ Round-trip error is reasonable")
        else:
            print("   ⚠️  Round-trip error is still high")
    
    # Test 4: Identity check (should be close to identity initially)
    print("\n4. Testing identity transformation...")
    with torch.no_grad():
        identity_error = torch.mean((z_samples - x_samples) ** 2).item()
        print(f"   Identity error: {identity_error:.6f}")
        
        if identity_error < 1.0:
            print("   ✓ Model starts close to identity (expected for untrained model)")
        else:
            print("   ⚠️  Model is not close to identity")
    
    # Test 5: Gradient computation
    print("\n5. Testing gradient computation...")
    test_data = torch.randn(10, input_dim, requires_grad=True)
    
    try:
        z, log_det = model.inverse(test_data)
        log_prob_z = base_dist.log_prob(z)
        loss = -(log_prob_z + log_det).mean()
        loss.backward()
        
        print(f"   ✓ Gradient computation successful")
        print(f"   Grad has NaN: {torch.isnan(test_data.grad).any() if test_data.grad is not None else 'No grad'}")
        
    except Exception as e:
        print(f"   ✗ Gradient computation failed: {e}")
    
    print("\n=== Summary ===")
    if (not torch.isnan(x_samples).any() and 
        not torch.isnan(z_inv).any() and 
        roundtrip_error < 10.0 and 
        identity_error < 10.0):
        print("✓ IAF stability fixes appear to be working!")
    else:
        print("⚠️  Some issues remain - further investigation needed")

if __name__ == "__main__":
    test_iaf_stability_fix() 