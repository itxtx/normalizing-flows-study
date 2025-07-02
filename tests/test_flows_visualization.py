import torch
import numpy as np
import matplotlib.pyplot as plt
from src.flows.autoregressive.masked_autoregressive_flow import MaskedAutoregressiveFlow
from src.flows.autoregressive.inverse_autoregressive_flow import InverseAutoregressiveFlow

def test_autoregressive_flows():
    """Test autoregressive flows to see if they produce reasonable outputs."""
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Create flows
    dim = 2
    hidden_dim = 32
    
    maf = MaskedAutoregressiveFlow(dim, hidden_dim)
    iaf = InverseAutoregressiveFlow(dim, hidden_dim)
    
    # Create base distribution
    base_dist = torch.distributions.MultivariateNormal(
        torch.zeros(dim),
        torch.eye(dim)
    )
    
    # Generate samples from base distribution
    num_samples = 1000
    z_samples = base_dist.sample((num_samples,))
    
    print("Testing MAF...")
    # Test MAF
    with torch.no_grad():
        maf_x_samples, maf_log_det = maf.forward(z_samples)
        maf_z_back, maf_log_det_inv = maf.inverse(maf_x_samples)
        
        print(f"MAF forward - x range: [{maf_x_samples.min():.3f}, {maf_x_samples.max():.3f}]")
        print(f"MAF forward - x mean: {maf_x_samples.mean(dim=0)}")
        print(f"MAF forward - x std: {maf_x_samples.std(dim=0)}")
        print(f"MAF inverse - z range: [{maf_z_back.min():.3f}, {maf_z_back.max():.3f}]")
        print(f"MAF inverse - z mean: {maf_z_back.mean(dim=0)}")
        print(f"MAF inverse - z std: {maf_z_back.std(dim=0)}")
    
    print("\nTesting IAF...")
    # Test IAF
    with torch.no_grad():
        iaf_x_samples, iaf_log_det = iaf.forward(z_samples)
        iaf_z_back, iaf_log_det_inv = iaf.inverse(iaf_x_samples)
        
        print(f"IAF forward - x range: [{iaf_x_samples.min():.3f}, {iaf_x_samples.max():.3f}]")
        print(f"IAF forward - x mean: {iaf_x_samples.mean(dim=0)}")
        print(f"IAF forward - x std: {iaf_x_samples.std(dim=0)}")
        print(f"IAF inverse - z range: [{iaf_z_back.min():.3f}, {iaf_z_back.max():.3f}]")
        print(f"IAF inverse - z mean: {iaf_z_back.mean(dim=0)}")
        print(f"IAF inverse - z std: {iaf_z_back.std(dim=0)}")
    
    # Create visualization
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # MAF plots
    axes[0, 0].scatter(z_samples[:, 0], z_samples[:, 1], alpha=0.6, s=1)
    axes[0, 0].set_title('MAF: Base Distribution (z)')
    axes[0, 0].set_xlabel('z1')
    axes[0, 0].set_ylabel('z2')
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].scatter(maf_x_samples[:, 0], maf_x_samples[:, 1], alpha=0.6, s=1, color='red')
    axes[0, 1].set_title('MAF: Generated Samples (x)')
    axes[0, 1].set_xlabel('x1')
    axes[0, 1].set_ylabel('x2')
    axes[0, 1].grid(True, alpha=0.3)
    
    axes[0, 2].scatter(maf_z_back[:, 0], maf_z_back[:, 1], alpha=0.6, s=1, color='green')
    axes[0, 2].set_title('MAF: Latent Space (z back)')
    axes[0, 2].set_xlabel('z1')
    axes[0, 2].set_ylabel('z2')
    axes[0, 2].grid(True, alpha=0.3)
    
    # IAF plots
    axes[1, 0].scatter(z_samples[:, 0], z_samples[:, 1], alpha=0.6, s=1)
    axes[1, 0].set_title('IAF: Base Distribution (z)')
    axes[1, 0].set_xlabel('z1')
    axes[1, 0].set_ylabel('z2')
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].scatter(iaf_x_samples[:, 0], iaf_x_samples[:, 1], alpha=0.6, s=1, color='red')
    axes[1, 1].set_title('IAF: Generated Samples (x)')
    axes[1, 1].set_xlabel('x1')
    axes[1, 1].set_ylabel('x2')
    axes[1, 1].grid(True, alpha=0.3)
    
    axes[1, 2].scatter(iaf_z_back[:, 0], iaf_z_back[:, 1], alpha=0.6, s=1, color='green')
    axes[1, 2].set_title('IAF: Latent Space (z back)')
    axes[1, 2].set_xlabel('z1')
    axes[1, 2].set_ylabel('z2')
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    test_autoregressive_flows() 