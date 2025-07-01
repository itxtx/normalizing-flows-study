import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from torch.distributions import MultivariateNormal

def create_mask(dim, mask_type="alternating"):
    """Helper function to create masks for coupling layers."""
    mask = torch.zeros(dim)
    if mask_type == "alternating":
        mask[::2] = 1
    elif mask_type == "half":
        mask[:dim//2] = 1
    return mask

def get_two_moons_data(n_samples=1000, noise=0.1):
    """
    Generate two moons dataset.
    """
    X, _ = make_moons(n_samples=n_samples, noise=noise, random_state=42)
    return torch.FloatTensor(X)

def train_with_stability(model, optimizer, data_loader, epochs, base_dist, flow_type='MAF'):
    """
    Improved training loop with stability measures to prevent NaN values.
    """
    model.train()
    print(f"Starting training for {flow_type}...")
    for epoch in range(epochs):
        total_loss = 0
        num_batches = 0
        for i, x in enumerate(data_loader):
            optimizer.zero_grad()
            
            try:
                # The inverse pass maps data to the latent space for likelihood calculation.
                z, log_det = model.inverse(x)
                
                # Check for NaN values in intermediate computations
                if torch.isnan(z).any() or torch.isnan(log_det).any():
                    print(f"Warning: NaN detected in {flow_type} at epoch {epoch+1}, batch {i+1}")
                    continue
                
                # Calculate the negative log-likelihood loss
                # log p(x) = log p(z) + log|det(J)|
                log_prob_z = base_dist.log_prob(z)
                
                # Check for NaN in log probability
                if torch.isnan(log_prob_z).any():
                    print(f"Warning: NaN in log probability for {flow_type} at epoch {epoch+1}, batch {i+1}")
                    continue
                
                loss = -(log_prob_z + log_det).mean()
                
                # Check for NaN in loss
                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"Warning: Invalid loss value for {flow_type} at epoch {epoch+1}, batch {i+1}")
                    continue
                
                loss.backward()
                
                # Add gradient clipping to prevent exploding gradients
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
                
            except Exception as e:
                print(f"Error in {flow_type} at epoch {epoch+1}, batch {i+1}: {e}")
                continue
            
        if (epoch + 1) % 100 == 0 and num_batches > 0:
            print(f'Epoch {epoch+1}/{epochs}, Loss: {total_loss / num_batches:.4f}')
    print("Training finished.")

def diagnose_flow_model(model, data, base_dist, flow_type, device):
    """
    Diagnostic function to check if a flow model is working correctly.
    """
    model.eval()
    with torch.no_grad():
        # Test 1: Check if forward pass produces reasonable outputs
        z_test = base_dist.sample((100,))
        x_test, log_det_fwd = model.forward(z_test)
        
        print(f"\n=== {flow_type} Diagnostics ===")
        print(f"Forward pass - x range: [{x_test.min():.3f}, {x_test.max():.3f}]")
        print(f"Forward pass - x mean: {x_test.mean(dim=0)}")
        print(f"Forward pass - x std: {x_test.std(dim=0)}")
        print(f"Forward pass - log_det range: [{log_det_fwd.min():.3f}, {log_det_fwd.max():.3f}]")
        
        # Test 2: Check if inverse pass works
        subset_size = min(100, len(data))
        data_subset = data[:subset_size]
        z_inv, log_det_inv = model.inverse(data_subset)
        
        print(f"Inverse pass - z range: [{z_inv.min():.3f}, {z_inv.max():.3f}]")
        print(f"Inverse pass - z mean: {z_inv.mean(dim=0)}")
        print(f"Inverse pass - z std: {z_inv.std(dim=0)}")
        print(f"Inverse pass - log_det range: [{log_det_inv.min():.3f}, {log_det_inv.max():.3f}]")
        
        # Test 3: Check round-trip consistency
        z_roundtrip, _ = model.inverse(x_test)
        roundtrip_error = torch.mean((z_test - z_roundtrip) ** 2).item()
        print(f"Round-trip error (z → x → z): {roundtrip_error:.6f}")
        
        # Test 4: Check if the model is close to identity (untrained)
        identity_error = torch.mean((z_test - x_test) ** 2).item()
        print(f"Identity error (z ≈ x): {identity_error:.6f}")
        
        if identity_error < 0.1:
            print("⚠️  WARNING: Model appears to be close to identity transformation (possibly untrained)")
        
        if roundtrip_error > 1.0:
            print("⚠️  WARNING: High round-trip error - model may not be invertible")
        
        return {
            'forward_stats': {'range': (x_test.min().item(), x_test.max().item()), 
                            'mean': x_test.mean(dim=0).tolist(),
                            'std': x_test.std(dim=0).tolist()},
            'inverse_stats': {'range': (z_inv.min().item(), z_inv.max().item()),
                            'mean': z_inv.mean(dim=0).tolist(),
                            'std': z_inv.std(dim=0).tolist()},
            'roundtrip_error': roundtrip_error,
            'identity_error': identity_error
        }

def plot_enhanced_visualizations(model, data, base_dist, flow_type, device, num_samples=5000):
    """
    Enhanced visualization function for flow models.
    """
    model.eval()
    with torch.no_grad():
        # Generate samples from the flow
        z_samples = base_dist.sample((num_samples,))
        x_samples, _ = model.forward(z_samples)
        
        # Create subplots
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Plot 1: Original data
        axes[0].scatter(data[:, 0], data[:, 1], alpha=0.6, s=1)
        axes[0].set_title(f'Original Data')
        axes[0].set_xlabel('x1')
        axes[0].set_ylabel('x2')
        axes[0].grid(True, alpha=0.3)
        
        # Plot 2: Generated samples
        axes[1].scatter(x_samples[:, 0], x_samples[:, 1], alpha=0.6, s=1, color='red')
        axes[1].set_title(f'{flow_type} Generated Samples')
        axes[1].set_xlabel('x1')
        axes[1].set_ylabel('x2')
        axes[1].grid(True, alpha=0.3)
        
        # Plot 3: Latent space - handle autoregressive flows correctly
        if 'MAF' or 'IAF' in flow_type:
            # For autoregressive flows, use a subset of data for efficiency
            # and show the latent space mapping
            subset_size = min(1000, len(data))
            data_subset = data[:subset_size]
            z_data, _ = model.inverse(data_subset)
            axes[2].scatter(z_data[:, 0], z_data[:, 1], alpha=0.6, s=1, color='green')
            axes[2].set_title(f'{flow_type} Latent Space (Data → Z)')
        else:
            # For non-autoregressive flows, use full data
            z_data, _ = model.inverse(data)
            axes[2].scatter(z_data[:, 0], z_data[:, 1], alpha=0.6, s=1, color='green')
            axes[2].set_title(f'{flow_type} Latent Space')
        
        axes[2].set_xlabel('z1')
        axes[2].set_ylabel('z2')
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()