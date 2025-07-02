#!/usr/bin/env python3
"""
Test script to demonstrate the benefits of batch normalization in normalizing flows.
Compares training stability and convergence for MAF with and without batch norm.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from src.flows.autoregressive import MaskedAutoregressiveFlow
from src.utils import get_two_moons_data
from src.flows.autoregressive import MADE, MaskedLinear

def create_maf_without_batch_norm(dim, hidden_dim):
    """Create MAF without batch normalization by modifying the MADE network."""
    class MADENoBN(MADE):
        def create_network(self):
            """Creates a deeper neural network with masked linear layers but NO batch normalization."""
            layers = []
            
            first_layer = MaskedLinear(
                self.input_dim, 
                self.hidden_dim, 
                mask=self.masks[0]
            )
            layers.append(first_layer)
            layers.append(nn.ReLU())
            
            second_layer = nn.Linear(self.hidden_dim, self.hidden_dim)
            layers.append(second_layer)
            layers.append(nn.ReLU())
            
            third_layer = nn.Linear(self.hidden_dim, self.hidden_dim)
            layers.append(third_layer)
            layers.append(nn.ReLU())
            
            final_layer = MaskedLinear(
                self.hidden_dim, 
                self.input_dim * self.output_dim_multiplier, 
                mask=self.masks[1]
            )
            layers.append(final_layer)
            
            # Initialize weights
            nn.init.xavier_normal_(first_layer.weight, gain=1.0)
            if first_layer.bias is not None:
                nn.init.zeros_(first_layer.bias)
            
            nn.init.xavier_normal_(second_layer.weight, gain=1.0)
            if second_layer.bias is not None:
                nn.init.zeros_(second_layer.bias)
                
            nn.init.xavier_normal_(third_layer.weight, gain=1.0)
            if third_layer.bias is not None:
                nn.init.zeros_(third_layer.bias)
            
            nn.init.normal_(final_layer.weight, mean=0.0, std=0.05)
            if final_layer.bias is not None:
                nn.init.normal_(final_layer.bias, mean=0.0, std=0.005)
            
            return nn.Sequential(*layers)
    
    class MaskedAutoregressiveFlowNoBN(MaskedAutoregressiveFlow):
        def __init__(self, dim, hidden_dim=64):
            super(MaskedAutoregressiveFlow, self).__init__()
            self.dim = dim
            self.conditioner = MADENoBN(dim, hidden_dim, 2)
    
    return MaskedAutoregressiveFlowNoBN(dim, hidden_dim)

def train_flow(flow, data, n_epochs=100, lr=1e-3, batch_size=512):
    """Train a flow and return the loss history."""
    optimizer = optim.Adam(flow.parameters(), lr=lr)
    base_dist = torch.distributions.Normal(0, 1)
    
    losses = []
    
    for epoch in range(n_epochs):
        # Sample random batches
        indices = torch.randperm(len(data))[:batch_size]
        batch = data[indices]
        
        optimizer.zero_grad()
        
        # Compute loss
        z, log_det = flow.inverse(batch)
        log_p_z = base_dist.log_prob(z).sum(dim=1)
        loss = -(log_p_z + log_det).mean()
        
        loss.backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(flow.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        losses.append(loss.item())
        
        if epoch % 20 == 0:
            print(f"Epoch {epoch}/{n_epochs}, Loss: {loss.item():.4f}")
    
    return losses

def main():
    print("Testing Batch Normalization Benefits in Normalizing Flows")
    print("=" * 60)
    
    # Generate data
    print("Generating two moons dataset...")
    data = get_two_moons_data(n_samples=10000, noise=0.1)
    data = torch.tensor(data, dtype=torch.float32)
    
    # Training parameters
    dim = 2
    hidden_dim = 64
    n_epochs = 100
    lr = 1e-3
    batch_size = 512
    
    print(f"Training parameters: dim={dim}, hidden_dim={hidden_dim}, epochs={n_epochs}")
    print(f"Learning rate: {lr}, Batch size: {batch_size}")
    print()
    
    # Train MAF with batch normalization
    print("Training MAF WITH batch normalization...")
    torch.manual_seed(42)
    maf_with_bn = MaskedAutoregressiveFlow(dim, hidden_dim)
    losses_with_bn = train_flow(maf_with_bn, data, n_epochs, lr, batch_size)
    
    print()
    
    # Train MAF without batch normalization
    print("Training MAF WITHOUT batch normalization...")
    torch.manual_seed(42)
    maf_without_bn = create_maf_without_batch_norm(dim, hidden_dim)
    losses_without_bn = train_flow(maf_without_bn, data, n_epochs, lr, batch_size)
    
    # Plot results
    plt.figure(figsize=(12, 8))
    
    # Loss comparison
    plt.subplot(2, 2, 1)
    plt.plot(losses_with_bn, label='With Batch Norm', color='blue', linewidth=2)
    plt.plot(losses_without_bn, label='Without Batch Norm', color='red', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Loss difference
    plt.subplot(2, 2, 2)
    loss_diff = np.array(losses_with_bn) - np.array(losses_without_bn)
    plt.plot(loss_diff, color='green', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss Difference (With BN - Without BN)')
    plt.title('Loss Difference')
    plt.grid(True, alpha=0.3)
    plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    
    # Final loss comparison
    plt.subplot(2, 2, 3)
    final_losses = [losses_with_bn[-1], losses_without_bn[-1]]
    labels = ['With Batch Norm', 'Without Batch Norm']
    colors = ['blue', 'red']
    bars = plt.bar(labels, final_losses, color=colors, alpha=0.7)
    plt.ylabel('Final Loss')
    plt.title('Final Loss Comparison')
    plt.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, loss in zip(bars, final_losses):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{loss:.4f}', ha='center', va='bottom')
    
    # Loss stability (standard deviation of last 20 epochs)
    plt.subplot(2, 2, 4)
    stability_with_bn = np.std(losses_with_bn[-20:])
    stability_without_bn = np.std(losses_without_bn[-20:])
    stabilities = [stability_with_bn, stability_without_bn]
    bars = plt.bar(labels, stabilities, color=colors, alpha=0.7)
    plt.ylabel('Loss Stability (Std Dev of Last 20 Epochs)')
    plt.title('Training Stability Comparison')
    plt.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, stability in zip(bars, stabilities):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001, 
                f'{stability:.4f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('batch_norm_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print summary statistics
    print("\n" + "=" * 60)
    print("SUMMARY STATISTICS")
    print("=" * 60)
    print(f"Final Loss (With BN):     {losses_with_bn[-1]:.4f}")
    print(f"Final Loss (Without BN):  {losses_without_bn[-1]:.4f}")
    print(f"Loss Improvement:         {losses_without_bn[-1] - losses_with_bn[-1]:.4f}")
    print()
    print(f"Stability (With BN):      {stability_with_bn:.4f}")
    print(f"Stability (Without BN):   {stability_without_bn:.4f}")
    print(f"Stability Improvement:    {stability_without_bn - stability_with_bn:.4f}")
    print()
    print("Lower values are better for both loss and stability.")
    print("Batch normalization typically improves both metrics.")

if __name__ == "__main__":
    main() 