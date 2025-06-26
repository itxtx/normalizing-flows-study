import torch
from sklearn.datasets import make_moons
import numpy as np
import matplotlib.pyplot as plt

def get_two_moons_data(n_samples=10000, noise=0.05):
    """
    Generates and returns the two-moons dataset as a PyTorch tensor.

    Args:
        n_samples (int): The total number of points to generate.
        noise (float): Standard deviation of Gaussian noise added to the data.

    Returns:
        torch.Tensor: A tensor of shape (n_samples, 2) containing the data.
    """
    # Generate data using scikit-learn
    moons, _ = make_moons(n_samples=n_samples, noise=noise, random_state=42)
    
    # Convert to PyTorch tensor
    return torch.from_numpy(moons.astype(np.float32))


def plot_enhanced_visualizations(model, data, base_dist, flow_type, device):
    """
    Generates and plots enhanced visualizations for a trained flow model,
    including the learned probability density.
    """
    model.eval()
    with torch.no_grad():
        # --- 1. Create a figure with subplots ---
        fig = plt.figure(figsize=(20, 6))
        gs = fig.add_gridspec(1, 3)

        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[0, 1])
        ax3 = fig.add_subplot(gs[0, 2])
        
        fig.suptitle(f'Enhanced Visualization for {flow_type}', fontsize=20)

        # --- 2. Plot Original vs. Generated Data ---
        ax1.set_title('Original vs. Generated Data', fontsize=14)
        ax1.scatter(data.cpu()[:, 0], data.cpu()[:, 1], s=15, alpha=0.5, label='Original Data')
        
        # Generate new samples for comparison
        z_samples = base_dist.sample((data.shape[0],))
        x_generated, _ = model.forward(z_samples)
        x_generated = x_generated.cpu().numpy()
        
        ax1.scatter(x_generated[:, 0], x_generated[:, 1], s=15, alpha=0.5, c='orange', label='Generated Data')
        ax1.legend()
        ax1.set_xlabel('x1')
        ax1.set_ylabel('x2')
        ax1.axis('equal')
        ax1.grid(True, linestyle='--', alpha=0.6)

        # --- 3. Plot Learned Probability Density ---
        ax2.set_title('Learned Probability Density', fontsize=14)
        
        # Create a grid of points to evaluate the density
        x_range = np.linspace(-3, 5, 100)
        y_range = np.linspace(-3, 3, 100)
        grid_x, grid_y = np.meshgrid(x_range, y_range)
        grid_points = torch.tensor(np.c_[grid_x.ravel(), grid_y.ravel()], dtype=torch.float32).to(device)
        
        # Evaluate log probability on the grid
        z, log_det = model.inverse(grid_points)
        log_prob = base_dist.log_prob(z) + log_det
        prob = torch.exp(log_prob).reshape(100, 100)
        
        # Plot the density contour
        contour = ax2.contourf(grid_x, grid_y, prob.cpu().numpy(), cmap='viridis', levels=20)
        plt.colorbar(contour, ax=ax2, label='Probability Density')
        
        # Overlay the original data for reference
        ax2.scatter(data.cpu()[:, 0], data.cpu()[:, 1], s=5, c='white', alpha=0.4, label='Original Data')
        ax2.set_xlabel('x1')
        ax2.set_ylabel('x2')
        ax2.axis('equal')
        ax2.legend()

        # --- 4. Plot Transformation Flow from Z -> X ---
        ax3.set_title('Transformation Flow: Z -> X', fontsize=14)
        
        # Start with base distribution samples arranged in a grid
        z_x = torch.linspace(-3, 3, 15)
        z_y = torch.linspace(-3, 3, 15)
        z_grid_x, z_grid_y = torch.meshgrid(z_x, z_y, indexing='ij')
        z_grid = torch.stack([z_grid_x.ravel(), z_grid_y.ravel()], dim=1).to(device)

        ax3.scatter(z_grid.cpu()[:, 0], z_grid.cpu()[:, 1], s=15, c='red', alpha=0.7, label='Base (Z)')
        
        # Transform the grid to the data space
        x_flow, _ = model.forward(z_grid)
        x_flow = x_flow.cpu()
        
        ax3.scatter(x_flow[:, 0], x_flow[:, 1], s=15, c='blue', alpha=0.7, label='Transformed (X)')
        
        # Draw lines to show the mapping
        for i in range(len(z_grid)):
            ax3.plot([z_grid[i, 0].cpu(), x_flow[i, 0]], [z_grid[i, 1].cpu(), x_flow[i, 1]], 'k-', lw=0.5, alpha=0.3)
            
        ax3.legend()
        ax3.set_xlabel('x1')
        ax3.set_ylabel('x2')
        ax3.axis('equal')
        ax3.grid(True, linestyle='--', alpha=0.6)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()