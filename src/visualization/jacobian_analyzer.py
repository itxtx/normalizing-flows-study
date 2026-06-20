"""
Jacobian analysis tools for normalizing flows.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Optional, Dict, List
import warnings

from src.flows.flow.flow import Flow


class JacobianAnalyzer:
    """
    Tools for analyzing Jacobian properties of normalizing flows.
    
    Provides methods for:
    - Eigenvalue spectrum analysis
    - Condition number monitoring and visualization
    - Gradient flow magnitude and direction analysis
    """
    
    def __init__(self, device: str = 'cpu'):
        """
        Initialize the Jacobian analyzer.
        
        Args:
            device: Device to run computations on ('cpu' or 'cuda')
        """
        self.device = device
    
    def compute_jacobian(
        self,
        flow: Flow,
        x: torch.Tensor,
        create_graph: bool = True
    ) -> torch.Tensor:
        """
        Compute the Jacobian matrix of the flow at given points.
        
        Args:
            flow: The normalizing flow
            x: Input points (batch_size, dim)
            create_graph: Whether to create computation graph for higher-order derivatives
            
        Returns:
            Jacobian tensor (batch_size, dim, dim)
        """
        batch_size, dim = x.shape
        x = x.requires_grad_(True)
        
        # Compute forward pass
        y, _ = flow.forward(x)
        
        # Compute Jacobian for each sample in the batch
        jacobians = []
        
        for i in range(batch_size):
            jacobian_i = []
            
            for j in range(dim):
                # Compute gradient of j-th output w.r.t. all inputs
                grad_outputs = torch.zeros_like(y)
                grad_outputs[i, j] = 1.0
                
                grads = torch.autograd.grad(
                    outputs=y,
                    inputs=x,
                    grad_outputs=grad_outputs,
                    create_graph=create_graph,
                    retain_graph=True,
                    only_inputs=True
                )[0]
                
                jacobian_i.append(grads[i])
            
            jacobians.append(torch.stack(jacobian_i, dim=0))
        
        return torch.stack(jacobians, dim=0)
    
    def compute_eigenvalue_spectrum(
        self,
        flow: Flow,
        x: torch.Tensor,
        return_eigenvectors: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Compute eigenvalue spectrum of the Jacobian.
        
        Args:
            flow: The normalizing flow
            x: Input points (batch_size, dim)
            return_eigenvectors: Whether to return eigenvectors
            
        Returns:
            Eigenvalues (batch_size, dim) and optionally eigenvectors (batch_size, dim, dim)
        """
        flow.eval()
        
        # Compute Jacobian
        jacobian = self.compute_jacobian(flow, x, create_graph=False)
        
        # Compute eigenvalues and eigenvectors
        if return_eigenvectors:
            eigenvals, eigenvecs = torch.linalg.eig(jacobian)
            return eigenvals, eigenvecs
        else:
            eigenvals = torch.linalg.eigvals(jacobian)
            return eigenvals, None
    
    def compute_condition_numbers(
        self,
        flow: Flow,
        x: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute condition numbers of the Jacobian matrices.
        
        Args:
            flow: The normalizing flow
            x: Input points (batch_size, dim)
            
        Returns:
            Condition numbers (batch_size,)
        """
        flow.eval()
        
        # Compute Jacobian
        jacobian = self.compute_jacobian(flow, x, create_graph=False)
        
        # Compute condition numbers using SVD
        try:
            # Compute singular values
            U, S, Vh = torch.linalg.svd(jacobian)
            
            # Condition number is ratio of largest to smallest singular value
            condition_numbers = S[:, 0] / (S[:, -1] + 1e-12)  # Add small epsilon for stability
            
        except RuntimeError as e:
            warnings.warn(f"SVD computation failed: {e}. Using eigenvalue-based condition number.")
            
            # Fallback to eigenvalue-based condition number
            eigenvals, _ = self.compute_eigenvalue_spectrum(flow, x)
            eigenvals_real = torch.real(eigenvals)
            
            max_eigenvals = torch.max(torch.abs(eigenvals_real), dim=1)[0]
            min_eigenvals = torch.min(torch.abs(eigenvals_real) + 1e-12, dim=1)[0]
            condition_numbers = max_eigenvals / min_eigenvals
        
        return condition_numbers
    
    def plot_eigenvalue_spectrum(
        self,
        flow: Flow,
        x: torch.Tensor,
        figsize: Tuple[int, int] = (12, 4),
        show_unit_circle: bool = True
    ) -> plt.Figure:
        """
        Plot eigenvalue spectrum in the complex plane.
        
        Args:
            flow: The normalizing flow
            x: Input points (batch_size, dim)
            figsize: Figure size
            show_unit_circle: Whether to show unit circle for stability reference
            
        Returns:
            matplotlib Figure object
        """
        eigenvals, _ = self.compute_eigenvalue_spectrum(flow, x)
        
        # Convert to numpy for plotting
        eigenvals_np = eigenvals.detach().cpu().numpy()
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Plot eigenvalues in complex plane
        real_parts = np.real(eigenvals_np.flatten())
        imag_parts = np.imag(eigenvals_np.flatten())
        
        ax1.scatter(real_parts, imag_parts, alpha=0.6, s=20)
        ax1.set_xlabel('Real Part')
        ax1.set_ylabel('Imaginary Part')
        ax1.set_title('Eigenvalue Spectrum (Complex Plane)')
        ax1.grid(True, alpha=0.3)
        ax1.set_aspect('equal')
        
        if show_unit_circle:
            circle = plt.Circle((0, 0), 1, fill=False, color='red', linestyle='--', alpha=0.7)
            ax1.add_patch(circle)
            ax1.legend(['Eigenvalues', 'Unit Circle'])
        
        # Plot eigenvalue magnitudes
        magnitudes = np.abs(eigenvals_np)
        
        ax2.hist(magnitudes.flatten(), bins=30, alpha=0.7, edgecolor='black')
        ax2.axvline(x=1.0, color='red', linestyle='--', alpha=0.7, label='Unit magnitude')
        ax2.set_xlabel('Eigenvalue Magnitude')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Eigenvalue Magnitude Distribution')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        plt.tight_layout()
        return fig
    
    def plot_condition_number_evolution(
        self,
        flow_layers: List[Flow],
        x: torch.Tensor,
        figsize: Tuple[int, int] = (10, 6)
    ) -> plt.Figure:
        """
        Plot condition number evolution through flow layers.
        
        Args:
            flow_layers: List of flow layers
            x: Input points (batch_size, dim)
            figsize: Figure size
            
        Returns:
            matplotlib Figure object
        """
        condition_numbers = []
        layer_names = []
        
        # Compute condition numbers for each layer
        current_x = x.clone()
        
        for i, layer in enumerate(flow_layers):
            layer.eval()
            
            # Compute condition number for current layer
            cond_nums = self.compute_condition_numbers(layer, current_x)
            condition_numbers.append(cond_nums.detach().cpu().numpy())
            layer_names.append(f'Layer {i+1}')
            
            # Apply layer transformation for next iteration
            with torch.no_grad():
                current_x, _ = layer.forward(current_x)
        
        # Create plot
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize)
        
        # Plot condition numbers for each layer
        positions = np.arange(len(layer_names))
        
        # Box plot showing distribution across batch
        condition_data = [cond_nums for cond_nums in condition_numbers]
        bp = ax1.boxplot(condition_data, positions=positions, patch_artist=True)
        
        # Color the boxes
        colors = plt.cm.viridis(np.linspace(0, 1, len(layer_names)))
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax1.set_xticks(positions)
        ax1.set_xticklabels(layer_names)
        ax1.set_ylabel('Condition Number')
        ax1.set_title('Condition Number Distribution by Layer')
        ax1.set_yscale('log')
        ax1.grid(True, alpha=0.3)
        
        # Plot mean condition numbers
        mean_condition_numbers = [np.mean(cond_nums) for cond_nums in condition_numbers]
        ax2.plot(positions, mean_condition_numbers, 'o-', linewidth=2, markersize=8)
        ax2.set_xticks(positions)
        ax2.set_xticklabels(layer_names)
        ax2.set_ylabel('Mean Condition Number')
        ax2.set_title('Mean Condition Number Evolution')
        ax2.set_yscale('log')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def analyze_gradient_flow(
        self,
        flow: Flow,
        x: torch.Tensor,
        target: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Analyze gradient flow magnitude and direction.
        
        Args:
            flow: The normalizing flow
            x: Input points (batch_size, dim)
            target: Target values for gradient computation (if None, uses log-likelihood)
            
        Returns:
            Dictionary containing gradient analysis results
        """
        flow.train()  # Enable gradients
        x = x.requires_grad_(True)
        
        if target is None:
            # Use negative log-likelihood as default target
            base_dist = torch.distributions.MultivariateNormal(
                torch.zeros(x.shape[1], device=x.device),
                torch.eye(x.shape[1], device=x.device)
            )
            target = -flow.log_prob(x, base_dist)
        
        # Compute gradients
        loss = target.mean()
        gradients = torch.autograd.grad(
            outputs=loss,
            inputs=x,
            create_graph=True,
            retain_graph=True
        )[0]
        
        # Analyze gradient properties
        gradient_magnitudes = torch.norm(gradients, dim=1)
        gradient_directions = gradients / (gradient_magnitudes.unsqueeze(1) + 1e-12)
        
        # Compute gradient statistics
        results = {
            'gradients': gradients,
            'magnitudes': gradient_magnitudes,
            'directions': gradient_directions,
            'mean_magnitude': gradient_magnitudes.mean(),
            'std_magnitude': gradient_magnitudes.std(),
            'max_magnitude': gradient_magnitudes.max(),
            'min_magnitude': gradient_magnitudes.min()
        }
        
        return results
    
    def plot_gradient_flow_analysis(
        self,
        flow: Flow,
        x: torch.Tensor,
        figsize: Tuple[int, int] = (15, 5)
    ) -> plt.Figure:
        """
        Plot gradient flow analysis results.
        
        Args:
            flow: The normalizing flow
            x: Input points (batch_size, dim)
            figsize: Figure size
            
        Returns:
            matplotlib Figure object
        """
        # Analyze gradient flow
        grad_analysis = self.analyze_gradient_flow(flow, x)
        
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        
        # Plot gradient magnitudes
        magnitudes = grad_analysis['magnitudes'].detach().cpu().numpy()
        axes[0].hist(magnitudes, bins=30, alpha=0.7, edgecolor='black')
        axes[0].axvline(x=magnitudes.mean(), color='red', linestyle='--', 
                       label=f'Mean: {magnitudes.mean():.3f}')
        axes[0].set_xlabel('Gradient Magnitude')
        axes[0].set_ylabel('Frequency')
        axes[0].set_title('Gradient Magnitude Distribution')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Plot gradient directions (for 2D case)
        if x.shape[1] == 2:
            directions = grad_analysis['directions'].detach().cpu().numpy()
            x_pos = x.detach().cpu().numpy()
            
            # Subsample for visualization
            n_arrows = min(50, len(x_pos))
            indices = np.random.choice(len(x_pos), n_arrows, replace=False)
            
            axes[1].quiver(
                x_pos[indices, 0], x_pos[indices, 1],
                directions[indices, 0], directions[indices, 1],
                magnitudes[indices], scale=20, alpha=0.7, cmap='viridis'
            )
            axes[1].set_xlabel('x₁')
            axes[1].set_ylabel('x₂')
            axes[1].set_title('Gradient Direction Field')
            axes[1].set_aspect('equal')
        else:
            # For higher dimensions, show gradient component correlations
            gradients = grad_analysis['gradients'].detach().cpu().numpy()
            corr_matrix = np.corrcoef(gradients.T)
            
            im = axes[1].imshow(corr_matrix, cmap='RdBu', vmin=-1, vmax=1)
            axes[1].set_title('Gradient Component Correlations')
            axes[1].set_xlabel('Dimension')
            axes[1].set_ylabel('Dimension')
            plt.colorbar(im, ax=axes[1])
        
        # Plot magnitude vs position (for 2D case)
        if x.shape[1] == 2:
            x_pos = x.detach().cpu().numpy()
            scatter = axes[2].scatter(
                x_pos[:, 0], x_pos[:, 1], c=magnitudes, 
                cmap='viridis', alpha=0.7, s=20
            )
            axes[2].set_xlabel('x₁')
            axes[2].set_ylabel('x₂')
            axes[2].set_title('Gradient Magnitude by Position')
            plt.colorbar(scatter, ax=axes[2])
        else:
            # For higher dimensions, show magnitude vs first principal component
            from sklearn.decomposition import PCA
            pca = PCA(n_components=1)
            x_pca = pca.fit_transform(x.detach().cpu().numpy())
            
            axes[2].scatter(x_pca.flatten(), magnitudes, alpha=0.7, s=20)
            axes[2].set_xlabel('First Principal Component')
            axes[2].set_ylabel('Gradient Magnitude')
            axes[2].set_title('Gradient Magnitude vs PC1')
        
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def compute_jacobian_determinant_accuracy(
        self,
        flow: Flow,
        x: torch.Tensor,
        tolerance: float = 1e-6
    ) -> Dict[str, float]:
        """
        Test numerical accuracy of Jacobian determinant computation.
        
        Args:
            flow: The normalizing flow
            x: Input points (batch_size, dim)
            tolerance: Tolerance for accuracy check
            
        Returns:
            Dictionary with accuracy metrics
        """
        flow.eval()
        
        # Get log-determinant from flow
        _, log_det_flow = flow.forward(x)
        
        # Compute Jacobian manually and get determinant
        jacobian = self.compute_jacobian(flow, x, create_graph=False)
        log_det_manual = torch.logdet(jacobian)
        
        # Compare results
        abs_error = torch.abs(log_det_flow - log_det_manual)
        rel_error = abs_error / (torch.abs(log_det_manual) + 1e-12)
        
        # Compute accuracy metrics
        results = {
            'mean_abs_error': abs_error.mean().item(),
            'max_abs_error': abs_error.max().item(),
            'mean_rel_error': rel_error.mean().item(),
            'max_rel_error': rel_error.max().item(),
            'accuracy_within_tolerance': (abs_error < tolerance).float().mean().item()
        }
        
        return results