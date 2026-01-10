"""
Flow visualization toolkit for 2D transformations and density evolution.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import LogNorm
from typing import List, Optional, Tuple, Union
import warnings

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    warnings.warn("Plotly not available. Interactive visualizations will be disabled.")

from src.flows.flow.flow import Flow


class FlowVisualizer:
    """
    Comprehensive visualization toolkit for normalizing flows.
    
    Provides methods for:
    - 2D transformation visualization
    - Density evolution animation through flow layers
    - Interactive visualizations (when plotly is available)
    """
    
    def __init__(self, device: str = 'cpu'):
        """
        Initialize the flow visualizer.
        
        Args:
            device: Device to run computations on ('cpu' or 'cuda')
        """
        self.device = device
        
    def plot_2d_transformation(
        self,
        flow: Flow,
        grid_size: int = 100,
        xlim: Tuple[float, float] = (-3, 3),
        ylim: Tuple[float, float] = (-3, 3),
        base_dist: Optional[torch.distributions.Distribution] = None,
        figsize: Tuple[int, int] = (12, 5),
        show_grid: bool = True,
        show_samples: bool = True,
        num_samples: int = 1000
    ) -> plt.Figure:
        """
        Visualize how a 2D flow transforms space.
        
        Args:
            flow: The normalizing flow to visualize
            grid_size: Number of grid points along each axis
            xlim: X-axis limits for visualization
            ylim: Y-axis limits for visualization
            base_dist: Base distribution (defaults to standard normal)
            figsize: Figure size
            show_grid: Whether to show transformation grid
            show_samples: Whether to show sample transformations
            num_samples: Number of samples to show
            
        Returns:
            matplotlib Figure object
        """
        if base_dist is None:
            base_dist = torch.distributions.MultivariateNormal(
                torch.zeros(2), torch.eye(2)
            )
            
        flow.eval()
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Create grid for transformation visualization
        if show_grid:
            x = np.linspace(xlim[0], xlim[1], grid_size)
            y = np.linspace(ylim[0], ylim[1], grid_size)
            X, Y = np.meshgrid(x, y)
            
            # Transform grid points
            grid_points = torch.tensor(
                np.stack([X.flatten(), Y.flatten()], axis=1),
                dtype=torch.float32,
                device=self.device
            )
            
            with torch.no_grad():
                transformed_points, _ = flow.forward(grid_points)
                
            transformed_points = transformed_points.cpu().numpy()
            X_trans = transformed_points[:, 0].reshape(X.shape)
            Y_trans = transformed_points[:, 1].reshape(Y.shape)
            
            # Plot original grid
            ax1.contour(X, Y, X, levels=20, colors='blue', alpha=0.3, linewidths=0.5)
            ax1.contour(X, Y, Y, levels=20, colors='blue', alpha=0.3, linewidths=0.5)
            ax1.set_title('Original Space')
            ax1.set_xlabel('z₁')
            ax1.set_ylabel('z₂')
            ax1.grid(True, alpha=0.3)
            ax1.set_aspect('equal')
            
            # Plot transformed grid
            ax2.contour(X_trans, Y_trans, X, levels=20, colors='red', alpha=0.3, linewidths=0.5)
            ax2.contour(X_trans, Y_trans, Y, levels=20, colors='red', alpha=0.3, linewidths=0.5)
            ax2.set_title('Transformed Space')
            ax2.set_xlabel('x₁')
            ax2.set_ylabel('x₂')
            ax2.grid(True, alpha=0.3)
            ax2.set_aspect('equal')
        
        # Show sample transformations
        if show_samples:
            with torch.no_grad():
                z_samples = base_dist.sample((num_samples,)).to(self.device)
                x_samples, _ = flow.forward(z_samples)
                
            z_samples = z_samples.cpu().numpy()
            x_samples = x_samples.cpu().numpy()
            
            ax1.scatter(z_samples[:, 0], z_samples[:, 1], 
                       c='blue', alpha=0.6, s=10, label='Base samples')
            ax2.scatter(x_samples[:, 0], x_samples[:, 1], 
                       c='red', alpha=0.6, s=10, label='Transformed samples')
            
            ax1.legend()
            ax2.legend()
        
        plt.tight_layout()
        return fig
    
    def plot_density_evolution(
        self,
        flow_layers: List[Flow],
        base_dist: Optional[torch.distributions.Distribution] = None,
        grid_size: int = 100,
        xlim: Tuple[float, float] = (-3, 3),
        ylim: Tuple[float, float] = (-3, 3),
        figsize: Tuple[int, int] = (15, 4)
    ) -> plt.Figure:
        """
        Show density evolution through multiple flow layers.
        
        Args:
            flow_layers: List of flow layers to visualize
            base_dist: Base distribution
            grid_size: Grid resolution for density estimation
            xlim: X-axis limits
            ylim: Y-axis limits
            figsize: Figure size
            
        Returns:
            matplotlib Figure object
        """
        if base_dist is None:
            base_dist = torch.distributions.MultivariateNormal(
                torch.zeros(2), torch.eye(2)
            )
            
        num_layers = len(flow_layers)
        fig, axes = plt.subplots(1, num_layers + 1, figsize=figsize)
        
        # Ensure axes is always a list
        if num_layers == 0:
            axes = [axes]
        elif not isinstance(axes, (list, np.ndarray)):
            axes = [axes]
        
        # Create evaluation grid
        x = np.linspace(xlim[0], xlim[1], grid_size)
        y = np.linspace(ylim[0], ylim[1], grid_size)
        X, Y = np.meshgrid(x, y)
        grid_points = torch.tensor(
            np.stack([X.flatten(), Y.flatten()], axis=1),
            dtype=torch.float32,
            device=self.device
        )
        
        # Plot base distribution
        with torch.no_grad():
            log_prob_base = base_dist.log_prob(grid_points).cpu().numpy()
            
        prob_base = np.exp(log_prob_base).reshape(X.shape)
        im0 = axes[0].contourf(X, Y, prob_base, levels=20, cmap='viridis')
        axes[0].set_title('Base Distribution')
        axes[0].set_xlabel('z₁')
        axes[0].set_ylabel('z₂')
        plt.colorbar(im0, ax=axes[0])
        
        # Apply layers sequentially and plot
        current_points = grid_points.clone()
        
        for i, layer in enumerate(flow_layers):
            layer.eval()
            with torch.no_grad():
                current_points, log_det = layer.forward(current_points)
                
                # Compute density at transformed points
                # This is approximate - we're showing where the mass goes
                transformed_points = current_points.cpu().numpy()
                
            # Create density plot by binning transformed points
            H, xedges, yedges = np.histogram2d(
                transformed_points[:, 0], 
                transformed_points[:, 1],
                bins=50,
                range=[[xlim[0], xlim[1]], [ylim[0], ylim[1]]]
            )
            
            # Smooth the histogram
            try:
                from scipy.ndimage import gaussian_filter
                H_smooth = gaussian_filter(H.T, sigma=1.0)
            except ImportError:
                # Fallback to unsmoothed histogram if scipy not available
                H_smooth = H.T
            
            X_hist = (xedges[:-1] + xedges[1:]) / 2
            Y_hist = (yedges[:-1] + yedges[1:]) / 2
            X_hist, Y_hist = np.meshgrid(X_hist, Y_hist)
            
            im = axes[i + 1].contourf(X_hist, Y_hist, H_smooth, levels=20, cmap='viridis')
            axes[i + 1].set_title(f'After Layer {i + 1}')
            axes[i + 1].set_xlabel('x₁')
            axes[i + 1].set_ylabel('x₂')
            plt.colorbar(im, ax=axes[i + 1])
        
        plt.tight_layout()
        return fig
    
    def animate_density_evolution(
        self,
        flow_layers: List[Flow],
        base_dist: Optional[torch.distributions.Distribution] = None,
        num_samples: int = 2000,
        xlim: Tuple[float, float] = (-4, 4),
        ylim: Tuple[float, float] = (-4, 4),
        interval: int = 800,
        figsize: Tuple[int, int] = (8, 8)
    ) -> animation.FuncAnimation:
        """
        Create an animation showing density evolution through flow layers.
        
        Args:
            flow_layers: List of flow layers
            base_dist: Base distribution
            num_samples: Number of samples to animate
            xlim: X-axis limits
            ylim: Y-axis limits
            interval: Animation interval in milliseconds
            figsize: Figure size
            
        Returns:
            matplotlib FuncAnimation object
        """
        if base_dist is None:
            base_dist = torch.distributions.MultivariateNormal(
                torch.zeros(2), torch.eye(2)
            )
            
        fig, ax = plt.subplots(figsize=figsize)
        
        # Generate samples
        with torch.no_grad():
            samples = base_dist.sample((num_samples,)).to(self.device)
            
        # Store all intermediate transformations
        all_samples = [samples.cpu().numpy()]
        current_samples = samples.clone()
        
        for layer in flow_layers:
            layer.eval()
            with torch.no_grad():
                current_samples, _ = layer.forward(current_samples)
                all_samples.append(current_samples.cpu().numpy())
        
        # Set up the plot
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        
        scat = ax.scatter([], [], s=10, alpha=0.6)
        title = ax.set_title('')
        
        def animate(frame):
            if frame < len(all_samples):
                data = all_samples[frame]
                scat.set_offsets(data)
                
                if frame == 0:
                    title.set_text('Base Distribution')
                    scat.set_color('blue')
                else:
                    title.set_text(f'After Layer {frame}')
                    scat.set_color('red')
            
            return scat, title
        
        anim = animation.FuncAnimation(
            fig, animate, frames=len(all_samples),
            interval=interval, blit=False, repeat=True
        )
        
        return anim
    
    def plot_interactive_2d_transformation(
        self,
        flow: Flow,
        grid_size: int = 50,
        xlim: Tuple[float, float] = (-3, 3),
        ylim: Tuple[float, float] = (-3, 3),
        base_dist: Optional[torch.distributions.Distribution] = None,
        num_samples: int = 1000
    ):
        """
        Create an interactive 2D transformation plot using Plotly.
        
        Args:
            flow: The normalizing flow to visualize
            grid_size: Number of grid points along each axis
            xlim: X-axis limits
            ylim: Y-axis limits
            base_dist: Base distribution
            num_samples: Number of samples to show
            
        Returns:
            Plotly figure object (if plotly is available)
        """
        if not PLOTLY_AVAILABLE:
            raise ImportError("Plotly is required for interactive visualizations")
            
        if base_dist is None:
            base_dist = torch.distributions.MultivariateNormal(
                torch.zeros(2), torch.eye(2)
            )
            
        flow.eval()
        
        # Create subplots
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Original Space', 'Transformed Space'),
            specs=[[{'type': 'scatter'}, {'type': 'scatter'}]]
        )
        
        # Generate samples
        with torch.no_grad():
            z_samples = base_dist.sample((num_samples,)).to(self.device)
            x_samples, _ = flow.forward(z_samples)
            
        z_samples = z_samples.cpu().numpy()
        x_samples = x_samples.cpu().numpy()
        
        # Add sample points
        fig.add_trace(
            go.Scatter(
                x=z_samples[:, 0], y=z_samples[:, 1],
                mode='markers',
                marker=dict(size=4, opacity=0.6, color='blue'),
                name='Base samples',
                hovertemplate='z₁: %{x:.2f}<br>z₂: %{y:.2f}<extra></extra>'
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=x_samples[:, 0], y=x_samples[:, 1],
                mode='markers',
                marker=dict(size=4, opacity=0.6, color='red'),
                name='Transformed samples',
                hovertemplate='x₁: %{x:.2f}<br>x₂: %{y:.2f}<extra></extra>'
            ),
            row=1, col=2
        )
        
        # Add grid lines
        x_grid = np.linspace(xlim[0], xlim[1], 10)
        y_grid = np.linspace(ylim[0], ylim[1], 10)
        
        # Vertical grid lines
        for x_val in x_grid:
            y_line = np.linspace(ylim[0], ylim[1], grid_size)
            x_line = np.full_like(y_line, x_val)
            
            grid_points = torch.tensor(
                np.stack([x_line, y_line], axis=1),
                dtype=torch.float32,
                device=self.device
            )
            
            with torch.no_grad():
                transformed_points, _ = flow.forward(grid_points)
                
            transformed_points = transformed_points.cpu().numpy()
            
            # Original space
            fig.add_trace(
                go.Scatter(
                    x=x_line, y=y_line,
                    mode='lines',
                    line=dict(color='lightblue', width=1),
                    showlegend=False,
                    hoverinfo='skip'
                ),
                row=1, col=1
            )
            
            # Transformed space
            fig.add_trace(
                go.Scatter(
                    x=transformed_points[:, 0], y=transformed_points[:, 1],
                    mode='lines',
                    line=dict(color='lightcoral', width=1),
                    showlegend=False,
                    hoverinfo='skip'
                ),
                row=1, col=2
            )
        
        # Horizontal grid lines
        for y_val in y_grid:
            x_line = np.linspace(xlim[0], xlim[1], grid_size)
            y_line = np.full_like(x_line, y_val)
            
            grid_points = torch.tensor(
                np.stack([x_line, y_line], axis=1),
                dtype=torch.float32,
                device=self.device
            )
            
            with torch.no_grad():
                transformed_points, _ = flow.forward(grid_points)
                
            transformed_points = transformed_points.cpu().numpy()
            
            # Original space
            fig.add_trace(
                go.Scatter(
                    x=x_line, y=y_line,
                    mode='lines',
                    line=dict(color='lightblue', width=1),
                    showlegend=False,
                    hoverinfo='skip'
                ),
                row=1, col=1
            )
            
            # Transformed space
            fig.add_trace(
                go.Scatter(
                    x=transformed_points[:, 0], y=transformed_points[:, 1],
                    mode='lines',
                    line=dict(color='lightcoral', width=1),
                    showlegend=False,
                    hoverinfo='skip'
                ),
                row=1, col=2
            )
        
        # Update layout
        fig.update_layout(
            title='Interactive 2D Flow Transformation',
            height=600,
            showlegend=True
        )
        
        fig.update_xaxes(title_text='z₁', row=1, col=1)
        fig.update_yaxes(title_text='z₂', row=1, col=1)
        fig.update_xaxes(title_text='x₁', row=1, col=2)
        fig.update_yaxes(title_text='x₂', row=1, col=2)
        
        return fig
    
    def save_animation(
        self,
        animation_obj: animation.FuncAnimation,
        filename: str,
        writer: str = 'pillow',
        fps: int = 2
    ):
        """
        Save animation to file.
        
        Args:
            animation_obj: The animation object to save
            filename: Output filename
            writer: Animation writer ('pillow' for GIF, 'ffmpeg' for MP4)
            fps: Frames per second
        """
        animation_obj.save(filename, writer=writer, fps=fps)
        print(f"Animation saved to {filename}")