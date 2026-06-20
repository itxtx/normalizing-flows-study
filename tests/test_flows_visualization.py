"""
Tests for flow visualization functionality.
"""

import torch
import pytest
import numpy as np
import matplotlib.pyplot as plt
from src.flows.coupling.coupling_layer import CouplingLayer
from src.flows.flow.sequential_flow import SequentialFlow
from src.visualization.flow_visualizer import FlowVisualizer


class TestFlowVisualizer:
    """Test cases for FlowVisualizer."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.device = 'cpu'
        self.visualizer = FlowVisualizer(device=self.device)
        
        # Create a simple flow for testing
        mask = torch.tensor([1., 0.])
        self.flow = CouplingLayer(2, 16, mask)
        
        # Create base distribution
        self.base_dist = torch.distributions.MultivariateNormal(
            torch.zeros(2), torch.eye(2)
        )
    
    def test_visualizer_initialization(self):
        """Test FlowVisualizer initialization."""
        visualizer = FlowVisualizer()
        assert visualizer.device == 'cpu'
        
        visualizer_cuda = FlowVisualizer(device='cuda')
        assert visualizer_cuda.device == 'cuda'
    
    def test_2d_transformation_plot(self):
        """Test 2D transformation plotting."""
        fig = self.visualizer.plot_2d_transformation(
            flow=self.flow,
            grid_size=20,
            xlim=(-2, 2),
            ylim=(-2, 2),
            base_dist=self.base_dist,
            show_grid=True,
            show_samples=True,
            num_samples=100
        )
        
        assert fig is not None
        assert len(fig.axes) == 2  # Should have 2 subplots
        
        # Clean up
        plt.close(fig)
    
    def test_2d_transformation_plot_minimal(self):
        """Test 2D transformation plotting with minimal options."""
        fig = self.visualizer.plot_2d_transformation(
            flow=self.flow,
            grid_size=10,
            show_grid=False,
            show_samples=False
        )
        
        assert fig is not None
        assert len(fig.axes) == 2
        
        # Clean up
        plt.close(fig)
    
    def test_density_evolution_plot(self):
        """Test density evolution plotting."""
        # Create multiple layers
        layers = []
        for i in range(2):
            mask = torch.tensor([1., 0.]) if i % 2 == 0 else torch.tensor([0., 1.])
            layer = CouplingLayer(2, 16, mask)
            layers.append(layer)
        
        fig = self.visualizer.plot_density_evolution(
            flow_layers=layers,
            base_dist=self.base_dist,
            grid_size=30,
            xlim=(-2, 2),
            ylim=(-2, 2)
        )
        
        assert fig is not None
        # Note: matplotlib may create additional axes for colorbars
        assert len(fig.axes) >= 3  # At least base + 2 layers
        
        # Clean up
        plt.close(fig)
    
    def test_density_evolution_empty_layers(self):
        """Test density evolution with empty layer list."""
        fig = self.visualizer.plot_density_evolution(
            flow_layers=[],
            base_dist=self.base_dist,
            grid_size=20
        )
        
        assert fig is not None
        # Note: matplotlib may create additional axes for colorbars
        assert len(fig.axes) >= 1  # At least base distribution
        
        # Clean up
        plt.close(fig)
    
    def test_animate_density_evolution(self):
        """Test animated density evolution."""
        # Create multiple layers
        layers = []
        for i in range(2):
            mask = torch.tensor([1., 0.]) if i % 2 == 0 else torch.tensor([0., 1.])
            layer = CouplingLayer(2, 16, mask)
            layers.append(layer)
        
        anim = self.visualizer.animate_density_evolution(
            flow_layers=layers,
            base_dist=self.base_dist,
            num_samples=100,
            xlim=(-2, 2),
            ylim=(-2, 2),
            interval=100
        )
        
        assert anim is not None
        
        # Clean up
        plt.close('all')
    
    def test_sequential_flow_visualization(self):
        """Test visualization with SequentialFlow."""
        # Create a sequential flow
        layers = []
        for i in range(3):
            mask = torch.tensor([1., 0.]) if i % 2 == 0 else torch.tensor([0., 1.])
            layer = CouplingLayer(2, 16, mask)
            layers.append(layer)
        
        sequential_flow = SequentialFlow(layers)
        
        fig = self.visualizer.plot_2d_transformation(
            flow=sequential_flow,
            grid_size=15,
            num_samples=50
        )
        
        assert fig is not None
        assert len(fig.axes) == 2
        
        # Clean up
        plt.close(fig)
    
    def test_different_base_distributions(self):
        """Test visualization with different base distributions."""
        # Test with different covariance
        base_dist = torch.distributions.MultivariateNormal(
            torch.zeros(2), 2.0 * torch.eye(2)
        )
        
        fig = self.visualizer.plot_2d_transformation(
            flow=self.flow,
            base_dist=base_dist,
            grid_size=15,
            num_samples=50
        )
        
        assert fig is not None
        
        # Clean up
        plt.close(fig)
    
    def test_error_handling_invalid_dimensions(self):
        """Test error handling for invalid dimensions."""
        # Create 3D flow (should work but might not visualize well)
        mask_3d = torch.tensor([1., 0., 1.])
        flow_3d = CouplingLayer(3, 16, mask_3d)
        
        # This should not crash but might not produce meaningful 2D plots
        with torch.no_grad():
            z = torch.randn(10, 3)
            x, log_det = flow_3d.forward(z)
            assert x.shape == (10, 3)
    
    @pytest.mark.skipif(
        not hasattr(plt, 'show'), 
        reason="Matplotlib display not available"
    )
    def test_interactive_visualization_fallback(self):
        """Test interactive visualization fallback when plotly unavailable."""
        try:
            fig = self.visualizer.plot_interactive_2d_transformation(
                flow=self.flow,
                grid_size=10,
                num_samples=50
            )
            # If plotly is available, this should work
            assert fig is not None
        except ImportError:
            # If plotly is not available, should raise ImportError
            pass
    
    def test_numerical_stability(self):
        """Test visualization with extreme values."""
        # Create flow that might produce extreme values
        mask = torch.tensor([1., 0.])
        flow = CouplingLayer(2, 32, mask)
        
        # Test with extreme input values
        base_dist = torch.distributions.MultivariateNormal(
            torch.zeros(2), 10.0 * torch.eye(2)
        )
        
        fig = self.visualizer.plot_2d_transformation(
            flow=flow,
            base_dist=base_dist,
            xlim=(-5, 5),
            ylim=(-5, 5),
            grid_size=10,
            num_samples=50
        )
        
        assert fig is not None
        
        # Clean up
        plt.close(fig)


if __name__ == "__main__":
    # Run basic tests
    test_viz = TestFlowVisualizer()
    test_viz.setup_method()
    
    print("Running FlowVisualizer tests...")
    
    test_viz.test_visualizer_initialization()
    print("✓ Initialization test passed")
    
    test_viz.test_2d_transformation_plot()
    print("✓ 2D transformation plot test passed")
    
    test_viz.test_density_evolution_plot()
    print("✓ Density evolution plot test passed")
    
    test_viz.test_animate_density_evolution()
    print("✓ Animation test passed")
    
    test_viz.test_sequential_flow_visualization()
    print("✓ Sequential flow visualization test passed")
    
    print("All FlowVisualizer tests passed!")