"""
Comprehensive demonstration script for flow visualization and analysis capabilities.
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from src.flows.coupling.coupling_layer import CouplingLayer
from src.flows.flow.sequential_flow import SequentialFlow
from src.visualization.flow_visualizer import FlowVisualizer
from src.visualization.jacobian_analyzer import JacobianAnalyzer
from src.visualization.diagnostics import FlowDiagnostics


def create_alternating_mask(dim: int, even: bool = True) -> torch.Tensor:
    """Create alternating binary mask for coupling layers."""
    mask = torch.zeros(dim)
    if even:
        mask[::2] = 1
    else:
        mask[1::2] = 1
    return mask


def create_simple_flow(dim: int = 2, num_layers: int = 4, hidden_dim: int = 64) -> SequentialFlow:
    """Create a simple Real NVP flow for demonstration."""
    layers = []
    
    for i in range(num_layers):
        # Alternate masks
        mask = create_alternating_mask(dim, even=(i % 2 == 0))
        layer = CouplingLayer(dim, hidden_dim, mask)
        layers.append(layer)
    
    return SequentialFlow(layers)


def demo_2d_transformation():
    """Demonstrate 2D transformation visualization."""
    print("Creating 2D transformation visualization...")
    
    # Create a simple flow
    flow = create_simple_flow(dim=2, num_layers=4, hidden_dim=32)
    
    # Initialize visualizer
    visualizer = FlowVisualizer(device='cpu')
    
    # Create base distribution
    base_dist = torch.distributions.MultivariateNormal(
        torch.zeros(2), torch.eye(2)
    )
    
    # Plot transformation
    fig = visualizer.plot_2d_transformation(
        flow=flow,
        grid_size=50,
        xlim=(-3, 3),
        ylim=(-3, 3),
        base_dist=base_dist,
        show_grid=True,
        show_samples=True,
        num_samples=500
    )
    
    plt.savefig('examples/2d_transformation_demo.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("2D transformation plot saved to examples/2d_transformation_demo.png")


def demo_density_evolution():
    """Demonstrate density evolution visualization."""
    print("Creating density evolution visualization...")
    
    # Create flow layers
    layers = []
    for i in range(3):
        mask = create_alternating_mask(2, even=(i % 2 == 0))
        layer = CouplingLayer(2, 32, mask)
        layers.append(layer)
    
    # Initialize visualizer
    visualizer = FlowVisualizer(device='cpu')
    
    # Create base distribution
    base_dist = torch.distributions.MultivariateNormal(
        torch.zeros(2), torch.eye(2)
    )
    
    # Plot density evolution
    fig = visualizer.plot_density_evolution(
        flow_layers=layers,
        base_dist=base_dist,
        grid_size=80,
        xlim=(-4, 4),
        ylim=(-4, 4)
    )
    
    plt.savefig('examples/density_evolution_demo.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Density evolution plot saved to examples/density_evolution_demo.png")


def demo_animation():
    """Demonstrate animated density evolution."""
    print("Creating animated density evolution...")
    
    # Create flow layers
    layers = []
    for i in range(4):
        mask = create_alternating_mask(2, even=(i % 2 == 0))
        layer = CouplingLayer(2, 32, mask)
        layers.append(layer)
    
    # Initialize visualizer
    visualizer = FlowVisualizer(device='cpu')
    
    # Create base distribution
    base_dist = torch.distributions.MultivariateNormal(
        torch.zeros(2), torch.eye(2)
    )
    
    # Create animation
    anim = visualizer.animate_density_evolution(
        flow_layers=layers,
        base_dist=base_dist,
        num_samples=1000,
        xlim=(-4, 4),
        ylim=(-4, 4),
        interval=1000
    )
    
    # Save animation
    try:
        visualizer.save_animation(anim, 'examples/density_evolution_animation.gif')
        print("Animation saved to examples/density_evolution_animation.gif")
    except Exception as e:
        print(f"Could not save animation: {e}")
        print("Showing animation instead...")
        plt.show()


def demo_interactive_visualization():
    """Demonstrate interactive visualization (requires plotly)."""
    print("Creating interactive visualization...")
    
    try:
        # Create a simple flow
        flow = create_simple_flow(dim=2, num_layers=3, hidden_dim=32)
        
        # Initialize visualizer
        visualizer = FlowVisualizer(device='cpu')
        
        # Create base distribution
        base_dist = torch.distributions.MultivariateNormal(
            torch.zeros(2), torch.eye(2)
        )
        
        # Create interactive plot
        fig = visualizer.plot_interactive_2d_transformation(
            flow=flow,
            grid_size=30,
            xlim=(-3, 3),
            ylim=(-3, 3),
            base_dist=base_dist,
            num_samples=500
        )
        
        # Save as HTML
        fig.write_html('examples/interactive_transformation.html')
        print("Interactive plot saved to examples/interactive_transformation.html")
        
        # Show in browser (if available)
        fig.show()
        
    except ImportError:
        print("Plotly not available. Skipping interactive visualization demo.")
    except Exception as e:
        print(f"Error creating interactive visualization: {e}")


if __name__ == "__main__":
    # Create examples directory if it doesn't exist
    import os
    os.makedirs('examples', exist_ok=True)
    
    print("Flow Visualization Demo")
    print("=" * 50)
    
    # Run demonstrations
    demo_2d_transformation()
    print()
    
    demo_density_evolution()
    print()
    
    demo_animation()
    print()
    
    demo_interactive_visualization()
    print()
    
    print("All visualization demos completed!")

def dem
o_jacobian_analysis():
    """Demonstrate Jacobian analysis capabilities."""
    print("Creating Jacobian analysis demonstration...")
    
    # Create a flow
    flow = create_simple_flow(dim=2, num_layers=3, hidden_dim=32)
    
    # Initialize analyzer
    analyzer = JacobianAnalyzer(device='cpu')
    
    # Create test data
    x = torch.randn(50, 2)
    
    # Plot eigenvalue spectrum
    fig = analyzer.plot_eigenvalue_spectrum(flow, x, show_unit_circle=True)
    plt.savefig('examples/eigenvalue_spectrum_demo.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Eigenvalue spectrum plot saved to examples/eigenvalue_spectrum_demo.png")
    
    # Create flow layers for condition number evolution
    layers = []
    for i in range(4):
        mask = create_alternating_mask(2, even=(i % 2 == 0))
        layer = CouplingLayer(2, 32, mask)
        layers.append(layer)
    
    # Plot condition number evolution
    fig = analyzer.plot_condition_number_evolution(layers, x)
    plt.savefig('examples/condition_number_evolution_demo.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Condition number evolution plot saved to examples/condition_number_evolution_demo.png")
    
    # Plot gradient flow analysis
    fig = analyzer.plot_gradient_flow_analysis(flow, x)
    plt.savefig('examples/gradient_flow_analysis_demo.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Gradient flow analysis plot saved to examples/gradient_flow_analysis_demo.png")


def demo_comprehensive_diagnostics():
    """Demonstrate comprehensive diagnostic capabilities."""
    print("Running comprehensive flow diagnostics...")
    
    # Create a flow
    flow = create_simple_flow(dim=2, num_layers=4, hidden_dim=32)
    
    # Create synthetic dataset
    dataset = torch.randn(200, 2)
    
    # Initialize diagnostics
    diagnostics = FlowDiagnostics(device='cpu')
    
    # Run comprehensive diagnostics
    results = diagnostics.run_comprehensive_diagnostics(
        flow=flow,
        dataset=dataset,
        invertibility_tolerance=1e-6,
        num_samples=100
    )
    
    # Generate and print report
    report = diagnostics.generate_diagnostic_report(
        results, save_path='examples/diagnostic_report.txt'
    )
    print("\nDiagnostic Report:")
    print("=" * 50)
    print(report)
    
    # Create diagnostic summary plot
    fig = diagnostics.plot_diagnostic_summary(results)
    plt.savefig('examples/diagnostic_summary_demo.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Diagnostic summary plot saved to examples/diagnostic_summary_demo.png")
    
    return results


def demo_individual_diagnostics():
    """Demonstrate individual diagnostic tests."""
    print("Running individual diagnostic tests...")
    
    # Create a flow
    flow = create_simple_flow(dim=2, num_layers=3, hidden_dim=32)
    
    # Create test data
    test_data = torch.randn(30, 2)
    dataset = torch.randn(100, 2)
    
    # Initialize diagnostics
    diagnostics = FlowDiagnostics(device='cpu')
    
    # Test invertibility
    print("\n1. Testing invertibility precision...")
    invertibility_result = diagnostics.check_invertibility_precision(
        flow, test_data, tolerance=1e-7, num_iterations=3
    )
    print(f"   Status: {'PASS' if invertibility_result.passed else 'FAIL'}")
    print(f"   Score: {invertibility_result.score:.3f}")
    print(f"   Max error: {invertibility_result.details['overall']['max_error_across_iterations']:.2e}")
    
    # Test numerical stability
    print("\n2. Testing numerical stability...")
    stability_result = diagnostics.check_numerical_stability(
        flow, test_data, perturbation_scale=1e-6
    )
    print(f"   Status: {'PASS' if stability_result.passed else 'FAIL'}")
    print(f"   Score: {stability_result.score:.3f}")
    print(f"   Mean sensitivity: {stability_result.details['mean_sensitivity']:.2e}")
    
    # Test expressiveness
    print("\n3. Testing expressiveness...")
    expressiveness_result = diagnostics.measure_expressiveness(
        flow, dataset, num_samples=100
    )
    print(f"   Status: {'PASS' if expressiveness_result.passed else 'FAIL'}")
    print(f"   Score: {expressiveness_result.score:.3f}")
    print(f"   Coverage: {expressiveness_result.details['coverage_score']:.3f}")
    print(f"   Diversity: {expressiveness_result.details['diversity_score']:.3f}")


def demo_advanced_analysis():
    """Demonstrate advanced analysis capabilities."""
    print("Creating advanced analysis demonstration...")
    
    # Create different types of flows for comparison
    flows = {
        'Simple Flow': create_simple_flow(dim=2, num_layers=2, hidden_dim=16),
        'Deep Flow': create_simple_flow(dim=2, num_layers=6, hidden_dim=32),
        'Wide Flow': create_simple_flow(dim=2, num_layers=3, hidden_dim=64)
    }
    
    # Initialize analyzers
    analyzer = JacobianAnalyzer(device='cpu')
    diagnostics = FlowDiagnostics(device='cpu')
    
    # Test data
    x = torch.randn(30, 2)
    dataset = torch.randn(100, 2)
    
    print("\nComparing different flow architectures:")
    print("-" * 60)
    
    for flow_name, flow in flows.items():
        print(f"\n{flow_name}:")
        
        # Compute condition numbers
        condition_numbers = analyzer.compute_condition_numbers(flow, x)
        mean_condition = condition_numbers.mean().item()
        
        # Test invertibility
        invertibility_result = diagnostics.check_invertibility_precision(
            flow, x, tolerance=1e-6
        )
        
        # Test expressiveness
        expressiveness_result = diagnostics.measure_expressiveness(
            flow, dataset, num_samples=50
        )
        
        print(f"  Mean Condition Number: {mean_condition:.2f}")
        print(f"  Invertibility Score: {invertibility_result.score:.3f}")
        print(f"  Expressiveness Score: {expressiveness_result.score:.3f}")
        print(f"  Overall Health: {(invertibility_result.score + expressiveness_result.score) / 2:.3f}")


if __name__ == "__main__":
    # Create examples directory if it doesn't exist
    import os
    os.makedirs('examples', exist_ok=True)
    
    print("Comprehensive Flow Analysis Demo")
    print("=" * 60)
    
    # Run all demonstrations
    demo_2d_transformation()
    print()
    
    demo_density_evolution()
    print()
    
    demo_animation()
    print()
    
    demo_interactive_visualization()
    print()
    
    demo_jacobian_analysis()
    print()
    
    demo_individual_diagnostics()
    print()
    
    demo_comprehensive_diagnostics()
    print()
    
    demo_advanced_analysis()
    print()
    
    print("All visualization and analysis demos completed!")
    print("\nGenerated files:")
    print("- examples/2d_transformation_demo.png")
    print("- examples/density_evolution_demo.png")
    print("- examples/density_evolution_animation.gif")
    print("- examples/interactive_transformation.html")
    print("- examples/eigenvalue_spectrum_demo.png")
    print("- examples/condition_number_evolution_demo.png")
    print("- examples/gradient_flow_analysis_demo.png")
    print("- examples/diagnostic_summary_demo.png")
    print("- examples/diagnostic_report.txt")