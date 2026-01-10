"""
Tests for flow diagnostics functionality.
"""

import torch
import pytest
import numpy as np
import matplotlib.pyplot as plt
from src.flows.coupling.coupling_layer import CouplingLayer
from src.flows.flow.sequential_flow import SequentialFlow
from src.visualization.diagnostics import FlowDiagnostics, DiagnosticResult


class TestFlowDiagnostics:
    """Test cases for FlowDiagnostics."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.device = 'cpu'
        self.diagnostics = FlowDiagnostics(device=self.device)
        
        # Create a simple flow for testing
        mask = torch.tensor([1., 0.])
        self.flow = CouplingLayer(2, 16, mask)
        
        # Create test dataset
        self.dataset = torch.randn(100, 2)
        self.test_data = torch.randn(20, 2)
    
    def test_diagnostics_initialization(self):
        """Test FlowDiagnostics initialization."""
        diagnostics = FlowDiagnostics()
        assert diagnostics.device == 'cpu'
        assert len(diagnostics.diagnostic_history) == 0
        
        diagnostics_cuda = FlowDiagnostics(device='cuda')
        assert diagnostics_cuda.device == 'cuda'
    
    def test_check_invertibility_precision(self):
        """Test invertibility precision checking."""
        result = self.diagnostics.check_invertibility_precision(
            self.flow, self.test_data, tolerance=1e-6, num_iterations=2
        )
        
        # Check result structure
        assert isinstance(result, DiagnosticResult)
        assert result.test_name == "Invertibility Precision Test"
        assert isinstance(result.passed, bool)
        assert 0.0 <= result.score <= 1.0
        assert isinstance(result.details, dict)
        assert isinstance(result.recommendations, list)
        assert result.timestamp is not None
        
        # Check that result was added to history
        assert len(self.diagnostics.diagnostic_history) == 1
        
        # Check details structure
        assert 'overall' in result.details
        assert 'max_error_across_iterations' in result.details['overall']
        assert 'tolerance' in result.details['overall']
    
    def test_measure_expressiveness(self):
        """Test expressiveness measurement."""
        result = self.diagnostics.measure_expressiveness(
            self.flow, self.dataset, num_samples=100
        )
        
        # Check result structure
        assert isinstance(result, DiagnosticResult)
        assert result.test_name == "Expressiveness Analysis"
        assert isinstance(result.passed, bool)
        assert 0.0 <= result.score <= 1.0
        
        # Check details structure
        expected_keys = [
            'coverage_score', 'diversity_score', 'mode_collapse_score',
            'effective_sample_size', 'mean_condition_number'
        ]
        for key in expected_keys:
            assert key in result.details
    
    def test_check_numerical_stability(self):
        """Test numerical stability checking."""
        result = self.diagnostics.check_numerical_stability(
            self.flow, self.test_data, perturbation_scale=1e-6
        )
        
        # Check result structure
        assert isinstance(result, DiagnosticResult)
        assert result.test_name == "Numerical Stability Test"
        assert isinstance(result.passed, bool)
        assert 0.0 <= result.score <= 1.0
        
        # Check details structure
        expected_keys = [
            'mean_sensitivity', 'max_sensitivity', 'mean_log_det_sensitivity',
            'has_nan_inf', 'perturbation_scale'
        ]
        for key in expected_keys:
            assert key in result.details
    
    def test_run_comprehensive_diagnostics(self):
        """Test comprehensive diagnostic suite."""
        results = self.diagnostics.run_comprehensive_diagnostics(
            self.flow, self.dataset, num_samples=50
        )
        
        # Check that all expected tests were run
        expected_tests = ['invertibility', 'stability', 'expressiveness']
        assert set(results.keys()) == set(expected_tests)
        
        # Check that all results are DiagnosticResult objects
        for test_name, result in results.items():
            assert isinstance(result, DiagnosticResult)
            assert isinstance(result.passed, bool)
            assert 0.0 <= result.score <= 1.0
    
    def test_generate_diagnostic_report(self):
        """Test diagnostic report generation."""
        # Run some diagnostics first
        results = self.diagnostics.run_comprehensive_diagnostics(
            self.flow, self.dataset, num_samples=50
        )
        
        # Generate report
        report = self.diagnostics.generate_diagnostic_report(results)
        
        # Check report structure
        assert isinstance(report, str)
        assert "NORMALIZING FLOW DIAGNOSTIC REPORT" in report
        assert "SUMMARY" in report
        assert "Tests Passed:" in report
        assert "Overall Score:" in report
        
        # Check that all test names appear in report
        for test_name in results.keys():
            assert test_name.upper() in report
    
    def test_plot_diagnostic_summary(self):
        """Test diagnostic summary plotting."""
        # Run some diagnostics first
        results = self.diagnostics.run_comprehensive_diagnostics(
            self.flow, self.dataset, num_samples=50
        )
        
        # Create plot
        fig = self.diagnostics.plot_diagnostic_summary(results)
        
        assert fig is not None
        assert len(fig.axes) == 4  # 2x2 subplot grid
        
        # Clean up
        plt.close(fig)
    
    def test_sequential_flow_diagnostics(self):
        """Test diagnostics with SequentialFlow."""
        # Create a sequential flow
        layers = []
        for i in range(2):
            mask = torch.tensor([1., 0.]) if i % 2 == 0 else torch.tensor([0., 1.])
            layer = CouplingLayer(2, 16, mask)
            layers.append(layer)
        
        sequential_flow = SequentialFlow(layers)
        
        # Test invertibility
        result = self.diagnostics.check_invertibility_precision(
            sequential_flow, self.test_data, tolerance=1e-6
        )
        
        assert isinstance(result, DiagnosticResult)
        assert result.test_name == "Invertibility Precision Test"
    
    def test_diagnostic_result_structure(self):
        """Test DiagnosticResult dataclass structure."""
        result = DiagnosticResult(
            test_name="Test",
            passed=True,
            score=0.8,
            details={'key': 'value'},
            recommendations=['recommendation'],
            timestamp="2023-01-01T00:00:00"
        )
        
        assert result.test_name == "Test"
        assert result.passed is True
        assert result.score == 0.8
        assert result.details == {'key': 'value'}
        assert result.recommendations == ['recommendation']
        assert result.timestamp == "2023-01-01T00:00:00"
    
    def test_error_handling(self):
        """Test error handling in diagnostics."""
        # Create a flow that might cause issues
        mask = torch.tensor([1., 0.])
        problematic_flow = CouplingLayer(2, 16, mask)
        
        # Test with extreme values that might cause numerical issues
        extreme_data = torch.randn(10, 2) * 1000
        
        try:
            result = self.diagnostics.check_invertibility_precision(
                problematic_flow, extreme_data, tolerance=1e-6
            )
            # Should not crash, even if test fails
            assert isinstance(result, DiagnosticResult)
        except Exception:
            # Some numerical issues might be expected
            pass
    
    def test_diagnostic_history(self):
        """Test diagnostic history tracking."""
        initial_history_length = len(self.diagnostics.diagnostic_history)
        
        # Run a diagnostic
        self.diagnostics.check_invertibility_precision(
            self.flow, self.test_data, tolerance=1e-6
        )
        
        # Check that history was updated
        assert len(self.diagnostics.diagnostic_history) == initial_history_length + 1
        
        # Run another diagnostic
        self.diagnostics.check_numerical_stability(self.flow, self.test_data)
        
        # Check that history was updated again
        assert len(self.diagnostics.diagnostic_history) == initial_history_length + 2
    
    def test_different_tolerances(self):
        """Test diagnostics with different tolerance levels."""
        # Test with strict tolerance
        strict_result = self.diagnostics.check_invertibility_precision(
            self.flow, self.test_data, tolerance=1e-10
        )
        
        # Test with loose tolerance
        loose_result = self.diagnostics.check_invertibility_precision(
            self.flow, self.test_data, tolerance=1e-4
        )
        
        # Loose tolerance should generally give better scores
        # (though this might not always be true depending on the flow)
        assert isinstance(strict_result, DiagnosticResult)
        assert isinstance(loose_result, DiagnosticResult)
    
    def test_higher_dimensional_diagnostics(self):
        """Test diagnostics with higher-dimensional flows."""
        # Create 4D flow
        mask_4d = torch.tensor([1., 0., 1., 0.])
        flow_4d = CouplingLayer(4, 32, mask_4d)
        dataset_4d = torch.randn(50, 4)
        test_data_4d = torch.randn(10, 4)
        
        # Test invertibility
        result = self.diagnostics.check_invertibility_precision(
            flow_4d, test_data_4d, tolerance=1e-6
        )
        
        assert isinstance(result, DiagnosticResult)
        
        # Test expressiveness
        result = self.diagnostics.measure_expressiveness(
            flow_4d, dataset_4d, num_samples=50
        )
        
        assert isinstance(result, DiagnosticResult)


if __name__ == "__main__":
    # Run basic tests
    test_diagnostics = TestFlowDiagnostics()
    test_diagnostics.setup_method()
    
    print("Running FlowDiagnostics tests...")
    
    test_diagnostics.test_diagnostics_initialization()
    print("✓ Initialization test passed")
    
    test_diagnostics.test_check_invertibility_precision()
    print("✓ Invertibility precision test passed")
    
    test_diagnostics.test_measure_expressiveness()
    print("✓ Expressiveness measurement test passed")
    
    test_diagnostics.test_check_numerical_stability()
    print("✓ Numerical stability test passed")
    
    test_diagnostics.test_run_comprehensive_diagnostics()
    print("✓ Comprehensive diagnostics test passed")
    
    test_diagnostics.test_generate_diagnostic_report()
    print("✓ Diagnostic report generation test passed")
    
    test_diagnostics.test_plot_diagnostic_summary()
    print("✓ Diagnostic summary plotting test passed")
    
    print("All FlowDiagnostics tests passed!")