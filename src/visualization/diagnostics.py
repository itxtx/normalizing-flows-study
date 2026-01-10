"""
Comprehensive diagnostic framework for normalizing flows.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Any
import warnings
from dataclasses import dataclass
from datetime import datetime

from src.flows.flow.flow import Flow
from .jacobian_analyzer import JacobianAnalyzer


@dataclass
class DiagnosticResult:
    """Container for diagnostic test results."""
    test_name: str
    passed: bool
    score: float
    details: Dict[str, Any]
    recommendations: List[str]
    timestamp: str


class FlowDiagnostics:
    """
    Comprehensive diagnostic framework for normalizing flows.
    
    Provides methods for:
    - High-precision invertibility testing
    - Expressiveness metrics and capacity analysis
    - Automated diagnostic reports with actionable recommendations
    """
    
    def __init__(self, device: str = 'cpu'):
        """
        Initialize the flow diagnostics framework.
        
        Args:
            device: Device to run computations on ('cpu' or 'cuda')
        """
        self.device = device
        self.jacobian_analyzer = JacobianAnalyzer(device=device)
        self.diagnostic_history = []
    
    def check_invertibility_precision(
        self,
        flow: Flow,
        x: torch.Tensor,
        tolerance: float = 1e-8,
        num_iterations: int = 3
    ) -> DiagnosticResult:
        """
        High-precision invertibility testing with configurable tolerances.
        
        Args:
            flow: The normalizing flow to test
            x: Input points for testing (batch_size, dim)
            tolerance: Tolerance for invertibility check
            num_iterations: Number of forward-inverse iterations to test
            
        Returns:
            DiagnosticResult with invertibility test results
        """
        flow.eval()
        recommendations = []
        details = {}
        
        try:
            current_x = x.clone()
            max_error = 0.0
            errors_per_iteration = []
            
            for i in range(num_iterations):
                # Forward pass
                y, log_det_forward = flow.forward(current_x)
                
                # Inverse pass
                x_reconstructed, log_det_inverse = flow.inverse(y)
                
                # Compute reconstruction error
                reconstruction_error = torch.norm(current_x - x_reconstructed, dim=1)
                max_iter_error = reconstruction_error.max().item()
                mean_iter_error = reconstruction_error.mean().item()
                
                errors_per_iteration.append({
                    'iteration': i + 1,
                    'max_error': max_iter_error,
                    'mean_error': mean_iter_error,
                    'std_error': reconstruction_error.std().item()
                })
                
                max_error = max(max_error, max_iter_error)
                
                # Check log-determinant consistency
                log_det_error = torch.abs(log_det_forward + log_det_inverse)
                max_log_det_error = log_det_error.max().item()
                
                details[f'iteration_{i+1}'] = {
                    'reconstruction_error': {
                        'max': max_iter_error,
                        'mean': mean_iter_error,
                        'std': reconstruction_error.std().item()
                    },
                    'log_det_consistency_error': {
                        'max': max_log_det_error,
                        'mean': log_det_error.mean().item()
                    }
                }
                
                # Use reconstructed x for next iteration
                current_x = x_reconstructed.detach()
            
            # Overall assessment
            passed = bool(max_error < tolerance)
            score = max(0.0, 1.0 - max_error / tolerance)
            
            details['overall'] = {
                'max_error_across_iterations': max_error,
                'tolerance': tolerance,
                'num_iterations': num_iterations,
                'errors_per_iteration': errors_per_iteration
            }
            
            # Generate recommendations
            if not passed:
                if max_error > 10 * tolerance:
                    recommendations.append("Severe invertibility issues detected. Consider:")
                    recommendations.append("- Reducing learning rate")
                    recommendations.append("- Adding gradient clipping")
                    recommendations.append("- Using spectral normalization")
                elif max_error > tolerance:
                    recommendations.append("Moderate invertibility issues detected. Consider:")
                    recommendations.append("- Fine-tuning numerical precision")
                    recommendations.append("- Checking for vanishing/exploding gradients")
            else:
                recommendations.append("Invertibility test passed successfully")
            
        except Exception as e:
            passed = False
            score = 0.0
            details['error'] = str(e)
            recommendations.append(f"Invertibility test failed with error: {e}")
            recommendations.append("Check flow implementation for numerical stability issues")
        
        result = DiagnosticResult(
            test_name="Invertibility Precision Test",
            passed=passed,
            score=score,
            details=details,
            recommendations=recommendations,
            timestamp=datetime.now().isoformat()
        )
        
        self.diagnostic_history.append(result)
        return result
    
    def measure_expressiveness(
        self,
        flow: Flow,
        dataset: torch.Tensor,
        base_dist: Optional[torch.distributions.Distribution] = None,
        num_samples: int = 1000
    ) -> DiagnosticResult:
        """
        Measure flow capacity and expressiveness metrics.
        
        Args:
            flow: The normalizing flow to analyze
            dataset: Dataset to analyze expressiveness on
            base_dist: Base distribution (defaults to standard normal)
            num_samples: Number of samples for analysis
            
        Returns:
            DiagnosticResult with expressiveness metrics
        """
        if base_dist is None:
            base_dist = torch.distributions.MultivariateNormal(
                torch.zeros(dataset.shape[1], device=self.device),
                torch.eye(dataset.shape[1], device=self.device)
            )
        
        flow.eval()
        recommendations = []
        details = {}
        
        try:
            # Sample from flow
            with torch.no_grad():
                flow_samples = flow.sample(num_samples, base_dist, device=self.device)
            
            # Compute various expressiveness metrics
            
            # 1. Coverage: How well does the flow cover the data distribution?
            coverage_score = self._compute_coverage_score(flow_samples, dataset)
            
            # 2. Diversity: How diverse are the flow samples?
            diversity_score = self._compute_diversity_score(flow_samples)
            
            # 3. Mode collapse detection
            mode_collapse_score = self._detect_mode_collapse(flow_samples, dataset)
            
            # 4. Effective sample size
            effective_sample_size = self._compute_effective_sample_size(flow_samples)
            
            # 5. Jacobian condition number analysis
            subset_indices = torch.randperm(len(dataset))[:min(100, len(dataset))]
            subset_data = dataset[subset_indices]
            condition_numbers = self.jacobian_analyzer.compute_condition_numbers(flow, subset_data)
            mean_condition_number = condition_numbers.mean().item()
            
            details = {
                'coverage_score': coverage_score,
                'diversity_score': diversity_score,
                'mode_collapse_score': mode_collapse_score,
                'effective_sample_size': effective_sample_size,
                'mean_condition_number': mean_condition_number,
                'condition_number_stats': {
                    'mean': mean_condition_number,
                    'std': condition_numbers.std().item(),
                    'max': condition_numbers.max().item(),
                    'min': condition_numbers.min().item()
                }
            }
            
            # Overall expressiveness score (weighted combination)
            overall_score = (
                0.3 * coverage_score +
                0.3 * diversity_score +
                0.2 * mode_collapse_score +
                0.2 * min(1.0, effective_sample_size / num_samples)
            )
            
            passed = bool(overall_score > 0.7)  # Threshold for good expressiveness
            
            # Generate recommendations
            if coverage_score < 0.5:
                recommendations.append("Low coverage detected. Consider:")
                recommendations.append("- Increasing model capacity")
                recommendations.append("- Training for more epochs")
                recommendations.append("- Using different architecture")
            
            if diversity_score < 0.5:
                recommendations.append("Low diversity detected. Consider:")
                recommendations.append("- Checking for mode collapse")
                recommendations.append("- Using different regularization")
                recommendations.append("- Adjusting learning rate")
            
            if mode_collapse_score < 0.5:
                recommendations.append("Mode collapse detected. Consider:")
                recommendations.append("- Using different loss function")
                recommendations.append("- Adding noise to training")
                recommendations.append("- Increasing model capacity")
            
            if mean_condition_number > 1000:
                recommendations.append("High condition numbers detected. Consider:")
                recommendations.append("- Using spectral normalization")
                recommendations.append("- Adding regularization")
                recommendations.append("- Reducing learning rate")
            
            if not recommendations:
                recommendations.append("Flow shows good expressiveness across all metrics")
            
        except Exception as e:
            passed = False
            overall_score = 0.0
            details['error'] = str(e)
            recommendations.append(f"Expressiveness analysis failed with error: {e}")
        
        result = DiagnosticResult(
            test_name="Expressiveness Analysis",
            passed=passed,
            score=overall_score,
            details=details,
            recommendations=recommendations,
            timestamp=datetime.now().isoformat()
        )
        
        self.diagnostic_history.append(result)
        return result
    
    def check_numerical_stability(
        self,
        flow: Flow,
        x: torch.Tensor,
        perturbation_scale: float = 1e-6
    ) -> DiagnosticResult:
        """
        Check numerical stability of the flow.
        
        Args:
            flow: The normalizing flow to test
            x: Input points for testing
            perturbation_scale: Scale of perturbations for stability testing
            
        Returns:
            DiagnosticResult with stability test results
        """
        flow.eval()
        recommendations = []
        details = {}
        
        try:
            # Test with small perturbations
            perturbation = torch.randn_like(x) * perturbation_scale
            x_perturbed = x + perturbation
            
            # Forward pass on original and perturbed inputs
            y, log_det = flow.forward(x)
            y_perturbed, log_det_perturbed = flow.forward(x_perturbed)
            
            # Compute sensitivity
            output_change = torch.norm(y - y_perturbed, dim=1)
            input_change = torch.norm(perturbation, dim=1)
            sensitivity = output_change / (input_change + 1e-12)
            
            log_det_sensitivity = torch.abs(log_det - log_det_perturbed) / (input_change + 1e-12)
            
            # Check for NaN/Inf values
            has_nan_inf = (
                torch.any(torch.isnan(y)) or torch.any(torch.isinf(y)) or
                torch.any(torch.isnan(log_det)) or torch.any(torch.isinf(log_det))
            )
            
            # Compute stability metrics
            mean_sensitivity = sensitivity.mean().item()
            max_sensitivity = sensitivity.max().item()
            mean_log_det_sensitivity = log_det_sensitivity.mean().item()
            
            details = {
                'mean_sensitivity': mean_sensitivity,
                'max_sensitivity': max_sensitivity,
                'mean_log_det_sensitivity': mean_log_det_sensitivity,
                'has_nan_inf': has_nan_inf,
                'perturbation_scale': perturbation_scale,
                'sensitivity_stats': {
                    'mean': mean_sensitivity,
                    'std': sensitivity.std().item(),
                    'max': max_sensitivity,
                    'min': sensitivity.min().item()
                }
            }
            
            # Assess stability
            stability_issues = []
            if has_nan_inf:
                stability_issues.append("NaN/Inf values detected")
            if max_sensitivity > 1e6:
                stability_issues.append("Extremely high sensitivity detected")
            if mean_log_det_sensitivity > 1e3:
                stability_issues.append("High log-determinant sensitivity")
            
            passed = bool(len(stability_issues) == 0)
            score = max(0.0, 1.0 - len(stability_issues) * 0.3)
            
            # Generate recommendations
            if stability_issues:
                recommendations.append("Numerical stability issues detected:")
                recommendations.extend([f"- {issue}" for issue in stability_issues])
                recommendations.append("Consider:")
                recommendations.append("- Using gradient clipping")
                recommendations.append("- Reducing learning rate")
                recommendations.append("- Adding batch normalization")
                recommendations.append("- Using spectral normalization")
            else:
                recommendations.append("Flow shows good numerical stability")
            
        except Exception as e:
            passed = False
            score = 0.0
            details['error'] = str(e)
            recommendations.append(f"Stability test failed with error: {e}")
        
        result = DiagnosticResult(
            test_name="Numerical Stability Test",
            passed=passed,
            score=score,
            details=details,
            recommendations=recommendations,
            timestamp=datetime.now().isoformat()
        )
        
        self.diagnostic_history.append(result)
        return result
    
    def run_comprehensive_diagnostics(
        self,
        flow: Flow,
        dataset: torch.Tensor,
        base_dist: Optional[torch.distributions.Distribution] = None,
        invertibility_tolerance: float = 1e-8,
        num_samples: int = 1000
    ) -> Dict[str, DiagnosticResult]:
        """
        Run comprehensive diagnostic suite on a flow.
        
        Args:
            flow: The normalizing flow to diagnose
            dataset: Dataset for analysis
            base_dist: Base distribution
            invertibility_tolerance: Tolerance for invertibility tests
            num_samples: Number of samples for expressiveness analysis
            
        Returns:
            Dictionary of diagnostic results
        """
        print("Running comprehensive flow diagnostics...")
        
        # Select subset of data for testing
        test_indices = torch.randperm(len(dataset))[:min(100, len(dataset))]
        test_data = dataset[test_indices].to(self.device)
        
        results = {}
        
        # 1. Invertibility test
        print("- Running invertibility test...")
        results['invertibility'] = self.check_invertibility_precision(
            flow, test_data, tolerance=invertibility_tolerance
        )
        
        # 2. Numerical stability test
        print("- Running numerical stability test...")
        results['stability'] = self.check_numerical_stability(flow, test_data)
        
        # 3. Expressiveness analysis
        print("- Running expressiveness analysis...")
        results['expressiveness'] = self.measure_expressiveness(
            flow, dataset, base_dist, num_samples
        )
        
        print("Comprehensive diagnostics completed!")
        return results
    
    def generate_diagnostic_report(
        self,
        results: Dict[str, DiagnosticResult],
        save_path: Optional[str] = None
    ) -> str:
        """
        Generate a comprehensive diagnostic report.
        
        Args:
            results: Dictionary of diagnostic results
            save_path: Optional path to save the report
            
        Returns:
            Report as string
        """
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("NORMALIZING FLOW DIAGNOSTIC REPORT")
        report_lines.append("=" * 80)
        report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("")
        
        # Summary
        total_tests = len(results)
        passed_tests = sum(1 for r in results.values() if r.passed)
        overall_score = np.mean([r.score for r in results.values()])
        
        report_lines.append("SUMMARY")
        report_lines.append("-" * 40)
        report_lines.append(f"Tests Passed: {passed_tests}/{total_tests}")
        report_lines.append(f"Overall Score: {overall_score:.3f}")
        report_lines.append(f"Overall Status: {'PASS' if passed_tests == total_tests else 'FAIL'}")
        report_lines.append("")
        
        # Detailed results
        for test_name, result in results.items():
            report_lines.append(f"{test_name.upper()} TEST")
            report_lines.append("-" * 40)
            report_lines.append(f"Status: {'PASS' if result.passed else 'FAIL'}")
            report_lines.append(f"Score: {result.score:.3f}")
            report_lines.append("")
            
            report_lines.append("Recommendations:")
            for rec in result.recommendations:
                report_lines.append(f"  • {rec}")
            report_lines.append("")
            
            if 'overall' in result.details:
                report_lines.append("Key Metrics:")
                overall_details = result.details['overall']
                for key, value in overall_details.items():
                    if isinstance(value, (int, float)):
                        report_lines.append(f"  • {key}: {value}")
                report_lines.append("")
        
        # Action items
        all_recommendations = []
        for result in results.values():
            if not result.passed:
                all_recommendations.extend(result.recommendations)
        
        if all_recommendations:
            report_lines.append("PRIORITY ACTION ITEMS")
            report_lines.append("-" * 40)
            unique_recommendations = list(set(all_recommendations))
            for i, rec in enumerate(unique_recommendations[:10], 1):  # Top 10
                if not rec.startswith("Consider:") and not rec.endswith("detected:"):
                    report_lines.append(f"{i}. {rec}")
            report_lines.append("")
        
        report_lines.append("=" * 80)
        
        report = "\n".join(report_lines)
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report)
            print(f"Diagnostic report saved to {save_path}")
        
        return report
    
    def plot_diagnostic_summary(
        self,
        results: Dict[str, DiagnosticResult],
        figsize: Tuple[int, int] = (12, 8)
    ) -> plt.Figure:
        """
        Create a visual summary of diagnostic results.
        
        Args:
            results: Dictionary of diagnostic results
            figsize: Figure size
            
        Returns:
            matplotlib Figure object
        """
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        # Test scores
        test_names = list(results.keys())
        scores = [results[name].score for name in test_names]
        colors = ['green' if results[name].passed else 'red' for name in test_names]
        
        axes[0, 0].bar(test_names, scores, color=colors, alpha=0.7)
        axes[0, 0].set_ylabel('Score')
        axes[0, 0].set_title('Diagnostic Test Scores')
        axes[0, 0].set_ylim(0, 1)
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Pass/fail pie chart
        passed = sum(1 for r in results.values() if r.passed)
        failed = len(results) - passed
        
        axes[0, 1].pie([passed, failed], labels=['Passed', 'Failed'], 
                      colors=['green', 'red'], autopct='%1.1f%%')
        axes[0, 1].set_title('Test Results Overview')
        
        # Score distribution
        axes[1, 0].hist(scores, bins=10, alpha=0.7, edgecolor='black')
        axes[1, 0].set_xlabel('Score')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('Score Distribution')
        
        # Timeline of diagnostic history
        if self.diagnostic_history:
            timestamps = [datetime.fromisoformat(r.timestamp) for r in self.diagnostic_history]
            scores_history = [r.score for r in self.diagnostic_history]
            
            axes[1, 1].plot(timestamps, scores_history, 'o-', alpha=0.7)
            axes[1, 1].set_xlabel('Time')
            axes[1, 1].set_ylabel('Score')
            axes[1, 1].set_title('Diagnostic History')
            axes[1, 1].tick_params(axis='x', rotation=45)
        else:
            axes[1, 1].text(0.5, 0.5, 'No diagnostic history', 
                           ha='center', va='center', transform=axes[1, 1].transAxes)
            axes[1, 1].set_title('Diagnostic History')
        
        plt.tight_layout()
        return fig
    
    def _compute_coverage_score(self, samples: torch.Tensor, dataset: torch.Tensor) -> float:
        """Compute how well samples cover the data distribution."""
        # Simple coverage metric based on nearest neighbor distances
        try:
            from sklearn.neighbors import NearestNeighbors
            
            # Fit on dataset
            nbrs = NearestNeighbors(n_neighbors=1).fit(dataset.cpu().numpy())
            
            # Find distances from samples to nearest data points
            distances, _ = nbrs.kneighbors(samples.cpu().numpy())
            
            # Coverage score based on how close samples are to data
            mean_distance = np.mean(distances)
            coverage_score = max(0.0, 1.0 - mean_distance)
            
            return coverage_score
            
        except ImportError:
            # Fallback if sklearn not available
            return 0.5  # Neutral score
    
    def _compute_diversity_score(self, samples: torch.Tensor) -> float:
        """Compute diversity of samples."""
        # Compute pairwise distances
        n_samples = min(1000, len(samples))  # Limit for computational efficiency
        subset = samples[:n_samples]
        
        # Compute mean pairwise distance
        distances = torch.cdist(subset, subset)
        mean_distance = distances.mean().item()
        
        # Normalize by dimension (rough heuristic)
        diversity_score = min(1.0, mean_distance / np.sqrt(samples.shape[1]))
        
        return diversity_score
    
    def _detect_mode_collapse(self, samples: torch.Tensor, dataset: torch.Tensor) -> float:
        """Detect mode collapse in samples."""
        # Simple mode collapse detection based on sample variance
        sample_var = torch.var(samples, dim=0).mean().item()
        data_var = torch.var(dataset, dim=0).mean().item()
        
        # Score based on variance ratio
        variance_ratio = sample_var / (data_var + 1e-12)
        mode_collapse_score = min(1.0, variance_ratio)
        
        return mode_collapse_score
    
    def _compute_effective_sample_size(self, samples: torch.Tensor) -> float:
        """Compute effective sample size."""
        # Simple effective sample size based on unique samples
        # This is a rough approximation
        n_samples = len(samples)
        
        # Count approximately unique samples (within tolerance)
        unique_samples = torch.unique(samples.round(decimals=4), dim=0)
        effective_size = len(unique_samples)
        
        return float(effective_size)