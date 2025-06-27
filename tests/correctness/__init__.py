"""
Correctness Test Suite for Normalizing Flows

This package contains comprehensive algorithmic correctness tests for all flow implementations.
Tests are designed to catch critical bugs through rigorous mathematical verification.

Test Modules:
- test_invertibility.py: Tests forward/inverse consistency and log-determinant properties
- test_logdet_autodiff.py: Validates log-determinant computation against autodiff
- test_gradcheck.py: Verifies gradient correctness using torch.autograd.gradcheck
- test_distribution_preservation.py: Tests training behavior and distribution modeling

All test failures include the **critical-bug** tag for automatic indexing.
"""

__version__ = "1.0.0"
__author__ = "Normalizing Flows Study"

# Test suite metadata
TEST_MODULES = [
    "test_invertibility",
    "test_logdet_autodiff", 
    "test_gradcheck",
    "test_distribution_preservation"
]

FLOW_CLASSES_TESTED = [
    "CouplingLayer",
    "SplineCouplingLayer", 
    "MaskedAutoregressiveFlow",
    "InverseAutoregressiveFlow",
    "ContinuousFlow",
    "RealNVP",
    "RealNVPSpline",
    "NormalizingFlowModel"
]
