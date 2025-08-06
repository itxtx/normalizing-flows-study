"""
Optimization utilities for normalizing flows.
"""

from .gradient_checkpointing import CheckpointedFlow, CheckpointedSequentialFlow
from .mixed_precision import MixedPrecisionFlow, MixedPrecisionTrainer, apply_mixed_precision
from .cuda_kernels import (
    CUDASplineEvaluator,
    CUDAAutoregressiveSampler,
    CUDALogDetComputer,
    CUDAFlowOptimizer,
    get_cuda_optimizer,
    benchmark_cuda_kernels
)

__all__ = [
    'CheckpointedFlow',
    'CheckpointedSequentialFlow',
    'MixedPrecisionFlow',
    'MixedPrecisionTrainer',
    'apply_mixed_precision',
    'CUDASplineEvaluator',
    'CUDAAutoregressiveSampler', 
    'CUDALogDetComputer',
    'CUDAFlowOptimizer',
    'get_cuda_optimizer',
    'benchmark_cuda_kernels'
]