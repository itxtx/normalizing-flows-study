"""
Gradient checkpointing utilities for memory-efficient training of normalizing flows.

This module provides wrappers that apply gradient checkpointing to reduce memory usage
during backpropagation at the cost of additional forward computation.
"""

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from typing import List, Optional, Tuple, Union
import warnings

from ..flow.flow import Flow
from ..flow.sequential_flow import SequentialFlow


class CheckpointedFlow(Flow):
    """
    A wrapper that applies gradient checkpointing to a single flow layer.
    
    This reduces memory usage during backpropagation by not storing intermediate
    activations, instead recomputing them during the backward pass.
    
    Args:
        flow: The flow layer to wrap with checkpointing
        preserve_rng_state: Whether to preserve RNG state during checkpointing
    """
    
    def __init__(self, flow: Flow, preserve_rng_state: bool = True):
        super().__init__()
        self.flow = flow
        self.preserve_rng_state = preserve_rng_state
        self.data_dim = getattr(flow, 'data_dim', None)
    
    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass with gradient checkpointing."""
        if self.training and z.requires_grad:
            return checkpoint(
                self._forward_impl,
                z,
                preserve_rng_state=self.preserve_rng_state
            )
        else:
            return self._forward_impl(z)
    
    def _forward_impl(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Implementation of forward pass."""
        return self.flow.forward(z)
    
    def inverse(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Inverse pass with gradient checkpointing."""
        if self.training and x.requires_grad:
            return checkpoint(
                self._inverse_impl,
                x,
                preserve_rng_state=self.preserve_rng_state
            )
        else:
            return self._inverse_impl(x)
    
    def _inverse_impl(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Implementation of inverse pass."""
        return self.flow.inverse(x)


class CheckpointedSequentialFlow(SequentialFlow):
    """
    A sequential flow with configurable gradient checkpointing segments.
    
    This implementation automatically segments the flow layers and applies
    checkpointing to reduce memory usage with configurable memory-time tradeoffs.
    
    Args:
        flows: List of flow layers
        checkpoint_segments: Number of segments to divide flows into for checkpointing.
                           If None, no checkpointing is applied.
                           If 1, checkpoint the entire sequence.
                           If > 1, divide flows into segments and checkpoint each segment.
        preserve_rng_state: Whether to preserve RNG state during checkpointing
    """
    
    def __init__(
        self,
        flows: List[Flow],
        checkpoint_segments: Optional[int] = None,
        preserve_rng_state: bool = True
    ):
        super().__init__(flows)
        self.checkpoint_segments = checkpoint_segments
        self.preserve_rng_state = preserve_rng_state
        
        if checkpoint_segments is not None:
            if checkpoint_segments < 1:
                raise ValueError("checkpoint_segments must be >= 1 or None")
            if checkpoint_segments > len(flows):
                warnings.warn(
                    f"checkpoint_segments ({checkpoint_segments}) is greater than "
                    f"number of flows ({len(flows)}). Setting to {len(flows)}."
                )
                self.checkpoint_segments = len(flows)
        
        # Create segments for checkpointing
        self._create_segments()
    
    def _create_segments(self):
        """Create flow segments for checkpointing."""
        if self.checkpoint_segments is None:
            self.segments = None
            return
        
        flows_per_segment = len(self.flows) // self.checkpoint_segments
        remainder = len(self.flows) % self.checkpoint_segments
        
        self.segments = []
        start_idx = 0
        
        for i in range(self.checkpoint_segments):
            # Distribute remainder flows across first segments
            segment_size = flows_per_segment + (1 if i < remainder else 0)
            end_idx = start_idx + segment_size
            
            segment_flows = self.flows[start_idx:end_idx]
            self.segments.append(segment_flows)
            start_idx = end_idx
    
    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass with segmented checkpointing."""
        if self.segments is None or not self.training or not z.requires_grad:
            return super().forward(z)
        
        total_log_det = torch.zeros(z.size(0), device=z.device)
        
        for segment in self.segments:
            z, log_det = checkpoint(
                self._forward_segment,
                z,
                segment,
                preserve_rng_state=self.preserve_rng_state
            )
            total_log_det += log_det
        
        return z, total_log_det
    
    def _forward_segment(
        self, 
        z: torch.Tensor, 
        segment: List[Flow]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through a segment of flows."""
        total_log_det = torch.zeros(z.size(0), device=z.device)
        
        for flow in segment:
            z, log_det = flow.forward(z)
            total_log_det += log_det
        
        return z, total_log_det
    
    def inverse(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Inverse pass with segmented checkpointing."""
        if self.segments is None or not self.training or not x.requires_grad:
            return super().inverse(x)
        
        total_log_det = torch.zeros(x.size(0), device=x.device)
        
        # Apply segments in reverse order for inverse
        for segment in reversed(self.segments):
            x, log_det = checkpoint(
                self._inverse_segment,
                x,
                segment,
                preserve_rng_state=self.preserve_rng_state
            )
            total_log_det += log_det
        
        return x, total_log_det
    
    def _inverse_segment(
        self, 
        x: torch.Tensor, 
        segment: List[Flow]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Inverse pass through a segment of flows."""
        total_log_det = torch.zeros(x.size(0), device=x.device)
        
        # Apply flows in reverse order within segment
        for flow in reversed(segment):
            x, log_det = flow.inverse(x)
            total_log_det += log_det
        
        return x, total_log_det


def apply_gradient_checkpointing(
    flow: Flow,
    checkpoint_segments: Optional[int] = None,
    preserve_rng_state: bool = True
) -> Flow:
    """
    Apply gradient checkpointing to a flow.
    
    Args:
        flow: The flow to apply checkpointing to
        checkpoint_segments: Number of segments for sequential flows.
                           If None, applies checkpointing to individual layers.
        preserve_rng_state: Whether to preserve RNG state during checkpointing
    
    Returns:
        Flow with gradient checkpointing applied
    """
    if isinstance(flow, SequentialFlow):
        return CheckpointedSequentialFlow(
            list(flow.flows),
            checkpoint_segments=checkpoint_segments,
            preserve_rng_state=preserve_rng_state
        )
    else:
        return CheckpointedFlow(flow, preserve_rng_state=preserve_rng_state)


class MemoryEfficientWrapper(nn.Module):
    """
    A wrapper that provides memory usage monitoring and optimization suggestions.
    
    This wrapper tracks memory usage during training and provides recommendations
    for optimal checkpointing configuration.
    """
    
    def __init__(self, flow: Flow, monitor_memory: bool = True):
        super().__init__()
        self.flow = flow
        self.monitor_memory = monitor_memory
        self.memory_stats = {
            'peak_memory': 0,
            'forward_memory': 0,
            'backward_memory': 0,
            'num_calls': 0
        }
    
    def forward(self, *args, **kwargs):
        """Forward pass with memory monitoring."""
        if self.monitor_memory and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            initial_memory = torch.cuda.memory_allocated()
        
        result = self.flow(*args, **kwargs)
        
        if self.monitor_memory:
            self.memory_stats['num_calls'] += 1
            
            if torch.cuda.is_available():
                peak_memory = torch.cuda.max_memory_allocated()
                forward_memory = peak_memory - initial_memory
                
                self.memory_stats['peak_memory'] = max(
                    self.memory_stats['peak_memory'], peak_memory
                )
                self.memory_stats['forward_memory'] += forward_memory
        
        return result
    
    def get_memory_stats(self) -> dict:
        """Get memory usage statistics."""
        stats = self.memory_stats.copy()
        if stats['num_calls'] > 0:
            stats['avg_forward_memory'] = stats['forward_memory'] / stats['num_calls']
        return stats
    
    def suggest_checkpointing_config(self, target_memory_reduction: float = 0.5) -> dict:
        """
        Suggest optimal checkpointing configuration based on memory usage.
        
        Args:
            target_memory_reduction: Target memory reduction (0.0 to 1.0)
        
        Returns:
            Dictionary with checkpointing recommendations
        """
        if not isinstance(self.flow, SequentialFlow):
            return {
                'recommendation': 'single_layer_checkpointing',
                'reason': 'Non-sequential flow, apply CheckpointedFlow wrapper'
            }
        
        num_layers = len(self.flow.flows)
        
        # Estimate optimal number of segments based on target memory reduction
        # More segments = more memory savings but more recomputation
        if target_memory_reduction < 0.3:
            segments = max(1, num_layers // 4)
        elif target_memory_reduction < 0.6:
            segments = max(1, num_layers // 2)
        else:
            segments = num_layers
        
        return {
            'recommendation': 'segmented_checkpointing',
            'checkpoint_segments': min(segments, num_layers),
            'expected_memory_reduction': min(target_memory_reduction, 0.8),
            'expected_compute_overhead': f"{segments * 100 / num_layers:.1f}%",
            'reason': f'Sequential flow with {num_layers} layers'
        }