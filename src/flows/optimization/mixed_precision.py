"""
Mixed precision training utilities for normalizing flows.

This module provides wrappers and utilities for automatic mixed precision (AMP)
training to improve performance and reduce memory usage while maintaining
numerical stability.
"""

import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from typing import Optional, Dict, Any, Tuple, Union
import warnings

from ..flow.flow import Flow
from ..flow.sequential_flow import SequentialFlow


class MixedPrecisionFlow(Flow):
    """
    A wrapper that applies automatic mixed precision to a flow.
    
    This wrapper automatically handles mixed precision training with proper
    loss scaling and gradient clipping to maintain numerical stability.
    
    Args:
        flow: The flow to wrap with mixed precision
        enabled: Whether to enable mixed precision (requires CUDA)
        init_scale: Initial scale factor for gradient scaling
        growth_factor: Factor by which to multiply the scale when no overflow occurs
        backoff_factor: Factor by which to multiply the scale when overflow occurs
        growth_interval: Number of consecutive iterations without overflow before scaling up
        autocast_enabled: Whether to use autocast for forward passes
        autocast_dtype: Data type to use for autocast (torch.float16 or torch.bfloat16)
    """
    
    def __init__(
        self,
        flow: Flow,
        enabled: bool = True,
        init_scale: float = 2.**16,
        growth_factor: float = 2.0,
        backoff_factor: float = 0.5,
        growth_interval: int = 2000,
        autocast_enabled: bool = True,
        autocast_dtype: torch.dtype = torch.float16
    ):
        super().__init__()
        self.flow = flow
        self.data_dim = getattr(flow, 'data_dim', None)
        
        # Check if mixed precision is supported
        if enabled and not torch.cuda.is_available():
            warnings.warn(
                "Mixed precision training requires CUDA. Disabling mixed precision.",
                UserWarning
            )
            enabled = False
        
        if enabled and autocast_dtype not in [torch.float16, torch.bfloat16]:
            raise ValueError(
                f"autocast_dtype must be torch.float16 or torch.bfloat16, got {autocast_dtype}"
            )
        
        self.enabled = enabled
        self.autocast_enabled = autocast_enabled
        self.autocast_dtype = autocast_dtype
        
        # Initialize gradient scaler
        if self.enabled:
            self.scaler = GradScaler(
                init_scale=init_scale,
                growth_factor=growth_factor,
                backoff_factor=backoff_factor,
                growth_interval=growth_interval,
                enabled=enabled
            )
        else:
            self.scaler = None
        
        # Track statistics
        self.stats = {
            'forward_calls': 0,
            'overflow_count': 0,
            'scale_updates': 0,
            'current_scale': init_scale if enabled else 1.0
        }
    
    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass with mixed precision."""
        self.stats['forward_calls'] += 1
        
        if self.enabled and self.autocast_enabled:
            with autocast(dtype=self.autocast_dtype):
                return self.flow.forward(z)
        else:
            return self.flow.forward(z)
    
    def inverse(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Inverse pass with mixed precision."""
        if self.enabled and self.autocast_enabled:
            with autocast(dtype=self.autocast_dtype):
                return self.flow.inverse(x)
        else:
            return self.flow.inverse(x)
    
    def scale_loss(self, loss: torch.Tensor) -> torch.Tensor:
        """Scale loss for mixed precision training."""
        if self.enabled and self.scaler is not None:
            return self.scaler.scale(loss)
        return loss
    
    def step_optimizer(self, optimizer: torch.optim.Optimizer) -> bool:
        """
        Step optimizer with gradient scaling and overflow detection.
        
        Args:
            optimizer: The optimizer to step
            
        Returns:
            bool: True if the optimizer step was successful, False if overflow occurred
        """
        if not self.enabled or self.scaler is None:
            optimizer.step()
            return True
        
        # Check for overflow before stepping
        scale_before = self.scaler.get_scale()
        
        # Step with gradient scaling
        self.scaler.step(optimizer)
        self.scaler.update()
        
        # Check if overflow occurred
        scale_after = self.scaler.get_scale()
        overflow_occurred = scale_after < scale_before
        
        if overflow_occurred:
            self.stats['overflow_count'] += 1
        
        if scale_after != scale_before:
            self.stats['scale_updates'] += 1
            self.stats['current_scale'] = scale_after
        
        return not overflow_occurred
    
    def unscale_gradients(self, optimizer: torch.optim.Optimizer):
        """Unscale gradients for gradient clipping."""
        if self.enabled and self.scaler is not None:
            self.scaler.unscale_(optimizer)
    
    def get_scale(self) -> float:
        """Get current gradient scale."""
        if self.enabled and self.scaler is not None:
            return self.scaler.get_scale()
        return 1.0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get mixed precision training statistics."""
        stats = self.stats.copy()
        stats['current_scale'] = self.get_scale()
        stats['overflow_rate'] = (
            self.stats['overflow_count'] / max(1, self.stats['forward_calls'])
        )
        return stats
    
    def state_dict(self) -> Dict[str, Any]:
        """Get state dict including scaler state."""
        state = super().state_dict()
        if self.enabled and self.scaler is not None:
            state['scaler'] = self.scaler.state_dict()
        state['stats'] = self.stats
        return state
    
    def load_state_dict(self, state_dict: Dict[str, Any], strict: bool = True):
        """Load state dict including scaler state."""
        if 'scaler' in state_dict and self.enabled and self.scaler is not None:
            self.scaler.load_state_dict(state_dict.pop('scaler'))
        
        if 'stats' in state_dict:
            self.stats.update(state_dict.pop('stats'))
        
        super().load_state_dict(state_dict, strict)


class MixedPrecisionTrainer:
    """
    A training utility that handles mixed precision training for flows.
    
    This class provides a high-level interface for training flows with
    automatic mixed precision, including proper loss scaling, gradient
    clipping, and overflow handling.
    """
    
    def __init__(
        self,
        flow: Flow,
        optimizer: torch.optim.Optimizer,
        enabled: bool = True,
        max_grad_norm: Optional[float] = 1.0,
        **amp_kwargs
    ):
        self.flow = flow
        self.optimizer = optimizer
        self.max_grad_norm = max_grad_norm
        
        # Wrap flow with mixed precision if not already wrapped
        if not isinstance(flow, MixedPrecisionFlow):
            self.mp_flow = MixedPrecisionFlow(flow, enabled=enabled, **amp_kwargs)
        else:
            self.mp_flow = flow
        
        self.enabled = self.mp_flow.enabled
        
        # Training statistics
        self.training_stats = {
            'total_steps': 0,
            'successful_steps': 0,
            'overflow_steps': 0,
            'gradient_clips': 0
        }
    
    def training_step(
        self,
        batch: torch.Tensor,
        base_dist: torch.distributions.Distribution
    ) -> Dict[str, torch.Tensor]:
        """
        Perform a single training step with mixed precision.
        
        Args:
            batch: Input batch
            base_dist: Base distribution for computing log probability
            
        Returns:
            Dictionary containing loss and other metrics
        """
        self.training_stats['total_steps'] += 1
        
        # Zero gradients
        self.optimizer.zero_grad()
        
        # Forward pass with mixed precision
        log_prob = self.mp_flow.log_prob(batch, base_dist)
        loss = -log_prob.mean()
        
        # Scale loss for mixed precision
        scaled_loss = self.mp_flow.scale_loss(loss)
        
        # Backward pass
        scaled_loss.backward()
        
        # Unscale gradients for clipping
        if self.enabled:
            self.mp_flow.unscale_gradients(self.optimizer)
        
        # Gradient clipping
        if self.max_grad_norm is not None:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.flow.parameters(), 
                self.max_grad_norm
            )
            if grad_norm > self.max_grad_norm:
                self.training_stats['gradient_clips'] += 1
        else:
            grad_norm = torch.tensor(0.0)
        
        # Optimizer step with overflow detection
        step_successful = self.mp_flow.step_optimizer(self.optimizer)
        
        if step_successful:
            self.training_stats['successful_steps'] += 1
        else:
            self.training_stats['overflow_steps'] += 1
        
        return {
            'loss': loss.detach(),
            'scaled_loss': scaled_loss.detach(),
            'grad_norm': grad_norm,
            'step_successful': step_successful,
            'scale': self.mp_flow.get_scale()
        }
    
    def get_training_stats(self) -> Dict[str, Any]:
        """Get comprehensive training statistics."""
        stats = self.training_stats.copy()
        stats.update(self.mp_flow.get_stats())
        
        if stats['total_steps'] > 0:
            stats['success_rate'] = stats['successful_steps'] / stats['total_steps']
            stats['overflow_rate'] = stats['overflow_steps'] / stats['total_steps']
            stats['clip_rate'] = stats['gradient_clips'] / stats['total_steps']
        
        return stats
    
    def state_dict(self) -> Dict[str, Any]:
        """Get trainer state dict."""
        return {
            'mp_flow': self.mp_flow.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'training_stats': self.training_stats
        }
    
    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Load trainer state dict."""
        if 'mp_flow' in state_dict:
            self.mp_flow.load_state_dict(state_dict['mp_flow'])
        
        if 'optimizer' in state_dict:
            self.optimizer.load_state_dict(state_dict['optimizer'])
        
        if 'training_stats' in state_dict:
            self.training_stats.update(state_dict['training_stats'])


def check_mixed_precision_compatibility(flow: Flow) -> Dict[str, Any]:
    """
    Check if a flow is compatible with mixed precision training.
    
    Args:
        flow: The flow to check
        
    Returns:
        Dictionary with compatibility information and recommendations
    """
    compatibility = {
        'compatible': True,
        'warnings': [],
        'recommendations': []
    }
    
    # Check for CUDA availability
    if not torch.cuda.is_available():
        compatibility['compatible'] = False
        compatibility['warnings'].append("CUDA is not available")
        compatibility['recommendations'].append("Mixed precision requires CUDA")
        return compatibility
    
    # Check for operations that might be problematic with mixed precision
    problematic_ops = []
    
    # Check for custom operations that might not support autocast
    for name, module in flow.named_modules():
        if hasattr(module, 'forward'):
            # Check for operations that typically need full precision
            module_str = str(module)
            if any(op in module_str.lower() for op in ['batchnorm', 'layernorm', 'softmax']):
                problematic_ops.append(f"{name}: {type(module).__name__}")
    
    if problematic_ops:
        compatibility['warnings'].append(
            f"Found operations that may need careful handling: {problematic_ops}"
        )
        compatibility['recommendations'].append(
            "Consider using torch.bfloat16 instead of torch.float16 for better stability"
        )
    
    # Check parameter count for memory benefits
    param_count = sum(p.numel() for p in flow.parameters())
    if param_count < 1e6:  # Less than 1M parameters
        compatibility['recommendations'].append(
            f"Model has {param_count:,} parameters. Mixed precision benefits may be limited for small models."
        )
    
    return compatibility


def apply_mixed_precision(
    flow: Flow,
    enabled: bool = True,
    check_compatibility: bool = True,
    **kwargs
) -> MixedPrecisionFlow:
    """
    Apply mixed precision to a flow with automatic compatibility checking.
    
    Args:
        flow: The flow to wrap
        enabled: Whether to enable mixed precision
        check_compatibility: Whether to check compatibility first
        **kwargs: Additional arguments for MixedPrecisionFlow
        
    Returns:
        MixedPrecisionFlow wrapper
    """
    if check_compatibility:
        compat = check_mixed_precision_compatibility(flow)
        
        if not compat['compatible']:
            warnings.warn(
                f"Mixed precision may not work properly: {'; '.join(compat['warnings'])}",
                UserWarning
            )
            enabled = False
        
        for warning in compat['warnings']:
            warnings.warn(f"Mixed precision warning: {warning}", UserWarning)
        
        for rec in compat['recommendations']:
            print(f"Mixed precision recommendation: {rec}")
    
    return MixedPrecisionFlow(flow, enabled=enabled, **kwargs)