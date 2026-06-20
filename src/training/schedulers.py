"""
Advanced learning rate schedulers for normalizing flows.

This module provides flow-specific adaptive schedulers that monitor
log-likelihood convergence and other flow-specific metrics.
"""

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import _LRScheduler
from typing import Dict, List, Optional, Union, Any
import numpy as np
from collections import deque
import warnings


class AdaptiveFlowScheduler(_LRScheduler):
    """
    Adaptive learning rate scheduler specifically designed for normalizing flows.
    
    This scheduler monitors flow-specific metrics like log-likelihood convergence,
    gradient norms, and Jacobian condition numbers to adaptively adjust learning rates.
    
    Args:
        optimizer: The optimizer to schedule
        patience: Number of epochs to wait before reducing LR
        factor: Factor by which to reduce learning rate
        min_lr: Minimum learning rate
        threshold: Minimum change in monitored metric to qualify as improvement
        threshold_mode: 'rel' for relative change, 'abs' for absolute change
        cooldown: Number of epochs to wait after LR reduction before resuming monitoring
        verbose: Whether to print messages when LR is reduced
    """
    
    def __init__(
        self,
        optimizer: optim.Optimizer,
        patience: int = 10,
        factor: float = 0.5,
        min_lr: Union[float, List[float]] = 1e-8,
        threshold: float = 1e-4,
        threshold_mode: str = 'rel',
        cooldown: int = 0,
        verbose: bool = False
    ):
        self.patience = patience
        self.factor = factor
        self.min_lr = min_lr if isinstance(min_lr, list) else [min_lr] * len(optimizer.param_groups)
        self.threshold = threshold
        self.threshold_mode = threshold_mode
        self.cooldown = cooldown
        self.verbose = verbose
        
        # Internal state
        self.best_metric = None
        self.num_bad_epochs = 0
        self.cooldown_counter = 0
        self.last_epoch = 0
        
        # Metric history for trend analysis
        self.metric_history = deque(maxlen=50)
        self.gradient_history = deque(maxlen=20)
        
        super().__init__(optimizer, last_epoch=-1)
    
    def step(self, metrics: Optional[Dict[str, float]] = None):
        """
        Step the scheduler with flow-specific metrics.
        
        Args:
            metrics: Dictionary containing flow metrics like:
                - 'log_likelihood': Average log-likelihood
                - 'gradient_norm': Gradient norm
                - 'jacobian_condition': Jacobian condition number
                - 'loss': Training loss
        """
        if metrics is None:
            metrics = {}
            
        # Primary metric for scheduling (log-likelihood or loss)
        primary_metric = metrics.get('log_likelihood', metrics.get('loss'))
        
        if primary_metric is not None:
            self.metric_history.append(primary_metric)
            
            # Check for improvement
            if self.best_metric is None:
                self.best_metric = primary_metric
            else:
                if self._is_better(primary_metric, self.best_metric):
                    self.best_metric = primary_metric
                    self.num_bad_epochs = 0
                else:
                    self.num_bad_epochs += 1
        
        # Store gradient information for additional analysis
        if 'gradient_norm' in metrics:
            self.gradient_history.append(metrics['gradient_norm'])
        
        # Check if we should reduce learning rate
        if self.cooldown_counter > 0:
            self.cooldown_counter -= 1
        elif self.num_bad_epochs >= self.patience:
            self._reduce_lr()
            self.cooldown_counter = self.cooldown
            self.num_bad_epochs = 0
        
        self.last_epoch += 1
    
    def _is_better(self, current: float, best: float) -> bool:
        """Check if current metric is better than best."""
        if self.threshold_mode == 'rel':
            rel_epsilon = 1. - self.threshold
            return current > best * rel_epsilon
        else:  # abs
            return current > best + self.threshold
    
    def _reduce_lr(self):
        """Reduce learning rate for all parameter groups."""
        for i, param_group in enumerate(self.optimizer.param_groups):
            old_lr = param_group['lr']
            new_lr = max(old_lr * self.factor, self.min_lr[i])
            param_group['lr'] = new_lr
            
            if self.verbose:
                print(f'Reducing learning rate from {old_lr:.2e} to {new_lr:.2e}')
    
    def get_lr(self):
        """Return current learning rates."""
        return [group['lr'] for group in self.optimizer.param_groups]
    
    def get_metric_trend(self) -> Optional[str]:
        """Analyze recent metric trend."""
        if len(self.metric_history) < 5:
            return None
            
        recent_metrics = list(self.metric_history)[-5:]
        trend = np.polyfit(range(len(recent_metrics)), recent_metrics, 1)[0]
        
        if trend > self.threshold:
            return 'improving'
        elif trend < -self.threshold:
            return 'degrading'
        else:
            return 'stable'


class LogLikelihoodScheduler(_LRScheduler):
    """
    Scheduler that specifically monitors log-likelihood convergence.
    
    This scheduler is designed for density modeling tasks where log-likelihood
    is the primary metric of interest.
    """
    
    def __init__(
        self,
        optimizer: optim.Optimizer,
        patience: int = 15,
        factor: float = 0.7,
        min_lr: float = 1e-7,
        convergence_threshold: float = 1e-5,
        lookback_window: int = 10,
        verbose: bool = False
    ):
        self.patience = patience
        self.factor = factor
        self.min_lr = min_lr
        self.convergence_threshold = convergence_threshold
        self.lookback_window = lookback_window
        self.verbose = verbose
        
        # State tracking
        self.log_likelihood_history = deque(maxlen=100)
        self.best_log_likelihood = -float('inf')
        self.epochs_without_improvement = 0
        self.converged = False
        
        super().__init__(optimizer, last_epoch=-1)
    
    def step(self, log_likelihood: float):
        """
        Step with log-likelihood value.
        
        Args:
            log_likelihood: Current epoch's average log-likelihood
        """
        self.log_likelihood_history.append(log_likelihood)
        
        # Check for improvement
        if log_likelihood > self.best_log_likelihood + self.convergence_threshold:
            self.best_log_likelihood = log_likelihood
            self.epochs_without_improvement = 0
        else:
            self.epochs_without_improvement += 1
        
        # Check for convergence
        if len(self.log_likelihood_history) >= self.lookback_window:
            recent_ll = list(self.log_likelihood_history)[-self.lookback_window:]
            ll_std = np.std(recent_ll)
            
            if ll_std < self.convergence_threshold:
                if not self.converged:
                    self.converged = True
                    if self.verbose:
                        print(f'Log-likelihood converged (std: {ll_std:.2e})')
        
        # Reduce learning rate if no improvement
        if self.epochs_without_improvement >= self.patience:
            self._reduce_lr()
            self.epochs_without_improvement = 0
        
        self.last_epoch += 1
    
    def _reduce_lr(self):
        """Reduce learning rate."""
        for param_group in self.optimizer.param_groups:
            old_lr = param_group['lr']
            new_lr = max(old_lr * self.factor, self.min_lr)
            param_group['lr'] = new_lr
            
            if self.verbose:
                print(f'Reducing LR: {old_lr:.2e} -> {new_lr:.2e} '
                      f'(no improvement for {self.patience} epochs)')
    
    def get_lr(self):
        """Return current learning rates."""
        return [group['lr'] for group in self.optimizer.param_groups]
    
    def is_converged(self) -> bool:
        """Check if training has converged."""
        return self.converged
    
    def get_convergence_info(self) -> Dict[str, Any]:
        """Get detailed convergence information."""
        if len(self.log_likelihood_history) >= self.lookback_window:
            recent_ll = list(self.log_likelihood_history)[-self.lookback_window:]
            ll_mean = np.mean(recent_ll)
            ll_std = np.std(recent_ll)
            ll_trend = np.polyfit(range(len(recent_ll)), recent_ll, 1)[0]
        else:
            ll_mean = ll_std = ll_trend = None
        
        return {
            'converged': self.converged,
            'best_log_likelihood': self.best_log_likelihood,
            'epochs_without_improvement': self.epochs_without_improvement,
            'recent_mean': ll_mean,
            'recent_std': ll_std,
            'recent_trend': ll_trend
        }


class FlowPlateauScheduler(_LRScheduler):
    """
    Plateau scheduler with flow-specific enhancements.
    
    This scheduler monitors multiple flow metrics and uses sophisticated
    plateau detection algorithms.
    """
    
    def __init__(
        self,
        optimizer: optim.Optimizer,
        mode: str = 'max',
        factor: float = 0.5,
        patience: int = 10,
        threshold: float = 1e-4,
        threshold_mode: str = 'rel',
        cooldown: int = 0,
        min_lr: float = 1e-8,
        eps: float = 1e-8,
        verbose: bool = False,
        # Flow-specific parameters
        gradient_threshold: float = 1e-6,
        jacobian_threshold: float = 1e3,
        use_gradient_plateau: bool = True,
        use_jacobian_monitoring: bool = True
    ):
        self.mode = mode
        self.factor = factor
        self.patience = patience
        self.threshold = threshold
        self.threshold_mode = threshold_mode
        self.cooldown = cooldown
        self.min_lr = min_lr
        self.eps = eps
        self.verbose = verbose
        
        # Flow-specific parameters
        self.gradient_threshold = gradient_threshold
        self.jacobian_threshold = jacobian_threshold
        self.use_gradient_plateau = use_gradient_plateau
        self.use_jacobian_monitoring = use_jacobian_monitoring
        
        # State tracking
        self.best = None
        self.num_bad_epochs = 0
        self.cooldown_counter = 0
        
        # Flow-specific state
        self.gradient_history = deque(maxlen=20)
        self.jacobian_history = deque(maxlen=20)
        self.gradient_plateau_detected = False
        self.jacobian_instability_detected = False
        
        super().__init__(optimizer, last_epoch=-1)
    
    def step(self, metrics: Dict[str, float]):
        """
        Step with comprehensive flow metrics.
        
        Args:
            metrics: Dictionary with keys like 'loss', 'log_likelihood',
                    'gradient_norm', 'jacobian_condition', etc.
        """
        # Primary metric for plateau detection
        primary_metric = metrics.get('log_likelihood', metrics.get('loss'))
        
        if primary_metric is None:
            warnings.warn("No primary metric provided for plateau detection")
            return
        
        # Store gradient and Jacobian information
        if 'gradient_norm' in metrics:
            self.gradient_history.append(metrics['gradient_norm'])
        
        if 'jacobian_condition' in metrics:
            self.jacobian_history.append(metrics['jacobian_condition'])
        
        # Check for various plateau conditions
        primary_plateau = self._check_primary_plateau(primary_metric)
        gradient_plateau = self._check_gradient_plateau() if self.use_gradient_plateau else False
        jacobian_instability = self._check_jacobian_instability() if self.use_jacobian_monitoring else False
        
        # Reduce LR if any plateau condition is met
        if self.cooldown_counter > 0:
            self.cooldown_counter -= 1
        elif primary_plateau or gradient_plateau or jacobian_instability:
            self._reduce_lr(primary_plateau, gradient_plateau, jacobian_instability)
            self.cooldown_counter = self.cooldown
            self.num_bad_epochs = 0
        
        self.last_epoch += 1
    
    def _check_primary_plateau(self, metric: float) -> bool:
        """Check if primary metric has plateaued."""
        if self.best is None:
            self.best = metric
            return False
        
        if self.mode == 'min':
            is_better = metric < self.best - self._get_threshold()
        else:  # max
            is_better = metric > self.best + self._get_threshold()
        
        if is_better:
            self.best = metric
            self.num_bad_epochs = 0
            return False
        else:
            self.num_bad_epochs += 1
            return self.num_bad_epochs >= self.patience
    
    def _check_gradient_plateau(self) -> bool:
        """Check if gradients have plateaued (very small)."""
        if len(self.gradient_history) < 5:
            return False
        
        recent_gradients = list(self.gradient_history)[-5:]
        avg_gradient = np.mean(recent_gradients)
        
        plateau_detected = avg_gradient < self.gradient_threshold
        
        if plateau_detected and not self.gradient_plateau_detected:
            self.gradient_plateau_detected = True
            return True
        
        if not plateau_detected:
            self.gradient_plateau_detected = False
        
        return False
    
    def _check_jacobian_instability(self) -> bool:
        """Check for Jacobian instability (high condition numbers)."""
        if len(self.jacobian_history) < 3:
            return False
        
        recent_jacobians = list(self.jacobian_history)[-3:]
        max_condition = max(recent_jacobians)
        
        instability_detected = max_condition > self.jacobian_threshold
        
        if instability_detected and not self.jacobian_instability_detected:
            self.jacobian_instability_detected = True
            return True
        
        if not instability_detected:
            self.jacobian_instability_detected = False
        
        return False
    
    def _get_threshold(self) -> float:
        """Get threshold for improvement detection."""
        if self.threshold_mode == 'rel':
            return abs(self.best) * self.threshold
        else:
            return self.threshold
    
    def _reduce_lr(self, primary_plateau: bool, gradient_plateau: bool, jacobian_instability: bool):
        """Reduce learning rate with detailed logging."""
        reasons = []
        if primary_plateau:
            reasons.append("primary metric plateau")
        if gradient_plateau:
            reasons.append("gradient plateau")
        if jacobian_instability:
            reasons.append("Jacobian instability")
        
        for param_group in self.optimizer.param_groups:
            old_lr = param_group['lr']
            new_lr = max(old_lr * self.factor, self.min_lr)
            param_group['lr'] = new_lr
            
            if self.verbose:
                reason_str = ", ".join(reasons)
                print(f'Reducing LR: {old_lr:.2e} -> {new_lr:.2e} ({reason_str})')
    
    def get_lr(self):
        """Return current learning rates."""
        return [group['lr'] for group in self.optimizer.param_groups]
    
    def get_plateau_info(self) -> Dict[str, Any]:
        """Get detailed plateau detection information."""
        return {
            'best_metric': self.best,
            'num_bad_epochs': self.num_bad_epochs,
            'gradient_plateau_detected': self.gradient_plateau_detected,
            'jacobian_instability_detected': self.jacobian_instability_detected,
            'recent_gradient_norm': list(self.gradient_history)[-1] if self.gradient_history else None,
            'recent_jacobian_condition': list(self.jacobian_history)[-1] if self.jacobian_history else None
        }


def create_flow_scheduler(
    scheduler_type: str,
    optimizer: optim.Optimizer,
    **kwargs
) -> _LRScheduler:
    """
    Factory function to create flow-specific schedulers.
    
    Args:
        scheduler_type: Type of scheduler ('adaptive', 'log_likelihood', 'plateau')
        optimizer: PyTorch optimizer
        **kwargs: Scheduler-specific arguments
    
    Returns:
        Configured scheduler instance
    """
    scheduler_map = {
        'adaptive': AdaptiveFlowScheduler,
        'log_likelihood': LogLikelihoodScheduler,
        'plateau': FlowPlateauScheduler
    }
    
    if scheduler_type not in scheduler_map:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}. "
                        f"Available types: {list(scheduler_map.keys())}")
    
    return scheduler_map[scheduler_type](optimizer, **kwargs)