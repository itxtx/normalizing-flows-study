"""
Tests for advanced learning rate schedulers.
"""

import pytest
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from unittest.mock import Mock

from src.training.schedulers import (
    AdaptiveFlowScheduler,
    LogLikelihoodScheduler,
    FlowPlateauScheduler,
    create_flow_scheduler
)


class SimpleModel(nn.Module):
    """Simple model for testing schedulers."""
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 1)
    
    def forward(self, x):
        return self.linear(x)


@pytest.fixture
def model_and_optimizer():
    """Create a simple model and optimizer for testing."""
    model = SimpleModel()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    return model, optimizer


class TestAdaptiveFlowScheduler:
    """Test AdaptiveFlowScheduler functionality."""
    
    def test_initialization(self, model_and_optimizer):
        """Test scheduler initialization."""
        model, optimizer = model_and_optimizer
        scheduler = AdaptiveFlowScheduler(optimizer, patience=5, factor=0.5)
        
        assert scheduler.patience == 5
        assert scheduler.factor == 0.5
        assert scheduler.num_bad_epochs == 0
        assert scheduler.best_metric is None
    
    def test_step_with_improvement(self, model_and_optimizer):
        """Test scheduler step with improving metrics."""
        model, optimizer = model_and_optimizer
        scheduler = AdaptiveFlowScheduler(optimizer, patience=3, factor=0.5)
        
        initial_lr = optimizer.param_groups[0]['lr']
        
        # Simulate improving log-likelihood
        scheduler.step({'log_likelihood': -10.0})
        scheduler.step({'log_likelihood': -9.0})
        scheduler.step({'log_likelihood': -8.0})
        
        # LR should not change with improvement
        assert optimizer.param_groups[0]['lr'] == initial_lr
        assert scheduler.num_bad_epochs == 0
    
    def test_step_with_plateau(self, model_and_optimizer):
        """Test scheduler step with plateaued metrics."""
        model, optimizer = model_and_optimizer
        scheduler = AdaptiveFlowScheduler(optimizer, patience=2, factor=0.5, verbose=True)
        
        initial_lr = optimizer.param_groups[0]['lr']
        
        # Set initial best metric
        scheduler.step({'log_likelihood': -10.0})
        
        # Simulate plateau (no improvement)
        scheduler.step({'log_likelihood': -10.1})  # Worse
        scheduler.step({'log_likelihood': -10.2})  # Still worse
        scheduler.step({'log_likelihood': -10.1})  # Trigger reduction
        
        # LR should be reduced
        assert optimizer.param_groups[0]['lr'] == initial_lr * 0.5
    
    def test_step_with_loss_metric(self, model_and_optimizer):
        """Test scheduler with loss instead of log-likelihood."""
        model, optimizer = model_and_optimizer
        scheduler = AdaptiveFlowScheduler(optimizer, patience=2, factor=0.5)
        
        # Use loss metric (lower is better, but scheduler treats it as higher is better)
        scheduler.step({'loss': 1.0})
        scheduler.step({'loss': 2.0})  # Higher loss (worse)
        scheduler.step({'loss': 2.1})  # Still worse
        scheduler.step({'loss': 2.2})  # Trigger reduction
        
        assert scheduler.num_bad_epochs == 0  # Reset after reduction
    
    def test_gradient_history_tracking(self, model_and_optimizer):
        """Test gradient norm history tracking."""
        model, optimizer = model_and_optimizer
        scheduler = AdaptiveFlowScheduler(optimizer)
        
        scheduler.step({'log_likelihood': -10.0, 'gradient_norm': 1.0})
        scheduler.step({'log_likelihood': -9.0, 'gradient_norm': 0.5})
        
        assert len(scheduler.gradient_history) == 2
        assert list(scheduler.gradient_history) == [1.0, 0.5]
    
    def test_metric_trend_analysis(self, model_and_optimizer):
        """Test metric trend analysis."""
        model, optimizer = model_and_optimizer
        scheduler = AdaptiveFlowScheduler(optimizer)
        
        # Create improving trend
        for i in range(10):
            scheduler.step({'log_likelihood': -10.0 + i * 0.5})
        
        trend = scheduler.get_metric_trend()
        assert trend == 'improving'
        
        # Create degrading trend
        for i in range(5):
            scheduler.step({'log_likelihood': -5.0 - i * 0.5})
        
        trend = scheduler.get_metric_trend()
        assert trend == 'degrading'
    
    def test_cooldown_period(self, model_and_optimizer):
        """Test cooldown period after LR reduction."""
        model, optimizer = model_and_optimizer
        scheduler = AdaptiveFlowScheduler(optimizer, patience=1, cooldown=2, factor=0.5)
        
        initial_lr = optimizer.param_groups[0]['lr']
        
        # Trigger first reduction
        scheduler.step({'log_likelihood': -10.0})
        scheduler.step({'log_likelihood': -11.0})  # Worse
        scheduler.step({'log_likelihood': -12.0})  # Trigger reduction
        
        first_reduced_lr = optimizer.param_groups[0]['lr']
        assert first_reduced_lr == initial_lr * 0.5
        
        # During cooldown, no further reduction should occur
        scheduler.step({'log_likelihood': -13.0})  # Worse
        scheduler.step({'log_likelihood': -14.0})  # Still worse
        
        # LR should remain the same during cooldown
        assert optimizer.param_groups[0]['lr'] == first_reduced_lr
    
    def test_minimum_lr_constraint(self, model_and_optimizer):
        """Test minimum learning rate constraint."""
        model, optimizer = model_and_optimizer
        scheduler = AdaptiveFlowScheduler(optimizer, patience=1, factor=0.1, min_lr=1e-6)
        
        # Reduce LR multiple times
        scheduler.step({'log_likelihood': -10.0})
        for i in range(10):
            scheduler.step({'log_likelihood': -10.0 - i})  # Keep getting worse
        
        # LR should not go below min_lr
        assert optimizer.param_groups[0]['lr'] >= 1e-6


class TestLogLikelihoodScheduler:
    """Test LogLikelihoodScheduler functionality."""
    
    def test_initialization(self, model_and_optimizer):
        """Test scheduler initialization."""
        model, optimizer = model_and_optimizer
        scheduler = LogLikelihoodScheduler(optimizer, patience=10, factor=0.7)
        
        assert scheduler.patience == 10
        assert scheduler.factor == 0.7
        assert scheduler.best_log_likelihood == -float('inf')
        assert not scheduler.converged
    
    def test_improvement_tracking(self, model_and_optimizer):
        """Test log-likelihood improvement tracking."""
        model, optimizer = model_and_optimizer
        scheduler = LogLikelihoodScheduler(optimizer, patience=3, convergence_threshold=1e-3)
        
        scheduler.step(-10.0)
        assert scheduler.best_log_likelihood == -10.0
        assert scheduler.epochs_without_improvement == 0
        
        scheduler.step(-9.0)  # Improvement
        assert scheduler.best_log_likelihood == -9.0
        assert scheduler.epochs_without_improvement == 0
        
        scheduler.step(-9.1)  # No significant improvement
        assert scheduler.epochs_without_improvement == 1
    
    def test_convergence_detection(self, model_and_optimizer):
        """Test convergence detection based on stability."""
        model, optimizer = model_and_optimizer
        scheduler = LogLikelihoodScheduler(
            optimizer, 
            lookback_window=5, 
            convergence_threshold=1e-4
        )
        
        # Add stable log-likelihood values
        stable_ll = -5.0
        for i in range(10):
            scheduler.step(stable_ll + np.random.normal(0, 1e-5))
        
        assert scheduler.is_converged()
    
    def test_lr_reduction_on_plateau(self, model_and_optimizer):
        """Test LR reduction when no improvement occurs."""
        model, optimizer = model_and_optimizer
        scheduler = LogLikelihoodScheduler(optimizer, patience=2, factor=0.5)
        
        initial_lr = optimizer.param_groups[0]['lr']
        
        scheduler.step(-10.0)  # Set baseline
        scheduler.step(-10.1)  # No improvement
        scheduler.step(-10.2)  # No improvement
        scheduler.step(-10.1)  # Trigger reduction
        
        assert optimizer.param_groups[0]['lr'] == initial_lr * 0.5
    
    def test_convergence_info(self, model_and_optimizer):
        """Test convergence information retrieval."""
        model, optimizer = model_and_optimizer
        scheduler = LogLikelihoodScheduler(optimizer, lookback_window=3)
        
        # Add some history
        scheduler.step(-10.0)
        scheduler.step(-9.5)
        scheduler.step(-9.0)
        scheduler.step(-8.8)
        
        info = scheduler.get_convergence_info()
        
        assert 'converged' in info
        assert 'best_log_likelihood' in info
        assert 'epochs_without_improvement' in info
        assert 'recent_mean' in info
        assert 'recent_std' in info
        assert 'recent_trend' in info
        
        assert info['best_log_likelihood'] == -8.8


class TestFlowPlateauScheduler:
    """Test FlowPlateauScheduler functionality."""
    
    def test_initialization(self, model_and_optimizer):
        """Test scheduler initialization."""
        model, optimizer = model_and_optimizer
        scheduler = FlowPlateauScheduler(
            optimizer, 
            mode='max', 
            patience=5,
            use_gradient_plateau=True,
            use_jacobian_monitoring=True
        )
        
        assert scheduler.mode == 'max'
        assert scheduler.patience == 5
        assert scheduler.use_gradient_plateau
        assert scheduler.use_jacobian_monitoring
    
    def test_primary_metric_plateau(self, model_and_optimizer):
        """Test primary metric plateau detection."""
        model, optimizer = model_and_optimizer
        scheduler = FlowPlateauScheduler(optimizer, mode='max', patience=2, factor=0.5)
        
        initial_lr = optimizer.param_groups[0]['lr']
        
        # Simulate plateau in log-likelihood
        scheduler.step({'log_likelihood': -10.0})
        scheduler.step({'log_likelihood': -10.1})  # Worse
        scheduler.step({'log_likelihood': -10.2})  # Still worse
        scheduler.step({'log_likelihood': -10.1})  # Trigger reduction
        
        assert optimizer.param_groups[0]['lr'] == initial_lr * 0.5
    
    def test_gradient_plateau_detection(self, model_and_optimizer):
        """Test gradient plateau detection."""
        model, optimizer = model_and_optimizer
        scheduler = FlowPlateauScheduler(
            optimizer, 
            patience=10,  # High patience for primary metric
            gradient_threshold=1e-5,
            use_gradient_plateau=True,
            factor=0.5
        )
        
        initial_lr = optimizer.param_groups[0]['lr']
        
        # Simulate very small gradients (plateau)
        for i in range(6):
            scheduler.step({
                'log_likelihood': -10.0 + i * 0.01,  # Slight improvement
                'gradient_norm': 1e-6  # Very small gradient
            })
        
        # LR should be reduced due to gradient plateau
        assert optimizer.param_groups[0]['lr'] < initial_lr
    
    def test_jacobian_instability_detection(self, model_and_optimizer):
        """Test Jacobian instability detection."""
        model, optimizer = model_and_optimizer
        scheduler = FlowPlateauScheduler(
            optimizer,
            patience=10,  # High patience for primary metric
            jacobian_threshold=100,
            use_jacobian_monitoring=True,
            factor=0.5
        )
        
        initial_lr = optimizer.param_groups[0]['lr']
        
        # Simulate high condition numbers (instability)
        for i in range(4):
            scheduler.step({
                'log_likelihood': -10.0 + i * 0.01,  # Slight improvement
                'jacobian_condition': 1000  # High condition number
            })
        
        # LR should be reduced due to Jacobian instability
        assert optimizer.param_groups[0]['lr'] < initial_lr
    
    def test_plateau_info(self, model_and_optimizer):
        """Test plateau information retrieval."""
        model, optimizer = model_and_optimizer
        scheduler = FlowPlateauScheduler(optimizer)
        
        scheduler.step({
            'log_likelihood': -10.0,
            'gradient_norm': 0.1,
            'jacobian_condition': 10.0
        })
        
        info = scheduler.get_plateau_info()
        
        assert 'best_metric' in info
        assert 'num_bad_epochs' in info
        assert 'gradient_plateau_detected' in info
        assert 'jacobian_instability_detected' in info
        assert 'recent_gradient_norm' in info
        assert 'recent_jacobian_condition' in info
    
    def test_multiple_plateau_conditions(self, model_and_optimizer):
        """Test handling of multiple plateau conditions simultaneously."""
        model, optimizer = model_and_optimizer
        scheduler = FlowPlateauScheduler(
            optimizer,
            patience=1,
            gradient_threshold=1e-5,
            jacobian_threshold=100,
            factor=0.5,
            verbose=True
        )
        
        initial_lr = optimizer.param_groups[0]['lr']
        
        # Trigger multiple plateau conditions
        scheduler.step({'log_likelihood': -10.0})  # Baseline
        scheduler.step({
            'log_likelihood': -10.1,  # Worse (primary plateau)
            'gradient_norm': 1e-6,    # Small gradient
            'jacobian_condition': 1000  # High condition number
        })
        scheduler.step({
            'log_likelihood': -10.2,  # Still worse
            'gradient_norm': 1e-6,    # Still small
            'jacobian_condition': 1000  # Still high
        })
        
        # LR should be reduced
        assert optimizer.param_groups[0]['lr'] < initial_lr


class TestSchedulerFactory:
    """Test scheduler factory function."""
    
    def test_create_adaptive_scheduler(self, model_and_optimizer):
        """Test creating adaptive scheduler."""
        model, optimizer = model_and_optimizer
        scheduler = create_flow_scheduler('adaptive', optimizer, patience=5)
        
        assert isinstance(scheduler, AdaptiveFlowScheduler)
        assert scheduler.patience == 5
    
    def test_create_log_likelihood_scheduler(self, model_and_optimizer):
        """Test creating log-likelihood scheduler."""
        model, optimizer = model_and_optimizer
        scheduler = create_flow_scheduler('log_likelihood', optimizer, patience=10)
        
        assert isinstance(scheduler, LogLikelihoodScheduler)
        assert scheduler.patience == 10
    
    def test_create_plateau_scheduler(self, model_and_optimizer):
        """Test creating plateau scheduler."""
        model, optimizer = model_and_optimizer
        scheduler = create_flow_scheduler('plateau', optimizer, mode='min')
        
        assert isinstance(scheduler, FlowPlateauScheduler)
        assert scheduler.mode == 'min'
    
    def test_invalid_scheduler_type(self, model_and_optimizer):
        """Test error handling for invalid scheduler type."""
        model, optimizer = model_and_optimizer
        
        with pytest.raises(ValueError, match="Unknown scheduler type"):
            create_flow_scheduler('invalid_type', optimizer)


class TestSchedulerIntegration:
    """Integration tests for schedulers with different flow types."""
    
    def test_scheduler_with_mock_flow_training(self, model_and_optimizer):
        """Test scheduler integration with mock flow training loop."""
        model, optimizer = model_and_optimizer
        scheduler = AdaptiveFlowScheduler(optimizer, patience=3, verbose=True)
        
        # Simulate training loop
        initial_lr = optimizer.param_groups[0]['lr']
        
        for epoch in range(10):
            # Mock training step
            loss = 10.0 - epoch * 0.5 + np.random.normal(0, 0.1)
            log_likelihood = -loss
            gradient_norm = 1.0 / (epoch + 1)  # Decreasing gradients
            
            # Step scheduler
            scheduler.step({
                'log_likelihood': log_likelihood,
                'gradient_norm': gradient_norm,
                'jacobian_condition': 10.0 + epoch
            })
            
            # Mock optimizer step
            optimizer.step()
        
        # Verify scheduler tracked metrics
        assert len(scheduler.metric_history) > 0
        assert len(scheduler.gradient_history) > 0
    
    def test_scheduler_state_persistence(self, model_and_optimizer):
        """Test scheduler state persistence across steps."""
        model, optimizer = model_and_optimizer
        scheduler = LogLikelihoodScheduler(optimizer, patience=2)
        
        # Step multiple times
        scheduler.step(-10.0)
        scheduler.step(-10.1)
        
        # Check state persistence
        assert scheduler.best_log_likelihood == -10.0
        assert scheduler.epochs_without_improvement == 1
        assert len(scheduler.log_likelihood_history) == 2
    
    def test_scheduler_lr_bounds(self, model_and_optimizer):
        """Test scheduler respects learning rate bounds."""
        model, optimizer = model_and_optimizer
        scheduler = AdaptiveFlowScheduler(optimizer, min_lr=1e-6, factor=0.1)
        
        # Force many reductions
        for i in range(20):
            scheduler.step({'log_likelihood': -10.0 - i})
        
        # Verify LR doesn't go below minimum
        current_lr = optimizer.param_groups[0]['lr']
        assert current_lr >= 1e-6


if __name__ == '__main__':
    pytest.main([__file__])