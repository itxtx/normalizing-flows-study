"""
Tests for mixed precision training functionality.
"""

import torch
import torch.nn as nn
import pytest
from unittest.mock import patch, MagicMock

from src.flows.optimization.mixed_precision import (
    MixedPrecisionFlow,
    MixedPrecisionTrainer,
    check_mixed_precision_compatibility,
    apply_mixed_precision
)
from src.flows.flow.flow import Flow


class SimpleFlow(Flow):
    """Simple flow for testing."""
    
    def __init__(self, dim: int):
        super().__init__()
        self.linear = nn.Linear(dim, dim)
        self.data_dim = dim
    
    def forward(self, z):
        x = torch.tanh(self.linear(z))
        log_det = torch.zeros(z.size(0), device=z.device)
        return x, log_det
    
    def inverse(self, x):
        z = self.linear(torch.atanh(torch.clamp(x, -0.99, 0.99)))
        log_det = torch.zeros(x.size(0), device=x.device)
        return z, log_det


class TestMixedPrecisionFlow:
    """Test MixedPrecisionFlow wrapper."""
    
    def test_initialization_without_cuda(self):
        """Test initialization when CUDA is not available."""
        flow = SimpleFlow(10)
        
        with patch('torch.cuda.is_available', return_value=False):
            with pytest.warns(UserWarning, match="Mixed precision training requires CUDA"):
                mp_flow = MixedPrecisionFlow(flow, enabled=True)
                assert not mp_flow.enabled
                assert mp_flow.scaler is None
    
    def test_initialization_with_cuda(self):
        """Test initialization when CUDA is available."""
        flow = SimpleFlow(10)
        
        with patch('torch.cuda.is_available', return_value=True):
            mp_flow = MixedPrecisionFlow(flow, enabled=True)
            assert mp_flow.enabled
            assert mp_flow.scaler is not None
    
    def test_invalid_autocast_dtype(self):
        """Test error handling for invalid autocast dtype."""
        flow = SimpleFlow(10)
        
        with patch('torch.cuda.is_available', return_value=True):
            with pytest.raises(ValueError, match="autocast_dtype must be"):
                MixedPrecisionFlow(flow, autocast_dtype=torch.float32)
    
    def test_forward_without_mixed_precision(self):
        """Test forward pass when mixed precision is disabled."""
        flow = SimpleFlow(10)
        mp_flow = MixedPrecisionFlow(flow, enabled=False)
        
        z = torch.randn(32, 10)
        x, log_det = mp_flow.forward(z)
        
        assert x.shape == z.shape
        assert log_det.shape == (32,)
        assert mp_flow.stats['forward_calls'] == 1
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_forward_with_mixed_precision(self):
        """Test forward pass with mixed precision enabled."""
        flow = SimpleFlow(10).cuda()
        mp_flow = MixedPrecisionFlow(flow, enabled=True).cuda()
        
        z = torch.randn(32, 10, device='cuda')
        
        with patch('torch.cuda.amp.autocast') as mock_autocast:
            mock_autocast.return_value.__enter__ = MagicMock()
            mock_autocast.return_value.__exit__ = MagicMock()
            
            x, log_det = mp_flow.forward(z)
            
            # Check that autocast was called
            mock_autocast.assert_called_once_with(dtype=torch.float16)
    
    def test_inverse_equivalence(self):
        """Test that inverse pass works correctly."""
        flow = SimpleFlow(10)
        mp_flow = MixedPrecisionFlow(flow, enabled=False)
        
        x = torch.randn(32, 10)
        
        # Compare with original flow
        z_orig, log_det_orig = flow.inverse(x)
        z_mp, log_det_mp = mp_flow.inverse(x)
        
        assert torch.allclose(z_orig, z_mp)
        assert torch.allclose(log_det_orig, log_det_mp)
    
    def test_loss_scaling(self):
        """Test loss scaling functionality."""
        flow = SimpleFlow(10)
        
        # Test without mixed precision
        mp_flow_disabled = MixedPrecisionFlow(flow, enabled=False)
        loss = torch.tensor(1.0)
        scaled_loss = mp_flow_disabled.scale_loss(loss)
        assert torch.equal(loss, scaled_loss)
        
        # Test with mixed precision
        with patch('torch.cuda.is_available', return_value=True):
            mp_flow_enabled = MixedPrecisionFlow(flow, enabled=True)
            mock_scaler = MagicMock()
            mock_scaler.scale.return_value = loss * 2
            mp_flow_enabled.scaler = mock_scaler
            
            scaled_loss = mp_flow_enabled.scale_loss(loss)
            mock_scaler.scale.assert_called_once_with(loss)
    
    def test_optimizer_step(self):
        """Test optimizer stepping with gradient scaling."""
        flow = SimpleFlow(10)
        optimizer = torch.optim.Adam(flow.parameters())
        
        # Test without mixed precision
        mp_flow_disabled = MixedPrecisionFlow(flow, enabled=False)
        success = mp_flow_disabled.step_optimizer(optimizer)
        assert success
        
        # Test with mixed precision
        with patch('torch.cuda.is_available', return_value=True):
            mp_flow_enabled = MixedPrecisionFlow(flow, enabled=True)
            mock_scaler = MagicMock()
            mock_scaler.get_scale.side_effect = [1000.0, 1000.0]  # No overflow
            mp_flow_enabled.scaler = mock_scaler
            
            success = mp_flow_enabled.step_optimizer(optimizer)
            assert success
            mock_scaler.step.assert_called_once_with(optimizer)
            mock_scaler.update.assert_called_once()
    
    def test_overflow_detection(self):
        """Test overflow detection in optimizer step."""
        flow = SimpleFlow(10)
        optimizer = torch.optim.Adam(flow.parameters())
        
        with patch('torch.cuda.is_available', return_value=True):
            mp_flow = MixedPrecisionFlow(flow, enabled=True)
            mock_scaler = MagicMock()
            mock_scaler.get_scale.side_effect = [1000.0, 500.0]  # Scale decreased = overflow
            mp_flow.scaler = mock_scaler
            
            success = mp_flow.step_optimizer(optimizer)
            assert not success
            assert mp_flow.stats['overflow_count'] == 1
    
    def test_gradient_unscaling(self):
        """Test gradient unscaling."""
        flow = SimpleFlow(10)
        optimizer = torch.optim.Adam(flow.parameters())
        
        with patch('torch.cuda.is_available', return_value=True):
            mp_flow = MixedPrecisionFlow(flow, enabled=True)
            mock_scaler = MagicMock()
            mp_flow.scaler = mock_scaler
            
            mp_flow.unscale_gradients(optimizer)
            mock_scaler.unscale_.assert_called_once_with(optimizer)
    
    def test_stats_tracking(self):
        """Test statistics tracking."""
        flow = SimpleFlow(10)
        
        with patch('torch.cuda.is_available', return_value=True):
            mp_flow = MixedPrecisionFlow(flow, enabled=True)
            
            # Initial stats
            stats = mp_flow.get_stats()
            assert stats['forward_calls'] == 0
            assert stats['overflow_count'] == 0
            assert stats['overflow_rate'] == 0.0
            
            # After forward pass
            z = torch.randn(32, 10)
            mp_flow.forward(z)
            
            stats = mp_flow.get_stats()
            assert stats['forward_calls'] == 1
    
    def test_state_dict_save_load(self):
        """Test state dict saving and loading."""
        flow = SimpleFlow(10)
        
        with patch('torch.cuda.is_available', return_value=True):
            mp_flow = MixedPrecisionFlow(flow, enabled=True)
            
            # Modify some stats
            mp_flow.stats['forward_calls'] = 5
            mp_flow.stats['overflow_count'] = 1
            
            # Save state dict
            state_dict = mp_flow.state_dict()
            assert 'stats' in state_dict
            assert 'scaler' in state_dict
            
            # Create new flow and load state
            new_flow = SimpleFlow(10)
            new_mp_flow = MixedPrecisionFlow(new_flow, enabled=True)
            new_mp_flow.load_state_dict(state_dict)
            
            assert new_mp_flow.stats['forward_calls'] == 5
            assert new_mp_flow.stats['overflow_count'] == 1


class TestMixedPrecisionTrainer:
    """Test MixedPrecisionTrainer."""
    
    def test_initialization(self):
        """Test trainer initialization."""
        flow = SimpleFlow(10)
        optimizer = torch.optim.Adam(flow.parameters())
        
        trainer = MixedPrecisionTrainer(flow, optimizer, enabled=False)
        assert isinstance(trainer.mp_flow, MixedPrecisionFlow)
        assert trainer.optimizer is optimizer
        assert trainer.max_grad_norm == 1.0
    
    def test_training_step(self):
        """Test training step execution."""
        flow = SimpleFlow(10)
        optimizer = torch.optim.Adam(flow.parameters())
        trainer = MixedPrecisionTrainer(flow, optimizer, enabled=False)
        
        # Create mock base distribution
        base_dist = torch.distributions.Normal(0, 1)
        batch = torch.randn(32, 10)
        
        # Perform training step
        result = trainer.training_step(batch, base_dist)
        
        assert 'loss' in result
        assert 'scaled_loss' in result
        assert 'grad_norm' in result
        assert 'step_successful' in result
        assert 'scale' in result
        
        assert trainer.training_stats['total_steps'] == 1
        assert trainer.training_stats['successful_steps'] == 1
    
    def test_gradient_clipping(self):
        """Test gradient clipping functionality."""
        flow = SimpleFlow(10)
        optimizer = torch.optim.Adam(flow.parameters())
        trainer = MixedPrecisionTrainer(flow, optimizer, enabled=False, max_grad_norm=0.1)
        
        base_dist = torch.distributions.Normal(0, 1)
        batch = torch.randn(32, 10)
        
        # Create large gradients by using a large loss
        with patch('torch.nn.utils.clip_grad_norm_', return_value=torch.tensor(2.0)) as mock_clip:
            result = trainer.training_step(batch, base_dist)
            
            mock_clip.assert_called_once()
            assert trainer.training_stats['gradient_clips'] == 1
    
    def test_training_stats(self):
        """Test training statistics collection."""
        flow = SimpleFlow(10)
        optimizer = torch.optim.Adam(flow.parameters())
        trainer = MixedPrecisionTrainer(flow, optimizer, enabled=False)
        
        base_dist = torch.distributions.Normal(0, 1)
        batch = torch.randn(32, 10)
        
        # Perform multiple training steps
        for _ in range(5):
            trainer.training_step(batch, base_dist)
        
        stats = trainer.get_training_stats()
        assert stats['total_steps'] == 5
        assert stats['successful_steps'] == 5
        assert stats['success_rate'] == 1.0
        assert stats['overflow_rate'] == 0.0
    
    def test_state_dict_save_load(self):
        """Test trainer state dict functionality."""
        flow = SimpleFlow(10)
        optimizer = torch.optim.Adam(flow.parameters())
        trainer = MixedPrecisionTrainer(flow, optimizer, enabled=False)
        
        # Perform some training steps
        base_dist = torch.distributions.Normal(0, 1)
        batch = torch.randn(32, 10)
        trainer.training_step(batch, base_dist)
        
        # Save state
        state_dict = trainer.state_dict()
        assert 'mp_flow' in state_dict
        assert 'optimizer' in state_dict
        assert 'training_stats' in state_dict
        
        # Create new trainer and load state
        new_flow = SimpleFlow(10)
        new_optimizer = torch.optim.Adam(new_flow.parameters())
        new_trainer = MixedPrecisionTrainer(new_flow, new_optimizer, enabled=False)
        
        new_trainer.load_state_dict(state_dict)
        assert new_trainer.training_stats['total_steps'] == 1


class TestCompatibilityChecking:
    """Test mixed precision compatibility checking."""
    
    def test_compatibility_without_cuda(self):
        """Test compatibility check when CUDA is not available."""
        flow = SimpleFlow(10)
        
        with patch('torch.cuda.is_available', return_value=False):
            compat = check_mixed_precision_compatibility(flow)
            
            assert not compat['compatible']
            assert any('CUDA is not available' in w for w in compat['warnings'])
    
    def test_compatibility_with_cuda(self):
        """Test compatibility check when CUDA is available."""
        flow = SimpleFlow(10)
        
        with patch('torch.cuda.is_available', return_value=True):
            compat = check_mixed_precision_compatibility(flow)
            
            assert compat['compatible']
            assert 'compatible' in compat
            assert 'warnings' in compat
            assert 'recommendations' in compat
    
    def test_problematic_operations_detection(self):
        """Test detection of operations that may be problematic with mixed precision."""
        class FlowWithBatchNorm(Flow):
            def __init__(self):
                super().__init__()
                self.bn = nn.BatchNorm1d(10)
                self.linear = nn.Linear(10, 10)
            
            def forward(self, z):
                return z, torch.zeros(z.size(0))
            
            def inverse(self, x):
                return x, torch.zeros(x.size(0))
        
        flow = FlowWithBatchNorm()
        
        with patch('torch.cuda.is_available', return_value=True):
            compat = check_mixed_precision_compatibility(flow)
            
            assert compat['compatible']
            assert len(compat['warnings']) > 0
            assert any('batchnorm' in w.lower() for w in compat['warnings'])
    
    def test_small_model_recommendation(self):
        """Test recommendation for small models."""
        class TinyFlow(Flow):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(2, 2)  # Very small model
            
            def forward(self, z):
                return z, torch.zeros(z.size(0))
            
            def inverse(self, x):
                return x, torch.zeros(x.size(0))
        
        flow = TinyFlow()
        
        with patch('torch.cuda.is_available', return_value=True):
            compat = check_mixed_precision_compatibility(flow)
            
            assert compat['compatible']
            assert any('Mixed precision benefits may be limited' in r 
                      for r in compat['recommendations'])


class TestApplyMixedPrecision:
    """Test apply_mixed_precision utility function."""
    
    def test_apply_with_compatibility_check(self):
        """Test applying mixed precision with compatibility checking."""
        flow = SimpleFlow(10)
        
        with patch('torch.cuda.is_available', return_value=True):
            mp_flow = apply_mixed_precision(flow, enabled=True, check_compatibility=True)
            
            assert isinstance(mp_flow, MixedPrecisionFlow)
            assert mp_flow.flow is flow
    
    def test_apply_without_compatibility_check(self):
        """Test applying mixed precision without compatibility checking."""
        flow = SimpleFlow(10)
        
        with patch('torch.cuda.is_available', return_value=True):
            mp_flow = apply_mixed_precision(flow, enabled=True, check_compatibility=False)
            
            assert isinstance(mp_flow, MixedPrecisionFlow)
    
    def test_apply_with_incompatible_flow(self):
        """Test applying mixed precision to incompatible flow."""
        flow = SimpleFlow(10)
        
        with patch('torch.cuda.is_available', return_value=False):
            with pytest.warns(UserWarning):
                mp_flow = apply_mixed_precision(flow, enabled=True, check_compatibility=True)
                
                assert isinstance(mp_flow, MixedPrecisionFlow)
                assert not mp_flow.enabled


if __name__ == "__main__":
    pytest.main([__file__])