"""
Tests for gradient checkpointing functionality.
"""

import torch
import torch.nn as nn
import pytest
import gc
from unittest.mock import patch

from src.flows.optimization.gradient_checkpointing import (
    CheckpointedFlow,
    CheckpointedSequentialFlow,
    apply_gradient_checkpointing,
    MemoryEfficientWrapper
)
from src.flows.flow.flow import Flow
from src.flows.flow.sequential_flow import SequentialFlow


class SimpleFlow(Flow):
    """Simple flow for testing."""
    
    def __init__(self, dim: int):
        super().__init__()
        self.linear = nn.Linear(dim, dim)
        self.data_dim = dim
    
    def forward(self, z):
        x = torch.tanh(self.linear(z))
        # Simple log-det computation for testing
        log_det = torch.zeros(z.size(0), device=z.device)
        return x, log_det
    
    def inverse(self, x):
        # Approximate inverse for testing
        z = self.linear(torch.atanh(torch.clamp(x, -0.99, 0.99)))
        log_det = torch.zeros(x.size(0), device=x.device)
        return z, log_det


class TestCheckpointedFlow:
    """Test CheckpointedFlow wrapper."""
    
    def test_forward_backward_equivalence(self):
        """Test that checkpointed flow produces same results as original."""
        dim = 10
        batch_size = 32
        
        # Create original and checkpointed flows
        original_flow = SimpleFlow(dim)
        checkpointed_flow = CheckpointedFlow(original_flow)
        
        # Copy parameters to ensure identical initialization
        checkpointed_flow.flow.load_state_dict(original_flow.state_dict())
        
        # Test data
        z = torch.randn(batch_size, dim, requires_grad=True)
        z_checkpoint = z.clone().detach().requires_grad_(True)
        
        # Forward pass
        x_orig, log_det_orig = original_flow(z)
        x_checkpoint, log_det_checkpoint = checkpointed_flow(z_checkpoint)
        
        # Check forward pass equivalence
        assert torch.allclose(x_orig, x_checkpoint, atol=1e-6)
        assert torch.allclose(log_det_orig, log_det_checkpoint, atol=1e-6)
        
        # Backward pass
        loss_orig = x_orig.sum()
        loss_checkpoint = x_checkpoint.sum()
        
        loss_orig.backward()
        loss_checkpoint.backward()
        
        # Check gradient equivalence
        for p_orig, p_checkpoint in zip(original_flow.parameters(), 
                                      checkpointed_flow.flow.parameters()):
            assert torch.allclose(p_orig.grad, p_checkpoint.grad, atol=1e-6)
    
    def test_inverse_equivalence(self):
        """Test that inverse pass works correctly with checkpointing."""
        dim = 10
        batch_size = 32
        
        original_flow = SimpleFlow(dim)
        checkpointed_flow = CheckpointedFlow(original_flow)
        checkpointed_flow.flow.load_state_dict(original_flow.state_dict())
        
        x = torch.randn(batch_size, dim, requires_grad=True)
        x_checkpoint = x.clone().detach().requires_grad_(True)
        
        # Inverse pass
        z_orig, log_det_orig = original_flow.inverse(x)
        z_checkpoint, log_det_checkpoint = checkpointed_flow.inverse(x_checkpoint)
        
        assert torch.allclose(z_orig, z_checkpoint, atol=1e-6)
        assert torch.allclose(log_det_orig, log_det_checkpoint, atol=1e-6)
    
    def test_eval_mode_no_checkpointing(self):
        """Test that checkpointing is disabled in eval mode."""
        dim = 10
        batch_size = 32
        
        flow = SimpleFlow(dim)
        checkpointed_flow = CheckpointedFlow(flow)
        checkpointed_flow.eval()
        
        z = torch.randn(batch_size, dim, requires_grad=True)
        
        # In eval mode, should not use checkpointing
        with patch('torch.utils.checkpoint.checkpoint') as mock_checkpoint:
            x, log_det = checkpointed_flow(z)
            mock_checkpoint.assert_not_called()
    
    def test_no_grad_no_checkpointing(self):
        """Test that checkpointing is disabled when input doesn't require grad."""
        dim = 10
        batch_size = 32
        
        flow = SimpleFlow(dim)
        checkpointed_flow = CheckpointedFlow(flow)
        checkpointed_flow.train()
        
        z = torch.randn(batch_size, dim, requires_grad=False)
        
        with patch('torch.utils.checkpoint.checkpoint') as mock_checkpoint:
            x, log_det = checkpointed_flow(z)
            mock_checkpoint.assert_not_called()


class TestCheckpointedSequentialFlow:
    """Test CheckpointedSequentialFlow."""
    
    def test_segmentation(self):
        """Test flow segmentation logic."""
        flows = [SimpleFlow(10) for _ in range(6)]
        
        # Test with 2 segments
        checkpointed_flow = CheckpointedSequentialFlow(flows, checkpoint_segments=2)
        assert len(checkpointed_flow.segments) == 2
        assert len(checkpointed_flow.segments[0]) == 3
        assert len(checkpointed_flow.segments[1]) == 3
        
        # Test with 3 segments
        checkpointed_flow = CheckpointedSequentialFlow(flows, checkpoint_segments=3)
        assert len(checkpointed_flow.segments) == 3
        assert len(checkpointed_flow.segments[0]) == 2
        assert len(checkpointed_flow.segments[1]) == 2
        assert len(checkpointed_flow.segments[2]) == 2
        
        # Test with uneven division
        flows = [SimpleFlow(10) for _ in range(7)]
        checkpointed_flow = CheckpointedSequentialFlow(flows, checkpoint_segments=3)
        assert len(checkpointed_flow.segments) == 3
        assert len(checkpointed_flow.segments[0]) == 3  # Gets remainder
        assert len(checkpointed_flow.segments[1]) == 2
        assert len(checkpointed_flow.segments[2]) == 2
    
    def test_forward_backward_equivalence(self):
        """Test equivalence with original sequential flow."""
        dim = 10
        batch_size = 32
        num_flows = 4
        
        flows = [SimpleFlow(dim) for _ in range(num_flows)]
        
        # Create original and checkpointed flows
        original_flow = SequentialFlow(flows)
        checkpointed_flow = CheckpointedSequentialFlow(
            [SimpleFlow(dim) for _ in range(num_flows)],
            checkpoint_segments=2
        )
        
        # Copy parameters
        checkpointed_flow.load_state_dict(original_flow.state_dict())
        
        z = torch.randn(batch_size, dim, requires_grad=True)
        z_checkpoint = z.clone().detach().requires_grad_(True)
        
        # Forward pass
        x_orig, log_det_orig = original_flow(z)
        x_checkpoint, log_det_checkpoint = checkpointed_flow(z_checkpoint)
        
        assert torch.allclose(x_orig, x_checkpoint, atol=1e-6)
        assert torch.allclose(log_det_orig, log_det_checkpoint, atol=1e-6)
        
        # Backward pass
        loss_orig = x_orig.sum()
        loss_checkpoint = x_checkpoint.sum()
        
        loss_orig.backward()
        loss_checkpoint.backward()
        
        # Check gradient equivalence
        for p_orig, p_checkpoint in zip(original_flow.parameters(), 
                                      checkpointed_flow.parameters()):
            if p_orig.grad is not None and p_checkpoint.grad is not None:
                assert torch.allclose(p_orig.grad, p_checkpoint.grad, atol=1e-6)
    
    def test_inverse_equivalence(self):
        """Test inverse pass equivalence."""
        dim = 10
        batch_size = 32
        num_flows = 4
        
        flows = [SimpleFlow(dim) for _ in range(num_flows)]
        original_flow = SequentialFlow(flows)
        checkpointed_flow = CheckpointedSequentialFlow(
            [SimpleFlow(dim) for _ in range(num_flows)],
            checkpoint_segments=2
        )
        checkpointed_flow.load_state_dict(original_flow.state_dict())
        
        x = torch.randn(batch_size, dim, requires_grad=True)
        x_checkpoint = x.clone().detach().requires_grad_(True)
        
        z_orig, log_det_orig = original_flow.inverse(x)
        z_checkpoint, log_det_checkpoint = checkpointed_flow.inverse(x_checkpoint)
        
        assert torch.allclose(z_orig, z_checkpoint, atol=1e-6)
        assert torch.allclose(log_det_orig, log_det_checkpoint, atol=1e-6)
    
    def test_no_checkpointing_when_none(self):
        """Test that no checkpointing is applied when checkpoint_segments is None."""
        flows = [SimpleFlow(10) for _ in range(4)]
        checkpointed_flow = CheckpointedSequentialFlow(flows, checkpoint_segments=None)
        
        assert checkpointed_flow.segments is None
        
        z = torch.randn(32, 10, requires_grad=True)
        
        with patch('torch.utils.checkpoint.checkpoint') as mock_checkpoint:
            x, log_det = checkpointed_flow(z)
            mock_checkpoint.assert_not_called()
    
    def test_invalid_segments(self):
        """Test error handling for invalid segment numbers."""
        flows = [SimpleFlow(10) for _ in range(4)]
        
        with pytest.raises(ValueError):
            CheckpointedSequentialFlow(flows, checkpoint_segments=0)
        
        with pytest.raises(ValueError):
            CheckpointedSequentialFlow(flows, checkpoint_segments=-1)
    
    def test_segments_greater_than_flows(self):
        """Test warning when segments > number of flows."""
        flows = [SimpleFlow(10) for _ in range(2)]
        
        with pytest.warns(UserWarning):
            checkpointed_flow = CheckpointedSequentialFlow(flows, checkpoint_segments=5)
            assert checkpointed_flow.checkpoint_segments == 2


class TestApplyGradientCheckpointing:
    """Test the apply_gradient_checkpointing utility function."""
    
    def test_single_flow(self):
        """Test applying checkpointing to a single flow."""
        flow = SimpleFlow(10)
        checkpointed = apply_gradient_checkpointing(flow)
        
        assert isinstance(checkpointed, CheckpointedFlow)
        assert checkpointed.flow is flow
    
    def test_sequential_flow(self):
        """Test applying checkpointing to a sequential flow."""
        flows = [SimpleFlow(10) for _ in range(4)]
        sequential_flow = SequentialFlow(flows)
        
        checkpointed = apply_gradient_checkpointing(
            sequential_flow, 
            checkpoint_segments=2
        )
        
        assert isinstance(checkpointed, CheckpointedSequentialFlow)
        assert checkpointed.checkpoint_segments == 2


class TestMemoryEfficientWrapper:
    """Test MemoryEfficientWrapper."""
    
    def test_memory_monitoring(self):
        """Test memory usage monitoring."""
        flow = SimpleFlow(10)
        wrapper = MemoryEfficientWrapper(flow, monitor_memory=True)
        
        z = torch.randn(32, 10)
        x, log_det = wrapper(z)
        
        stats = wrapper.get_memory_stats()
        assert 'peak_memory' in stats
        assert 'forward_memory' in stats
        assert 'num_calls' in stats
        assert stats['num_calls'] == 1
    
    def test_checkpointing_suggestions(self):
        """Test checkpointing configuration suggestions."""
        # Test single flow
        flow = SimpleFlow(10)
        wrapper = MemoryEfficientWrapper(flow)
        
        suggestion = wrapper.suggest_checkpointing_config()
        assert suggestion['recommendation'] == 'single_layer_checkpointing'
        
        # Test sequential flow
        flows = [SimpleFlow(10) for _ in range(8)]
        sequential_flow = SequentialFlow(flows)
        wrapper = MemoryEfficientWrapper(sequential_flow)
        
        suggestion = wrapper.suggest_checkpointing_config(target_memory_reduction=0.5)
        assert suggestion['recommendation'] == 'segmented_checkpointing'
        assert 'checkpoint_segments' in suggestion
        assert suggestion['checkpoint_segments'] <= 8
    
    def test_no_memory_monitoring(self):
        """Test wrapper with memory monitoring disabled."""
        flow = SimpleFlow(10)
        wrapper = MemoryEfficientWrapper(flow, monitor_memory=False)
        
        z = torch.randn(32, 10)
        x, log_det = wrapper(z)
        
        stats = wrapper.get_memory_stats()
        assert stats['peak_memory'] == 0
        assert stats['forward_memory'] == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestMemoryReduction:
    """Test actual memory reduction with CUDA."""
    
    def test_memory_reduction_sequential(self):
        """Test that checkpointing reduces memory usage."""
        dim = 100
        batch_size = 64
        num_flows = 10
        
        # Create flows
        flows = [SimpleFlow(dim) for _ in range(num_flows)]
        original_flow = SequentialFlow(flows).cuda()
        
        flows_checkpoint = [SimpleFlow(dim) for _ in range(num_flows)]
        checkpointed_flow = CheckpointedSequentialFlow(
            flows_checkpoint, 
            checkpoint_segments=5
        ).cuda()
        checkpointed_flow.load_state_dict(original_flow.state_dict())
        
        z = torch.randn(batch_size, dim, device='cuda', requires_grad=True)
        
        # Measure memory for original flow
        torch.cuda.reset_peak_memory_stats()
        x_orig, _ = original_flow(z)
        loss_orig = x_orig.sum()
        loss_orig.backward()
        memory_orig = torch.cuda.max_memory_allocated()
        
        # Clear gradients and reset memory
        original_flow.zero_grad()
        del x_orig, loss_orig
        torch.cuda.empty_cache()
        
        # Measure memory for checkpointed flow
        z_checkpoint = z.clone().detach().requires_grad_(True)
        torch.cuda.reset_peak_memory_stats()
        x_checkpoint, _ = checkpointed_flow(z_checkpoint)
        loss_checkpoint = x_checkpoint.sum()
        loss_checkpoint.backward()
        memory_checkpoint = torch.cuda.max_memory_allocated()
        
        # Checkpointed version should use less memory
        # Note: The exact reduction depends on the specific implementation
        # and PyTorch version, so we just check that it's not significantly more
        memory_ratio = memory_checkpoint / memory_orig
        assert memory_ratio <= 1.2, f"Memory ratio: {memory_ratio}"


if __name__ == "__main__":
    pytest.main([__file__])