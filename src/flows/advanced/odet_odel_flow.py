import torch
from torchdiffeq import odeint
from ..flow.flow import Flow
from src.flows.advanced.dynamic_ode_func import DynamicODEFunc

class ODEtODElFlow(Flow):
    """
    Dynamic Network Depth Flow using ODEt(ODEl) architecture.
    ODEt: Outer ODE for time integration (noise to data)
    ODEl: Inner ODE for depth integration (network layers)
    """
    
    def __init__(self, dim, hidden_dim=64, max_depth=10):
        super().__init__()
        self.dim = dim
        self.max_depth = max_depth
        self.ode_func = DynamicODEFunc(dim, hidden_dim, max_depth)
        
    def forward(self, z, integration_times=torch.tensor([0.0, 1.0]), target_depth=None):
        """
        Forward pass with dynamic depth conditioning.
        
        Args:
            z: Input noise
            integration_times: Time points for integration
            target_depth: Target network depth (None for adaptive)
        """
        batch_size = z.size(0)
        device = z.device
        
        log_det_init = torch.zeros(batch_size, 1, device=device)
        augmented_state = torch.cat([z, log_det_init], dim=1)
        
        if integration_times[0] == integration_times[1]:
            return z, torch.zeros(batch_size, device=device)
        
        integration_times = integration_times.to(device)
        
        # Create depth condition
        if target_depth is None:
            depth_condition = torch.rand(batch_size, 1, device=device) * self.max_depth
        else:
            depth_condition = torch.full((batch_size, 1), target_depth, device=device)
        
        # Custom ODE function that includes depth conditioning
        def ode_func_with_depth(t, state):
            return self.ode_func(t, state, depth_condition)
        
        try:
            solution = odeint(
                ode_func_with_depth,
                augmented_state,
                integration_times,
                method='dopri5',
                atol=1e-5,
                rtol=1e-5
            )
        except Exception as e:
            solution = odeint(
                ode_func_with_depth,
                augmented_state,
                integration_times,
                method='rk4',
                options={'step_size': 0.01}
            )
        
        final_augmented_state = solution[-1]
        x = final_augmented_state[:, :self.dim]
        log_det_J = final_augmented_state[:, self.dim:].squeeze(-1)
        
        return x, log_det_J
    
    def inverse(self, x, integration_times=torch.tensor([1.0, 0.0]), target_depth=None):
        """Inverse pass with dynamic depth conditioning."""
        batch_size = x.size(0)
        device = x.device
        
        log_det_init = torch.zeros(batch_size, 1, device=device)
        augmented_state = torch.cat([x, log_det_init], dim=1)
        
        if integration_times[0] == integration_times[1]:
            return x, torch.zeros(batch_size, device=device)
        
        integration_times = integration_times.to(device)
        
        # Create depth condition
        if target_depth is None:
            depth_condition = torch.rand(batch_size, 1, device=device) * self.max_depth
        else:
            depth_condition = torch.full((batch_size, 1), target_depth, device=device)
        
        def ode_func_with_depth(t, state):
            return self.ode_func(t, state, depth_condition)
        
        try:
            solution = odeint(
                ode_func_with_depth,
                augmented_state,
                integration_times,
                method='dopri5',
                atol=1e-5,
                rtol=1e-5
            )
        except Exception as e:
            solution = odeint(
                ode_func_with_depth,
                augmented_state,
                integration_times,
                method='rk4',
                options={'step_size': 0.01}
            )
        
        final_augmented_state = solution[-1]
        z = final_augmented_state[:, :self.dim]
        log_det_J_inv = final_augmented_state[:, self.dim:].squeeze(-1)
        
        return z, log_det_J_inv
