import torch
from torchdiffeq import odeint
from ..flow.flow import Flow
from src.flows.continuous.ode_func import ODEFunc

class ContinuousFlow(Flow):
    """
    Continuous Normalizing Flow model with proper log-determinant calculation.
    """
    def __init__(self, dim, hidden_dim=64):
        super().__init__()
        self.data_dim = dim
        self.dim = dim
        self.ode_func = ODEFunc(dim, hidden_dim)

    def _integrate(self, state, integration_times):
        batch_size = state.size(0)
        if not state.requires_grad:
            state = state.detach().requires_grad_(True)

        log_det_init = state.new_zeros(batch_size, 1)
        augmented_state = torch.cat([state, log_det_init], dim=1)
        integration_times = torch.as_tensor(
            integration_times, device=state.device, dtype=state.dtype
        )
        if integration_times.numel() < 2:
            raise ValueError("integration_times must contain at least two values")
        if integration_times[0] == integration_times[-1]:
            return state, state.new_zeros(batch_size)

        trace_noise = torch.randn_like(state) if self.dim > 2 else None

        def dynamics(t, augmented):
            return self.ode_func(t, augmented, trace_noise=trace_noise)

        solution = odeint(
            dynamics,
            augmented_state,
            integration_times,
            method='rk4',
            options={'step_size': 0.01},
        )
        final_state = solution[-1]
        return final_state[:, :self.dim], final_state[:, self.dim:].squeeze(-1)

    def forward(self, z, integration_times=None):
        """
        Forward pass (sampling). Solves the ODE from t=0 to t=1.
        """
        times = (0.0, 1.0) if integration_times is None else integration_times
        return self._integrate(z, times)

    def inverse(self, x, integration_times=None):
        """
        Inverse pass (likelihood). Solves the ODE from t=1 to t=0.
        """
        times = (1.0, 0.0) if integration_times is None else integration_times
        return self._integrate(x, times)
