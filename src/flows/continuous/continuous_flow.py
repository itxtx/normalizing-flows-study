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
        self.dim = dim
        self.ode_func = ODEFunc(dim, hidden_dim)

    def forward(self, z, integration_times=torch.tensor([0.0, 1.0])):
        """
        Forward pass (sampling). Solves the ODE from t=0 to t=1.
        """
        batch_size = z.size(0)
        device = z.device
        
        log_det_init = torch.zeros(batch_size, 1, device=device)
        augmented_state = torch.cat([z, log_det_init], dim=1)
        
        if integration_times[0] == integration_times[1]:
            return z, torch.zeros(batch_size, device=device)
        
        integration_times = integration_times.to(device)
        
        try:
            solution = odeint(
                self.ode_func,
                augmented_state,
                integration_times,
                method='dopri5',
                atol=1e-5,
                rtol=1e-5
            )
        except Exception as e:
            solution = odeint(
                self.ode_func,
                augmented_state,
                integration_times,
                method='rk4',
                options={'step_size': 0.01}
            )
        
        final_augmented_state = solution[-1]
        x = final_augmented_state[:, :self.dim]
        log_det_J = final_augmented_state[:, self.dim:].squeeze(-1)
        
        x = torch.where(torch.isnan(x) | torch.isinf(x), z, x)
        log_det_J = torch.where(
            torch.isnan(log_det_J) | torch.isinf(log_det_J),
            torch.zeros_like(log_det_J),
            log_det_J
        )
        
        return x, log_det_J

    def inverse(self, x, integration_times=torch.tensor([1.0, 0.0])):
        """
        Inverse pass (likelihood). Solves the ODE from t=1 to t=0.
        """
        batch_size = x.size(0)
        device = x.device
        
        log_det_init = torch.zeros(batch_size, 1, device=device)
        augmented_state = torch.cat([x, log_det_init], dim=1)
        
        if integration_times[0] == integration_times[1]:
            return x, torch.zeros(batch_size, device=device)
        
        integration_times = integration_times.to(device)
        
        try:
            solution = odeint(
                self.ode_func,
                augmented_state,
                integration_times,
                method='dopri5',
                atol=1e-5,
                rtol=1e-5
            )
        except Exception as e:
            solution = odeint(
                self.ode_func,
                augmented_state,
                integration_times,
                method='rk4',
                options={'step_size': 0.01}
            )
        
        final_augmented_state = solution[-1]
        z = final_augmented_state[:, :self.dim]
        log_det_J_inv = final_augmented_state[:, self.dim:].squeeze(-1)
        
        z = torch.where(torch.isnan(z) | torch.isinf(z), x, z)
        log_det_J_inv = torch.where(
            torch.isnan(log_det_J_inv) | torch.isinf(log_det_J_inv),
            torch.zeros_like(log_det_J_inv),
            log_det_J_inv
        )
        
        return z, log_det_J_inv
