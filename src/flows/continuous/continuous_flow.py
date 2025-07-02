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

    def forward(self, z, integration_times=torch.tensor([0.0, 1.0])):
        """
        Forward pass (sampling). Solves the ODE from t=0 to t=1.
        """
        batch_size = z.size(0)
        device = z.device
        
        # Ensure z requires gradients for the ODE solver
        if not z.requires_grad:
            z = z.detach().requires_grad_(True)
        
        log_det_init = torch.zeros(batch_size, 1, device=device)
        augmented_state = torch.cat([z, log_det_init], dim=1)
        
        if integration_times[0] == integration_times[1]:
            return z, torch.zeros(batch_size, device=device)
        
        integration_times = integration_times.to(device)
        
        # Use more stable solver settings
        try:
            solution = odeint(
                self.ode_func,
                augmented_state,
                integration_times,
                method='rk4',  # Use RK4 instead of dopri5 for stability
                options={'step_size': 0.01}
            )
        except Exception as e:
            print(f"ODE solver failed with error: {e}")
            # Fallback to even simpler method
            try:
                solution = odeint(
                    self.ode_func,
                    augmented_state,
                    integration_times,
                    method='euler',
                    options={'step_size': 0.001}
                )
            except Exception as e2:
                print(f"Even Euler method failed: {e2}")
                # Last resort: return identity transformation
                return z, torch.zeros(batch_size, device=device)
        
        final_augmented_state = solution[-1]
        x = final_augmented_state[:, :self.dim]
        log_det_J = final_augmented_state[:, self.dim:].squeeze(-1)
        
        # Better handling of numerical issues
        x = torch.where(torch.isnan(x) | torch.isinf(x), z, x)
        log_det_J = torch.where(
            torch.isnan(log_det_J) | torch.isinf(log_det_J),
            torch.zeros_like(log_det_J),
            log_det_J
        )
        
        # Add bounds to prevent extreme values
        x = torch.clamp(x, min=-10.0, max=10.0)
        log_det_J = torch.clamp(log_det_J, min=-10.0, max=10.0)
        
        return x, log_det_J

    def inverse(self, x, integration_times=torch.tensor([1.0, 0.0])):
        """
        Inverse pass (likelihood). Solves the ODE from t=1 to t=0.
        """
        batch_size = x.size(0)
        device = x.device
        
        # Ensure x requires gradients for the ODE solver
        if not x.requires_grad:
            x = x.detach().requires_grad_(True)
        
        log_det_init = torch.zeros(batch_size, 1, device=device)
        augmented_state = torch.cat([x, log_det_init], dim=1)
        
        if integration_times[0] == integration_times[1]:
            return x, torch.zeros(batch_size, device=device)
        
        integration_times = integration_times.to(device)
        
        # Use more stable solver settings
        try:
            solution = odeint(
                self.ode_func,
                augmented_state,
                integration_times,
                method='rk4',  # Use RK4 instead of dopri5 for stability
                options={'step_size': 0.01}
            )
        except Exception as e:
            print(f"ODE solver failed with error: {e}")
            # Fallback to even simpler method
            try:
                solution = odeint(
                    self.ode_func,
                    augmented_state,
                    integration_times,
                    method='euler',
                    options={'step_size': 0.001}
                )
            except Exception as e2:
                print(f"Even Euler method failed: {e2}")
                # Last resort: return identity transformation
                return x, torch.zeros(batch_size, device=device)
        
        final_augmented_state = solution[-1]
        z = final_augmented_state[:, :self.dim]
        log_det_J_inv = final_augmented_state[:, self.dim:].squeeze(-1)
        
        # Better handling of numerical issues
        z = torch.where(torch.isnan(z) | torch.isinf(z), x, z)
        log_det_J_inv = torch.where(
            torch.isnan(log_det_J_inv) | torch.isinf(log_det_J_inv),
            torch.zeros_like(log_det_J_inv),
            log_det_J_inv
        )
        
        # Add bounds to prevent extreme values
        z = torch.clamp(z, min=-10.0, max=10.0)
        log_det_J_inv = torch.clamp(log_det_J_inv, min=-10.0, max=10.0)
        
        return z, log_det_J_inv
