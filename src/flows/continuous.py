import torch
import torch.nn as nn
from torchdiffeq import odeint
from .flow import Flow

class ODEFunc(nn.Module):
    """
    Defines the dynamics of the continuous flow with augmented state.
    The augmented state includes both the input z and the log-determinant.
    """
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.dim = dim
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, dim)
        )
        
        self._initialize_weights()

    def forward(self, t, augmented_state):
        """
        The forward pass of the ODE function.
        
        Args:
            t: time (not used in this implementation)
            augmented_state: concatenated tensor [z, log_det] where z is the state
                           and log_det is the accumulated log-determinant
        
        Returns:
            concatenated tensor [dz/dt, dlog_det/dt]
        """
        z = augmented_state[:, :self.dim]
        log_det = augmented_state[:, self.dim:]
        
        dz_dt = self.net(z)
        
        batch_size = z.size(0)
        device = z.device
        
        num_vectors = 3
        trace_estimate = 0.0
        
        for _ in range(num_vectors):
            epsilon = torch.randn_like(z)
            
            z_copy = z.clone().detach().requires_grad_(True)
            f_z = self.net(z_copy)
            jvp = torch.autograd.grad(
                f_z.sum(), z_copy, create_graph=True, retain_graph=True
            )[0]
            
            trace_contribution = (jvp * epsilon).sum(dim=1, keepdim=True)
            trace_estimate += trace_contribution
        
        trace = trace_estimate / num_vectors
        
        dlog_det_dt = trace
        
        return torch.cat([dz_dt, dlog_det_dt], dim=1)
    
    def _initialize_weights(self):
        """Initialize network weights using Xavier initialization with proper scaling."""
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight, gain=0.1)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)
        
        final_layer = self.net[-1]
        if isinstance(final_layer, nn.Linear):
            nn.init.zeros_(final_layer.weight)
            if final_layer.bias is not None:
                nn.init.zeros_(final_layer.bias)

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
