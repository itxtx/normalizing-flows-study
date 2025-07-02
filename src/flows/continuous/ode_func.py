import torch
import torch.nn as nn

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
            nn.BatchNorm1d(hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
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
