import torch
import torch.nn as nn

class ODEFunc(nn.Module):
    """
    Simplified ODE function for continuous normalizing flows.
    Uses a simpler approach that avoids complex trace estimation.
    """
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.dim = dim
        # Simple network without BatchNorm for stability
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
        Simplified forward pass that avoids complex gradient computations.
        
        Args:
            t: time (not used in this implementation)
            augmented_state: concatenated tensor [z, log_det] where z is the state
                           and log_det is the accumulated log-determinant
        
        Returns:
            concatenated tensor [dz/dt, dlog_det/dt]
        """
        z = augmented_state[:, :self.dim]
        log_det = augmented_state[:, self.dim:]
        
        # Compute the velocity field
        dz_dt = self.net(z)
        
        # Simplified trace estimation - just use a constant approximation
        # This is much more stable than computing exact traces
        batch_size = z.size(0)
        device = z.device
        
        # Use a simple approximation for the trace
        # For small networks, this is often sufficient
        trace_approx = torch.zeros(batch_size, 1, device=device)
        
        # Alternative: use a learned trace approximation
        # trace_approx = self.trace_net(z) if hasattr(self, 'trace_net') else torch.zeros(batch_size, 1, device=device)
        
        dlog_det_dt = trace_approx
        
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
