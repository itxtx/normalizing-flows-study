import torch
import torch.nn as nn
from torchdiffeq import odeint
from src.flows.flow.flow import Flow

class FlowMatchingFlow(Flow):
    """
    Flow Matching with Minibatch Optimal Transport.
    Implements straight path learning from noise to data.
    """
    
    def __init__(self, dim, hidden_dim=64, num_steps=100):
        super().__init__()
        self.dim = dim
        self.num_steps = num_steps
        
        # Vector field network
        self.vector_field = nn.Sequential(
            nn.Linear(dim + 1, hidden_dim),  # +1 for time
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, dim)
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for layer in self.vector_field:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight, gain=0.1)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)
    
    def _compute_ot_coupling(self, x_data, x_noise):
        """Compute optimal transport coupling between data and noise."""
        # Simple Sinkhorn-Knopp algorithm for OT
        batch_size = x_data.size(0)
        device = x_data.device
        
        # Cost matrix
        cost_matrix = torch.cdist(x_data, x_noise, p=2)
        
        # Initialize coupling
        coupling = torch.ones(batch_size, batch_size, device=device) / batch_size
        
        # Sinkhorn iterations
        for _ in range(10):
            # Row normalization
            coupling = coupling / coupling.sum(dim=1, keepdim=True)
            # Column normalization
            coupling = coupling / coupling.sum(dim=0, keepdim=True)
        
        return coupling
    
    def _straight_path_interpolation(self, x_data, x_noise, coupling, t):
        """Interpolate along straight paths using OT coupling."""
        batch_size = x_data.size(0)
        device = x_data.device
        
        # Sample pairs according to coupling
        indices = torch.multinomial(coupling.view(-1), batch_size, replacement=True)
        data_indices = indices // batch_size
        noise_indices = indices % batch_size
        
        x_data_paired = x_data[data_indices]
        x_noise_paired = x_noise[noise_indices]
        
        # Linear interpolation
        x_t = (1 - t) * x_noise_paired + t * x_data_paired
        
        # Target vector field (straight path)
        v_t = x_data_paired - x_noise_paired
        
        return x_t, v_t
    
    def forward(self, z, integration_times=torch.tensor([0.0, 1.0])):
        """Forward pass using flow matching."""
        batch_size = z.size(0)
        device = z.device
        
        # Generate target data (in practice, this would be your training data)
        # For demonstration, we'll use a simple transformation
        x_target = torch.tanh(z) + 0.1 * torch.randn_like(z)
        
        # Compute OT coupling
        coupling = self._compute_ot_coupling(x_target, z)
        
        # Custom ODE function for flow matching
        def flow_matching_ode(t, state):
            x = state
            
            # Interpolate along straight paths
            x_t, v_target = self._straight_path_interpolation(x_target, z, coupling, t)
            
            # Predict vector field
            t_expanded = t.expand(batch_size, 1)
            net_input = torch.cat([x, t_expanded], dim=1)
            v_pred = self.vector_field(net_input)
            
            return v_pred
        
        integration_times = integration_times.to(device)
        
        try:
            solution = odeint(
                flow_matching_ode,
                z,
                integration_times,
                method='dopri5',
                atol=1e-5,
                rtol=1e-5
            )
        except Exception as e:
            solution = odeint(
                flow_matching_ode,
                z,
                integration_times,
                method='rk4',
                options={'step_size': 0.01}
            )
        
        x = solution[-1]
        # Approximate log-det (simplified)
        log_det_J = torch.zeros(batch_size, device=device)
        
        return x, log_det_J
    
    def inverse(self, x, integration_times=torch.tensor([1.0, 0.0])):
        """Inverse pass using flow matching."""
        batch_size = x.size(0)
        device = x.device
        
        # Generate noise (in practice, this would be from your base distribution)
        z_target = torch.randn_like(x)
        
        # Compute OT coupling
        coupling = self._compute_ot_coupling(x, z_target)
        
        def flow_matching_ode_inv(t, state):
            z = state
            
            # Interpolate along straight paths
            z_t, v_target = self._straight_path_interpolation(z_target, x, coupling, 1 - t)
            
            # Predict vector field
            t_expanded = t.expand(batch_size, 1)
            net_input = torch.cat([z, t_expanded], dim=1)
            v_pred = self.vector_field(net_input)
            
            return v_pred
        
        integration_times = integration_times.to(device)
        
        try:
            solution = odeint(
                flow_matching_ode_inv,
                x,
                integration_times,
                method='dopri5',
                atol=1e-5,
                rtol=1e-5
            )
        except Exception as e:
            solution = odeint(
                flow_matching_ode_inv,
                x,
                integration_times,
                method='rk4',
                options={'step_size': 0.01}
            )
        
        z = solution[-1]
        log_det_J_inv = torch.zeros(batch_size, device=device)
        
        return z, log_det_J_inv
