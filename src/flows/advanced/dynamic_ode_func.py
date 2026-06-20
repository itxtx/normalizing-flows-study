import torch
import torch.nn as nn

class DynamicODEFunc(nn.Module):
    """Dynamic ODE function that can condition on network depth."""
    
    def __init__(self, dim, hidden_dim, max_depth=10):
        super().__init__()
        self.dim = dim
        self.max_depth = max_depth
        
        # Depth conditioning network
        self.depth_net = nn.Sequential(
            nn.Linear(1, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, hidden_dim // 4)
        )
        
        # Main network
        self.net = nn.Sequential(
            nn.Linear(dim + hidden_dim // 4, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, dim)
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
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
    
    def forward(self, t, augmented_state, depth_condition=None):
        z = augmented_state[:, :self.dim]
        log_det = augmented_state[:, self.dim:]
        
        # Process depth condition
        if depth_condition is None:
            depth_condition = torch.zeros(z.size(0), 1, device=z.device)
        
        depth_features = self.depth_net(depth_condition)
        
        # Concatenate input with depth features
        net_input = torch.cat([z, depth_features], dim=1)
        dz_dt = self.net(net_input)
        
        # Compute trace for log-determinant
        batch_size = z.size(0)
        trace_estimate = 0.0
        
        for _ in range(3):  # Hutchinson's trace estimator
            epsilon = torch.randn_like(z)
            z_copy = z.clone().detach().requires_grad_(True)
            depth_features_copy = depth_features.clone().detach()
            net_input_copy = torch.cat([z_copy, depth_features_copy], dim=1)
            f_z = self.net(net_input_copy)
            
            jvp = torch.autograd.grad(
                f_z.sum(), z_copy, create_graph=True, retain_graph=True
            )[0]
            
            trace_contribution = (jvp * epsilon).sum(dim=1, keepdim=True)
            trace_estimate += trace_contribution
        
        trace = trace_estimate / 3
        dlog_det_dt = trace
        
        return torch.cat([dz_dt, dlog_det_dt], dim=1)
