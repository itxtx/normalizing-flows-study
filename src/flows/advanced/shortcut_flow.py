import torch
import torch.nn as nn
from src.flows.flow.flow import Flow

class ShortcutFlow(Flow):
    """
    Shortcut Flow that can condition on the number of sampling steps.
    Enables high-quality generation with very few steps.
    """
    
    def __init__(self, dim, hidden_dim=64, max_steps=100):
        super().__init__()
        self.dim = dim
        self.max_steps = max_steps
        
        # Step conditioning network
        self.step_net = nn.Sequential(
            nn.Linear(1, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, hidden_dim // 4)
        )
        
        # Main flow network
        self.flow_net = nn.Sequential(
            nn.Linear(dim + hidden_dim // 4, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, dim)
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for layer in self.flow_net:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight, gain=0.1)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)
    
    def forward(self, z, num_steps=None):
        """Forward pass with step conditioning."""
        batch_size = z.size(0)
        device = z.device
        
        if num_steps is None:
            num_steps = torch.randint(1, self.max_steps + 1, (batch_size,), device=device)
        
        # Normalize step count
        step_normalized = num_steps.float() / self.max_steps
        step_features = self.step_net(step_normalized.unsqueeze(1))
        
        # Apply flow with step conditioning
        net_input = torch.cat([z, step_features], dim=1)
        x = self.flow_net(net_input)
        
        # Approximate log-det (simplified)
        log_det_J = torch.zeros(batch_size, device=device)
        
        return x, log_det_J
    
    def inverse(self, x, num_steps=None):
        """Inverse pass with step conditioning."""
        batch_size = x.size(0)
        device = x.device
        
        if num_steps is None:
            num_steps = torch.randint(1, self.max_steps + 1, (batch_size,), device=device)
        
        # Normalize step count
        step_normalized = num_steps.float() / self.max_steps
        step_features = self.step_net(step_normalized.unsqueeze(1))
        
        # Apply inverse flow with step conditioning
        net_input = torch.cat([x, step_features], dim=1)
        z = self.flow_net(net_input)
        
        # Approximate log-det (simplified)
        log_det_J_inv = torch.zeros(batch_size, device=device)
        
        return z, log_det_J_inv
