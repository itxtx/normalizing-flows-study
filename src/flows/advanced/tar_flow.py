import torch
from ..flow.flow import Flow
from src.flows.advanced.causal_transformer import CausalTransformer

class TarFlow(Flow):
    """
    Transformer-based Autoregressive Flow (TarFlow).
    Replaces MLPs with causal transformers for better expressiveness.
    """
    
    def __init__(self, dim, hidden_dim=128, num_layers=4, num_heads=8, dropout=0.1):
        super().__init__()
        self.dim = dim
        self.conditioner = CausalTransformer(dim, hidden_dim, num_layers, num_heads, dropout)
        
    def inverse(self, x):
        """Fast inverse transformation (density evaluation)."""
        # Reshape for transformer: (batch, dim) -> (batch, dim, 1)
        x_reshaped = x.unsqueeze(-1)
        
        # Get transformer output
        transformer_output = self.conditioner(x_reshaped)
        
        # Reshape back: (batch, dim, 2) -> (batch, dim*2)
        params = transformer_output.squeeze(-1)
        mu, alpha = params.chunk(2, dim=1)
        
        alpha = torch.clamp(alpha, min=-3, max=3)
        log_scale = -alpha
        scale = torch.exp(torch.clamp(log_scale, min=-5, max=5))
        
        z = (x - mu) * scale
        log_det_jacobian = -torch.sum(alpha, dim=1)
        
        return z, log_det_jacobian
    
    def forward(self, z):
        """Slow forward transformation (sampling)."""
        batch_size = z.size(0)
        x = torch.zeros(batch_size, self.dim, device=z.device, dtype=z.dtype)
        log_det_jacobian = torch.zeros(batch_size, device=z.device, dtype=z.dtype)
        
        for i in range(self.dim):
            # Reshape for transformer
            x_reshaped = x.unsqueeze(-1)
            transformer_output = self.conditioner(x_reshaped)
            params = transformer_output.squeeze(-1)
            mu, alpha = params.chunk(2, dim=1)
            
            alpha = torch.clamp(alpha, min=-3, max=3)
            log_scale = alpha[:, i]
            scale = torch.exp(torch.clamp(log_scale, min=-5, max=5))
            
            x_new = x.clone()
            x_new[:, i] = z[:, i] * scale + mu[:, i]
            x = x_new
            
            log_det_jacobian += alpha[:, i]
        
        return x, log_det_jacobian
