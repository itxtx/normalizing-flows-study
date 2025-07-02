import torch
from src.flows.flow.flow import Flow

class PaddingFlow(Flow):
    """
    Flow that handles variable dimensions by adding noise to padded dimensions.
    """
    
    def __init__(self, base_flow, max_dim, padding_noise_std=0.1):
        super().__init__()
        self.base_flow = base_flow
        self.max_dim = max_dim
        self.padding_noise_std = padding_noise_std
        
    def _add_padding_noise(self, x, actual_dims):
        """Add noise to padded dimensions."""
        batch_size = x.size(0)
        device = x.device
        
        # Create noise for all dimensions
        noise = torch.randn_like(x) * self.padding_noise_std
        
        # Zero out noise for actual data dimensions
        for i, dim in enumerate(actual_dims):
            if dim < self.max_dim:
                noise[i, dim:] = 0
        
        return x + noise
    
    def _create_padding_mask(self, actual_dims):
        """Create mask for padded dimensions."""
        batch_size = len(actual_dims)
        device = actual_dims[0].device if hasattr(actual_dims[0], 'device') else 'cpu'
        
        mask = torch.zeros(batch_size, self.max_dim, device=device)
        for i, dim in enumerate(actual_dims):
            mask[i, :dim] = 1
        
        return mask
    
    def inverse(self, x, actual_dims=None):
        """Inverse transformation with padding noise."""
        if actual_dims is None:
            actual_dims = [x.size(1)] * x.size(0)
        
        # Add noise to padded dimensions
        x_noisy = self._add_padding_noise(x, actual_dims)
        
        # Apply base flow
        z, log_det = self.base_flow.inverse(x_noisy)
        
        # Create padding mask
        padding_mask = self._create_padding_mask(actual_dims)
        
        # Zero out log-det for padded dimensions
        log_det = log_det * padding_mask.sum(dim=1) / self.max_dim
        
        return z, log_det
    
    def forward(self, z, actual_dims=None):
        """Forward transformation with padding handling."""
        if actual_dims is None:
            actual_dims = [self.max_dim] * z.size(0)
        
        # Apply base flow
        x, log_det = self.base_flow.forward(z)
        
        # Create padding mask
        padding_mask = self._create_padding_mask(actual_dims)
        
        # Zero out padded dimensions
        x = x * padding_mask
        
        # Adjust log-det for actual dimensions
        log_det = log_det * padding_mask.sum(dim=1) / self.max_dim
        
        return x, log_det
