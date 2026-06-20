import torch
from ..flow.flow import Flow
from .made import MADE

class InverseAutoregressiveFlow(Flow):
    """
    Inverse Autoregressive Flow (IAF). This flow has the same expressiveness as
    MAF but with a reversed computational trade-off.
    The forward transformation (sampling) is fast and parallel.
    The inverse transformation (density evaluation) is slow and sequential.
    """
    def __init__(self, dim, hidden_dim=64):
        super().__init__()
        self.data_dim = dim
        self.dim = dim
        self.conditioner = MADE(dim, hidden_dim, 2)
        
        # Initialize the conditioner to produce near-zero outputs initially
        self._initialize_conditioner()

    def _initialize_conditioner(self):
        """Initialize the conditioner to produce near-identity transformation."""
        # Initialize the final layer to produce small values
        final_layer = self.conditioner.net[-1]
        if hasattr(final_layer, 'weight'):
            torch.nn.init.normal_(final_layer.weight, mean=0.0, std=0.01)
        if hasattr(final_layer, 'bias'):
            torch.nn.init.zeros_(final_layer.bias)

    def forward(self, z):
        """
        Computes x = f(z). This direction is fast and parallel.
        x_i = z_i * exp(alpha_i) + mu_i, where mu_i and alpha_i are functions
        of z_1, ..., z_{i-1}.
        """
        params = self.conditioner(z)
        mu, alpha = params.chunk(2, dim=1)
        
        # More aggressive clamping for numerical stability
        alpha = torch.clamp(alpha, min=-2, max=2)
        mu = torch.clamp(mu, min=-10, max=10)
        
        # Use more conservative scale clamping
        scale = torch.exp(torch.clamp(alpha, min=-3, max=3))
        
        # Apply transformation with additional safety checks
        x = z * scale + mu
        
        # Compute log determinant
        log_det_jacobian = torch.sum(alpha, dim=1)
        
        # Comprehensive numerical stability checks
        x = torch.where(torch.isnan(x) | torch.isinf(x), z, x)
        log_det_jacobian = torch.where(
            torch.isnan(log_det_jacobian) | torch.isinf(log_det_jacobian),
            torch.zeros_like(log_det_jacobian),
            log_det_jacobian
        )
        
        # Clamp log determinant to prevent extreme values
        log_det_jacobian = torch.clamp(log_det_jacobian, min=-50, max=50)
        
        return x, log_det_jacobian

    def inverse(self, x):
        """
        Computes z = g(x). This is slow and sequential.
        z_i = (x_i - mu_i) * exp(-alpha_i)
        """
        batch_size = x.size(0)
        z = torch.zeros(batch_size, self.dim, device=x.device, dtype=x.dtype)
        log_det_jacobian = torch.zeros(batch_size, device=x.device, dtype=x.dtype)

        for i in range(self.dim):
            params = self.conditioner(z)
            mu, alpha = params.chunk(2, dim=1)
            
            # More aggressive clamping for numerical stability
            alpha = torch.clamp(alpha, min=-2, max=2)
            mu = torch.clamp(mu, min=-10, max=10)
            
            # Use more conservative scale clamping
            scale = torch.exp(torch.clamp(-alpha[:, i], min=-3, max=3))
            
            # Apply inverse transformation with safety checks
            z_new = z.clone()
            z_new[:, i] = (x[:, i] - mu[:, i]) * scale
            z = z_new
            
            log_det_jacobian -= alpha[:, i]

        # Comprehensive numerical stability checks
        z = torch.where(torch.isnan(z) | torch.isinf(z), x, z)
        log_det_jacobian = torch.where(
            torch.isnan(log_det_jacobian) | torch.isinf(log_det_jacobian),
            torch.zeros_like(log_det_jacobian),
            log_det_jacobian
        )

        # Clamp log determinant to prevent extreme values
        log_det_jacobian = torch.clamp(log_det_jacobian, min=-50, max=50)

        return z, log_det_jacobian
