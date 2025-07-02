import torch
from ..flow.flow import Flow
from src.flows.autoregressive.made import MADE

class InverseAutoregressiveFlow(Flow):
    """
    Inverse Autoregressive Flow (IAF). This flow has the same expressiveness as
    MAF but with a reversed computational trade-off.
    The forward transformation (sampling) is fast and parallel.
    The inverse transformation (density evaluation) is slow and sequential.
    """
    def __init__(self, dim, hidden_dim=64):
        super().__init__()
        self.dim = dim
        self.conditioner = MADE(dim, hidden_dim, 2)

    def forward(self, z):
        """
        Computes x = f(z). This direction is fast and parallel.
        x_i = z_i * exp(alpha_i) + mu_i, where mu_i and alpha_i are functions
        of z_1, ..., z_{i-1}.
        """
        params = self.conditioner(z)
        mu, alpha = params.chunk(2, dim=1)
        
        alpha = torch.clamp(alpha, min=-3, max=3)
        
        log_scale = alpha
        scale = torch.exp(torch.clamp(log_scale, min=-5, max=5))
        x = z * scale + mu
        
        log_det_jacobian = torch.sum(alpha, dim=1)
        
        x = torch.where(torch.isnan(x) | torch.isinf(x), torch.zeros_like(x), x)
        log_det_jacobian = torch.where(
            torch.isnan(log_det_jacobian) | torch.isinf(log_det_jacobian),
            torch.zeros_like(log_det_jacobian),
            log_det_jacobian
        )
        
        log_det_jacobian = torch.clamp(log_det_jacobian, min=-100, max=100)
        
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
            
            alpha = torch.clamp(alpha, min=-3, max=3)
            
            log_scale = -alpha[:, i]
            scale = torch.exp(torch.clamp(log_scale, min=-5, max=5))
            z_new = z.clone()
            z_new[:, i] = (x[:, i] - mu[:, i]) * scale
            z = z_new
            
            log_det_jacobian -= alpha[:, i]

        z = torch.where(torch.isnan(z) | torch.isinf(z), torch.zeros_like(z), z)
        log_det_jacobian = torch.where(
            torch.isnan(log_det_jacobian) | torch.isinf(log_det_jacobian),
            torch.zeros_like(log_det_jacobian),
            log_det_jacobian
        )

        log_det_jacobian = torch.clamp(log_det_jacobian, min=-100, max=100)

        return z, log_det_jacobian
