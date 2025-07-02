import torch
from ..flow.flow import Flow
from src.flows.autoregressive.made import MADE

class MaskedAutoregressiveFlow(Flow):
    """
    Masked Autoregressive Flow (MAF). This flow is universal, meaning it can
    approximate any density.
    The inverse transformation (density evaluation) is fast and parallel.
    The forward transformation (sampling) is slow and sequential.
    """
    def __init__(self, dim, hidden_dim=64):
        super().__init__()
        self.dim = dim
        self.conditioner = MADE(dim, hidden_dim, 2)
        
    def inverse(self, x):
        """
        Computes z = g(x). This direction is fast.
        z_i = (x_i - mu_i) * exp(-alpha_i), where mu_i and alpha_i are functions
        of x_1, ..., x_{i-1}.
        """
        params = self.conditioner(x)
        mu, alpha = params.chunk(2, dim=1)
        
        alpha = torch.clamp(alpha, min=-3, max=3)

        log_scale = -alpha
        scale = torch.exp(torch.clamp(log_scale, min=-5, max=5))
        z = (x - mu) * scale
        
        log_det_jacobian = -torch.sum(alpha, dim=1)
        
        z = torch.where(torch.isnan(z) | torch.isinf(z), torch.zeros_like(z), z)
        log_det_jacobian = torch.where(
            torch.isnan(log_det_jacobian) | torch.isinf(log_det_jacobian),
            torch.zeros_like(log_det_jacobian),
            log_det_jacobian
        )
        
        log_det_jacobian = torch.clamp(log_det_jacobian, min=-100, max=100)
        
        return z, log_det_jacobian

    def forward(self, z):
        """
        Computes x = f(z). This direction is slow as it must be done sequentially.
        x_i = z_i * exp(alpha_i) + mu_i
        """
        batch_size = z.size(0)
        x = torch.zeros(batch_size, self.dim, device=z.device, dtype=z.dtype)
        log_det_jacobian = torch.zeros(batch_size, device=z.device, dtype=z.dtype)

        for i in range(self.dim):
            params = self.conditioner(x)
            mu, alpha = params.chunk(2, dim=1)
            
            alpha = torch.clamp(alpha, min=-3, max=3)
            
            log_scale = alpha[:, i]
            scale = torch.exp(torch.clamp(log_scale, min=-5, max=5))
            x_new = x.clone()
            x_new[:, i] = z[:, i] * scale + mu[:, i]
            x = x_new
            
            log_det_jacobian += alpha[:, i]

        x = torch.where(torch.isnan(x) | torch.isinf(x), torch.zeros_like(x), x)
        log_det_jacobian = torch.where(
            torch.isnan(log_det_jacobian) | torch.isinf(log_det_jacobian),
            torch.zeros_like(log_det_jacobian),
            log_det_jacobian
        )
        
        log_det_jacobian = torch.clamp(log_det_jacobian, min=-100, max=100)
        
        return x, log_det_jacobian
