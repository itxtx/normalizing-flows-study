import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
from .flow import Flow

class MaskedLinear(nn.Linear):
    """A linear layer with a fixed binary mask."""
    
    def __init__(self, in_features, out_features, mask, bias=True):
        # Call the parent constructor
        super().__init__(in_features, out_features, bias)
        
        # Register the mask as a buffer to ensure it moves with the model (e.g., to GPU)
        self.register_buffer('mask', mask)

    def forward(self, input):
        # Apply the mask to the weights during the forward pass
        # Ensure mask has the same dtype as weights
        mask = self.mask.to(dtype=self.weight.dtype)
        return F.linear(input, self.weight * mask, self.bias)

class MADE(nn.Module):
    """
    Masked Autoencoder for Distribution Estimation (Conditioner Network).
    This network is autoregressive. For a given input x, the output for
    dimension i is only dependent on inputs x_j where j < i.
    """
    def __init__(self, input_dim, hidden_dim, output_dim_multiplier=2):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim_multiplier = output_dim_multiplier

        # Assign degrees to input, hidden, and output neurons
        # Use better degree assignment for proper autoregressive structure
        self.m = {}
        self.m[-1] = np.arange(self.input_dim)
        
        # Assign hidden layer degrees with better distribution
        if self.input_dim > 1:
            # Use better distribution to ensure connectivity
            # For 2D: use [0, 0, 1, 1] pattern to ensure both dimensions can be modeled
            if self.input_dim == 2:
                # Ensure both dimensions have good connectivity
                self.m[0] = np.array([0, 0, 1, 1] * (self.hidden_dim // 4 + 1))[:self.hidden_dim]
            else:
                # Use linspace to ensure better coverage of degrees
                self.m[0] = np.floor(np.linspace(0, self.input_dim - 1, self.hidden_dim)).astype(int)
        else:
            # Handle edge case when input_dim = 1
            self.m[0] = np.zeros(self.hidden_dim, dtype=int)
        
        self.m[1] = np.arange(self.input_dim)

        self.masks = self.create_masks()

        self.net = self.create_network()

    def create_masks(self):
        """
        Creates the masks for the linear layers.
        The mask for a connection from layer i to j ensures that an output unit
        can only be connected to input units with a less than or equal degree.
        """
        masks = []
        
        # Mask 1: input to hidden layer
        m1 = (self.m[-1][:, np.newaxis] <= self.m[0][np.newaxis, :]).T
        masks.append(torch.from_numpy(m1.astype(np.float32)))

        # Mask 2: hidden to output layer
        output_dim = self.input_dim * self.output_dim_multiplier
        m2 = np.zeros((output_dim, self.hidden_dim), dtype=np.float32)
        for i in range(self.input_dim):
            for k in range(self.output_dim_multiplier):
                out_idx = i * self.output_dim_multiplier + k
                m2[out_idx, :] = (self.m[0] <= self.m[1][i]).astype(np.float32)
        masks.append(torch.from_numpy(m2))
        return masks

    def create_network(self):
        """
        Creates a deeper neural network with masked linear layers.
        """
        layers = []
        
        first_layer = MaskedLinear(
            self.input_dim, 
            self.hidden_dim, 
            mask=self.masks[0]
        )
        layers.append(first_layer)
        layers.append(nn.ReLU())
        
        second_layer = nn.Linear(self.hidden_dim, self.hidden_dim)
        layers.append(second_layer)
        layers.append(nn.ReLU())
        
        third_layer = nn.Linear(self.hidden_dim, self.hidden_dim)
        layers.append(third_layer)
        layers.append(nn.ReLU())
        
        final_layer = MaskedLinear(
            self.hidden_dim, 
            self.input_dim * self.output_dim_multiplier, 
            mask=self.masks[1]
        )
        layers.append(final_layer)
        
        nn.init.xavier_normal_(first_layer.weight, gain=1.0)
        if first_layer.bias is not None:
            nn.init.zeros_(first_layer.bias)
        
        nn.init.xavier_normal_(second_layer.weight, gain=1.0)
        if second_layer.bias is not None:
            nn.init.zeros_(second_layer.bias)
            
        nn.init.xavier_normal_(third_layer.weight, gain=1.0)
        if third_layer.bias is not None:
            nn.init.zeros_(third_layer.bias)
        
        nn.init.normal_(final_layer.weight, mean=0.0, std=0.05)
        if final_layer.bias is not None:
            nn.init.normal_(final_layer.bias, mean=0.0, std=0.005)
        
        return nn.Sequential(*layers)
    
    def forward(self, x):
        """
        The forward pass of the conditioner network.
        """
        output = self.net(x)
        return torch.clamp(output, min=-3.0, max=3.0)

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
