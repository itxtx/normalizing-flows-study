import torch
import torch.nn as nn
from src.flows.coupling.coupling_layer import CouplingLayer 

class RealNVP(nn.Module):
    """
    A Real NVP model built by stacking multiple coupling layers.
    """
    def __init__(self, data_dim, n_layers, hidden_dim):
        """
        Args:
            data_dim (int): The dimensionality of the data.
            n_layers (int): The number of coupling layers to stack.
            hidden_dim (int): The hidden dimension for the conditioner MLPs.
        """
        super().__init__()
        
        assert n_layers % 2 == 0, "Number of layers must be even to ensure all dimensions are transformed."
        
        # Create a sequence of coupling layers
        self.layers = nn.ModuleList()
        for i in range(n_layers):
            # Create a general mask that splits dimensions in half.
            # This ensures that all dimensions are conditioned upon and transformed.
            mask = torch.zeros(data_dim)
            if i % 2 == 0:
                mask[:data_dim // 2] = 1 # Condition on the first half, transform the second
            else:
                mask[data_dim // 2:] = 1 # Condition on the second half, transform the first
            
            self.layers.append(CouplingLayer(data_dim, hidden_dim, mask))

    def forward(self, z):
        """
        Forward pass: maps latent samples z to data samples x.
        z -> x
        """
        log_det_J_total = 0
        x = z
        for layer in self.layers:
            x, log_det_J = layer.forward(x)
            log_det_J_total += log_det_J
        return x, log_det_J_total

    def inverse(self, x):
        """
        Inverse pass: maps data samples x to latent samples z.
        x -> z
        """
        log_det_J_inv_total = 0
        z = x
        # The inverse operation must be done in the reverse order of the layers
        for layer in reversed(self.layers):
            z, log_det_J_inv = layer.inverse(z)
            log_det_J_inv_total += log_det_J_inv
        return z, log_det_J_inv_total
