import torch
import torch.nn as nn
from src.flows import CouplingLayer

class RealNVP(nn.Module):
    """
    A Real NVP model built by stacking multiple coupling layers.
    """
    def __init__(self, data_dim, n_layers, hidden_dim):
        """
        Args:
            data_dim (int): The dimensionality of the data (e.g., 2 for two-moons).
            n_layers (int): The number of coupling layers to stack.
            hidden_dim (int): The hidden dimension for the conditioner MLPs.
        """
        super(RealNVP, self).__init__()
        
        assert n_layers % 2 == 0, "Number of layers must be even to ensure all dimensions are transformed."
        
        # Create a sequence of coupling layers
        self.layers = nn.ModuleList()
        for i in range(n_layers):
            # Create the mask. For a 2D dataset, this will alternate
            # between [1, 0] and [0, 1].
            mask = torch.zeros(data_dim)
            if i % 2 == 0:
                mask[0] = 1 # Transform the second dimension
            else:
                mask[1] = 1 # Transform the first dimension
                
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
    
    
class NormalizingFlowModel(nn.Module):
    """
    A sequence of normalizing flow layers.
    """
    def __init__(self, flows):
        super(NormalizingFlowModel, self).__init__()
        self.flows = nn.ModuleList(flows)

    def forward(self, z):
        """
        Apply a sequence of flows to a base distribution sample.
        """
        log_det_jacobian_sum = 0
        for flow in self.flows:
            z, log_det_jacobian = flow(z)
            log_det_jacobian_sum += log_det_jacobian
        return z, log_det_jacobian_sum

    def inverse(self, x):
        """
        Apply the inverse of a sequence of flows.
        """
        log_det_jacobian_sum = 0
        for flow in reversed(self.flows):
            x, log_det_jacobian = flow.inverse(x)
            log_det_jacobian_sum += log_det_jacobian
        return x, log_det_jacobian_sum