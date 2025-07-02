import torch
import torch.nn as nn
from src.flows.coupling import CouplingLayer 
from src.flows.spline import SplineCouplingLayer
from src.flows.autoregressive import MaskedAutoregressiveFlow, InverseAutoregressiveFlow


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
    
class RealNVPSpline(nn.Module):
    """
    A Real NVP model using rational-quadratic spline coupling layers.
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
        self.layers = nn.ModuleList()
        # Define the two mask patterns once
        mask_a = torch.zeros(data_dim)
        mask_a[:data_dim // 2] = 1

        mask_b = 1 - mask_a # A simple way to get the other half

        for i in range(n_layers):
            # Alternate between the two masks
            mask = mask_a if i % 2 == 0 else mask_b
            self.layers.append(SplineCouplingLayer(data_dim, hidden_dim, mask))

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
    A generic model to chain a sequence of normalizing flow layers.
    """
    def __init__(self, flows):
        super().__init__()
        self.flows = nn.ModuleList(flows)

    def forward(self, z):
        """
        Apply a sequence of flows to a base distribution sample.
        This is the sampling direction. z -> x
        """
        log_det_jacobian_sum = 0
        # The __call__ method of a nn.Module calls its forward method.
        # So, flow(z) is equivalent to flow.forward(z).
        for flow in self.flows:
            z, log_det_jacobian = flow(z)
            log_det_jacobian_sum += log_det_jacobian
        return z, log_det_jacobian_sum

    def inverse(self, x):
        """
        Apply the inverse of a sequence of flows.
        This is the density estimation direction. x -> z
        """
        log_det_jacobian_sum = 0
        for flow in reversed(self.flows):
            x, log_det_jacobian = flow.inverse(x)
            log_det_jacobian_sum += log_det_jacobian
        return x, log_det_jacobian_sum
