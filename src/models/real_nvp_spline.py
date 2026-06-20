import torch
import torch.nn as nn
from src.flows.spline.spline_coupling_layer import SplineCouplingLayer
from src.models.normalizing_flow_model import NormalizingFlowModel

class RealNVPSpline(nn.Module):
    """
    A Real NVP model using rational-quadratic spline coupling layers.
    """
    def __init__(self, data_dim, n_layers, hidden_dim, batch_norm_between_layers=False):
        """
        Args:
            data_dim (int): The dimensionality of the data.
            n_layers (int): The number of coupling layers to stack.
            hidden_dim (int): The hidden dimension for the conditioner MLPs.
            batch_norm_between_layers (bool): Whether to use batch norm between layers.
        """
        super().__init__()
        
        assert n_layers % 2 == 0, "Number of layers must be even to ensure all dimensions are transformed."
        
        layers = []
        # Define the two mask patterns once
        mask_a = torch.zeros(data_dim)
        mask_a[:data_dim // 2] = 1

        mask_b = 1 - mask_a # A simple way to get the other half

        for i in range(n_layers):
            # Alternate between the two masks
            mask = mask_a if i % 2 == 0 else mask_b
            layers.append(SplineCouplingLayer(data_dim, hidden_dim, mask))
            
        self.flow = NormalizingFlowModel(layers, batch_norm_between_layers)

    def forward(self, z):
        """
        Forward pass: maps latent samples z to data samples x.
        z -> x
        """
        return self.flow.forward(z)

    def inverse(self, x):
        """
        Inverse pass: maps data samples x to latent samples z.
        x -> z
        """
        return self.flow.inverse(x)
