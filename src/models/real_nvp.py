import torch
import torch.nn as nn
from src.flows.coupling.coupling_layer import CouplingLayer
from src.models.normalizing_flow_model import NormalizingFlowModel

class RealNVP(nn.Module):
    """
    A Real NVP model built by stacking multiple coupling layers.
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
        
        # Create a sequence of coupling layers
        layers = []
        for i in range(n_layers):
            # Create a general mask that splits dimensions in half.
            # This ensures that all dimensions are conditioned upon and transformed.
            mask = torch.zeros(data_dim)
            if i % 2 == 0:
                mask[:data_dim // 2] = 1 # Condition on the first half, transform the second
            else:
                mask[data_dim // 2:] = 1 # Condition on the second half, transform the first
            
            layers.append(CouplingLayer(data_dim, hidden_dim, mask))
            
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
