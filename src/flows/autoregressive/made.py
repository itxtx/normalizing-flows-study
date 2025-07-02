import torch
import torch.nn as nn
import numpy as np
from .masked_linear import MaskedLinear

class MADE(nn.Module):
    """
    Masked Autoencoder for Distribution Estimation (Conditioner Network).
    This network is autoregressive. For a given input x, the output for
    dimension i is only dependent on inputs x_j where j < i.
    """
    def __init__(self, input_dim, hidden_dim, output_dim_multiplier=2, use_batch_norm=True):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim_multiplier = output_dim_multiplier
        self.use_batch_norm = use_batch_norm
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
        Creates a deeper neural network with masked linear layers and batch normalization.
        """
        layers = []
        
        first_layer = MaskedLinear(
            self.input_dim, 
            self.hidden_dim, 
            mask=self.masks[0]
        )
        layers.append(first_layer)
        if self.use_batch_norm:
            layers.append(nn.BatchNorm1d(self.hidden_dim))
        layers.append(nn.ReLU())
        
        second_layer = nn.Linear(self.hidden_dim, self.hidden_dim)
        layers.append(second_layer)
        if self.use_batch_norm:
            layers.append(nn.BatchNorm1d(self.hidden_dim))
        layers.append(nn.ReLU())
        
        third_layer = nn.Linear(self.hidden_dim, self.hidden_dim)
        layers.append(third_layer)
        if self.use_batch_norm:
            layers.append(nn.BatchNorm1d(self.hidden_dim))
        layers.append(nn.ReLU())
        
        final_layer = MaskedLinear(
            self.hidden_dim, 
            self.input_dim * self.output_dim_multiplier, 
            mask=self.masks[1]
        )
        layers.append(final_layer)
        
        # Initialize with smaller weights for better stability
        nn.init.xavier_normal_(first_layer.weight, gain=0.5)
        if first_layer.bias is not None:
            nn.init.zeros_(first_layer.bias)
        
        nn.init.xavier_normal_(second_layer.weight, gain=0.5)
        if second_layer.bias is not None:
            nn.init.zeros_(second_layer.bias)
            
        nn.init.xavier_normal_(third_layer.weight, gain=0.5)
        if third_layer.bias is not None:
            nn.init.zeros_(third_layer.bias)
        
        # Initialize final layer with very small weights to start near identity
        nn.init.normal_(final_layer.weight, mean=0.0, std=0.01)
        if final_layer.bias is not None:
            nn.init.zeros_(final_layer.bias)
        
        return nn.Sequential(*layers)
    
    def forward(self, x):
        """
        The forward pass of the conditioner network.
        """
        output = self.net(x)
        # More conservative clamping to prevent extreme values
        return torch.clamp(output, min=-2.0, max=2.0)
