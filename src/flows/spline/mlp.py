import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    """
    A simple Multi-Layer Perceptron (MLP) with batch normalization.
    """
    def __init__(self, input_dim, output_dim, hidden_dim, num_layers, activation='relu', use_batch_norm=True):
        super().__init__()
        
        if activation == 'relu':
            self.activation = F.relu
        elif activation == 'tanh':
            self.activation = torch.tanh
        else:
            raise ValueError(f"Unknown activation function: {activation}")
        
        self.use_batch_norm = use_batch_norm
        layers = []
        
        # First layer
        layers.append(nn.Linear(input_dim, hidden_dim))
        if use_batch_norm:
            layers.append(nn.BatchNorm1d(hidden_dim))
        layers.append(self.activation)
        
        # Hidden layers
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(self.activation)
        
        # Output layer (no batch norm or activation)
        layers.append(nn.Linear(hidden_dim, output_dim))
        
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        """
        Forward pass through the MLP.
        """
        for layer in self.layers:
            x = layer(x)
        return x
