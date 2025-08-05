"""
Neural Autoregressive Flow (NAF) implementation.

This module implements a Neural Autoregressive Flow with deep autoregressive networks,
residual connections, layer normalization, and configurable activation functions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Optional, Union
from ..flow.flow import Flow
from ..autoregressive.masked_linear import MaskedLinear


class DeepMADE(nn.Module):
    """
    Deep Masked Autoencoder for Distribution Estimation with residual connections
    and layer normalization for improved stability and expressiveness.
    
    This extends the basic MADE with deeper networks and modern techniques.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        output_dim_multiplier: int = 2,
        activation: str = "relu",
        use_layer_norm: bool = True,
        use_residual: bool = True,
        dropout: float = 0.0
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim_multiplier = output_dim_multiplier
        self.use_layer_norm = use_layer_norm
        self.use_residual = use_residual
        self.dropout = dropout
        
        # Activation function
        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "elu":
            self.activation = nn.ELU()
        elif activation == "leaky_relu":
            self.activation = nn.LeakyReLU(0.1)
        elif activation == "gelu":
            self.activation = nn.GELU()
        else:
            raise ValueError(f"Unsupported activation: {activation}")
        
        # Use the same degree assignment as the original MADE for consistency
        self.m = {}
        self.m[-1] = np.arange(self.input_dim)
        
        # Assign hidden layer degrees with better distribution
        if len(hidden_dims) > 0:
            if self.input_dim > 1:
                # Use better distribution to ensure connectivity
                if self.input_dim == 2:
                    # Ensure both dimensions have good connectivity
                    self.m[0] = np.array([0, 0, 1, 1] * (self.hidden_dims[0] // 4 + 1))[:self.hidden_dims[0]]
                else:
                    # Use linspace to ensure better coverage of degrees
                    self.m[0] = np.floor(np.linspace(0, self.input_dim - 1, self.hidden_dims[0])).astype(int)
            else:
                # Handle edge case when input_dim = 1
                self.m[0] = np.zeros(self.hidden_dims[0], dtype=int)
        
        # For additional hidden layers, use similar distribution
        for i in range(1, len(hidden_dims)):
            if self.input_dim > 1:
                self.m[i] = np.floor(np.linspace(0, self.input_dim - 1, self.hidden_dims[i])).astype(int)
            else:
                self.m[i] = np.zeros(self.hidden_dims[i], dtype=int)
        
        # Create masks
        self.masks = self._create_masks()
        
        # Build the network
        self.network = self._build_network()
        
    def _create_masks(self):
        """Create masks for autoregressive connections."""
        masks = []
        
        if len(self.hidden_dims) == 0:
            return masks
        
        # Mask 1: input to first hidden layer
        m1 = (self.m[-1][:, np.newaxis] <= self.m[0][np.newaxis, :]).T
        masks.append(torch.from_numpy(m1.astype(np.float32)))
        
        # Masks for hidden to hidden layers
        for i in range(len(self.hidden_dims) - 1):
            m_hidden = (self.m[i][:, np.newaxis] <= self.m[i + 1][np.newaxis, :]).T
            masks.append(torch.from_numpy(m_hidden.astype(np.float32)))
        
        # Final mask: last hidden to output layer
        # Output format: [mu_0, mu_1, ..., mu_{d-1}, alpha_0, alpha_1, ..., alpha_{d-1}]
        output_dim = self.input_dim * self.output_dim_multiplier
        m_final = np.zeros((output_dim, self.hidden_dims[-1]), dtype=np.float32)
        
        # First half: mu parameters
        for i in range(self.input_dim):
            # mu_i should depend on hidden units with degree <= (i-1)
            m_final[i, :] = (self.m[len(self.hidden_dims) - 1] <= i - 1).astype(np.float32)
        
        # Second half: alpha parameters  
        for i in range(self.input_dim):
            # alpha_i should depend on hidden units with degree <= (i-1)
            m_final[self.input_dim + i, :] = (self.m[len(self.hidden_dims) - 1] <= i - 1).astype(np.float32)
        
        masks.append(torch.from_numpy(m_final))
        
        return masks
    
    def _build_network(self):
        """Build the deep autoregressive network."""
        layers = []
        
        if len(self.hidden_dims) == 0:
            # Direct input to output connection
            layer = MaskedLinear(
                self.input_dim,
                self.input_dim * self.output_dim_multiplier,
                mask=None
            )
            layers.append(layer)
            return nn.Sequential(*layers)
        
        # Input to first hidden layer
        first_layer = MaskedLinear(
            self.input_dim,
            self.hidden_dims[0],
            mask=self.masks[0]
        )
        layers.append(first_layer)
        
        if self.use_layer_norm:
            layers.append(nn.LayerNorm(self.hidden_dims[0]))
        
        layers.append(self.activation)
        
        if self.dropout > 0:
            layers.append(nn.Dropout(self.dropout))
        
        # Hidden layers
        for i in range(1, len(self.hidden_dims)):
            layer = MaskedLinear(
                self.hidden_dims[i - 1],
                self.hidden_dims[i],
                mask=self.masks[i]
            )
            
            if self.use_residual and self.hidden_dims[i - 1] == self.hidden_dims[i]:
                # Residual connection
                residual_block = ResidualBlock(
                    layer,
                    self.activation,
                    self.use_layer_norm,
                    self.dropout
                )
                layers.append(residual_block)
            else:
                layers.append(layer)
                if self.use_layer_norm:
                    layers.append(nn.LayerNorm(self.hidden_dims[i]))
                layers.append(self.activation)
                if self.dropout > 0:
                    layers.append(nn.Dropout(self.dropout))
        
        # Final layer to output
        final_layer = MaskedLinear(
            self.hidden_dims[-1],
            self.input_dim * self.output_dim_multiplier,
            mask=self.masks[-1]
        )
        layers.append(final_layer)
        
        # Initialize weights
        self._initialize_weights(layers)
        
        return nn.Sequential(*layers)
    
    def _initialize_weights(self, layers):
        """Initialize network weights for stability."""
        for layer in layers:
            if isinstance(layer, (MaskedLinear, nn.Linear)):
                nn.init.xavier_normal_(layer.weight, gain=0.5)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)
            elif isinstance(layer, ResidualBlock):
                nn.init.xavier_normal_(layer.linear.weight, gain=0.1)
                if layer.linear.bias is not None:
                    nn.init.zeros_(layer.linear.bias)
    
    def forward(self, x):
        """Forward pass through the deep autoregressive network."""
        output = self.network(x)
        # Conservative clamping to prevent extreme values
        return torch.clamp(output, min=-2.0, max=2.0)


class ResidualBlock(nn.Module):
    """Residual block for deep autoregressive networks."""
    
    def __init__(self, linear_layer, activation, use_layer_norm=True, dropout=0.0):
        super().__init__()
        self.linear = linear_layer
        self.activation = activation
        self.use_layer_norm = use_layer_norm
        self.dropout = dropout
        
        if use_layer_norm:
            self.layer_norm = nn.LayerNorm(linear_layer.out_features)
        
        if dropout > 0:
            self.dropout_layer = nn.Dropout(dropout)
    
    def forward(self, x):
        """Forward pass with residual connection."""
        residual = x
        
        out = self.linear(x)
        
        if self.use_layer_norm:
            out = self.layer_norm(out)
        
        out = self.activation(out)
        
        if self.dropout > 0:
            out = self.dropout_layer(out)
        
        # Add residual connection
        return out + residual


class NeuralAutoregressiveFlow(Flow):
    """
    Neural Autoregressive Flow (NAF) with deep autoregressive networks.
    
    This flow extends the standard Masked Autoregressive Flow (MAF) with:
    - Deeper, more expressive conditioner networks
    - Residual connections for improved gradient flow
    - Layer normalization for training stability
    - Configurable activation functions and dropout
    
    The inverse transformation (density evaluation) is fast and parallel.
    The forward transformation (sampling) is slow and sequential.
    """
    
    def __init__(
        self,
        dim: int,
        hidden_dims: List[int] = [512, 512, 512],
        activation: str = "relu",
        use_layer_norm: bool = True,
        use_residual: bool = True,
        dropout: float = 0.1,
        clamp_alpha: float = 3.0,
        clamp_log_scale: float = 5.0
    ):
        """
        Initialize Neural Autoregressive Flow.
        
        Args:
            dim: Dimensionality of the data
            hidden_dims: List of hidden layer dimensions
            activation: Activation function ("relu", "elu", "leaky_relu", "gelu")
            use_layer_norm: Whether to use layer normalization
            use_residual: Whether to use residual connections
            dropout: Dropout probability
            clamp_alpha: Clamping value for log-scale parameters
            clamp_log_scale: Clamping value for log-scale computation
        """
        super().__init__()
        self.data_dim = dim
        self.dim = dim
        self.clamp_alpha = clamp_alpha
        self.clamp_log_scale = clamp_log_scale
        
        # Deep autoregressive conditioner network
        self.conditioner = DeepMADE(
            input_dim=dim,
            hidden_dims=hidden_dims,
            output_dim_multiplier=2,
            activation=activation,
            use_layer_norm=use_layer_norm,
            use_residual=use_residual,
            dropout=dropout
        )
    
    def inverse(self, x):
        """
        Compute z = g(x) (inverse transformation).
        
        This direction is fast and can be computed in parallel.
        z_i = (x_i - mu_i) * exp(-alpha_i)
        
        Args:
            x: Input tensor from data space [batch_size, dim]
            
        Returns:
            z: Transformed tensor in base distribution space [batch_size, dim]
            log_det_jacobian: Log-determinant of Jacobian [batch_size]
        """
        # Get transformation parameters from conditioner network
        params = self.conditioner(x)
        mu, alpha = params.chunk(2, dim=1)
        
        # Clamp alpha for numerical stability
        alpha = torch.clamp(alpha, min=-self.clamp_alpha, max=self.clamp_alpha)
        
        # Compute transformation
        log_scale = -alpha
        log_scale = torch.clamp(log_scale, min=-self.clamp_log_scale, max=self.clamp_log_scale)
        scale = torch.exp(log_scale)
        
        z = (x - mu) * scale
        
        # Log-determinant of Jacobian
        log_det_jacobian = torch.sum(log_scale, dim=1)
        
        # Handle numerical issues
        z = torch.where(torch.isnan(z) | torch.isinf(z), torch.zeros_like(z), z)
        log_det_jacobian = torch.where(
            torch.isnan(log_det_jacobian) | torch.isinf(log_det_jacobian),
            torch.zeros_like(log_det_jacobian),
            log_det_jacobian
        )
        
        # Final clamping for stability
        log_det_jacobian = torch.clamp(log_det_jacobian, min=-100, max=100)
        
        return z, log_det_jacobian
    
    def forward(self, z):
        """
        Compute x = f(z) (forward transformation).
        
        This direction is slow as it must be computed sequentially.
        x_i = z_i * exp(alpha_i) + mu_i
        
        Args:
            z: Input tensor from base distribution space [batch_size, dim]
            
        Returns:
            x: Transformed tensor in data space [batch_size, dim]
            log_det_jacobian: Log-determinant of Jacobian [batch_size]
        """
        batch_size = z.size(0)
        device = z.device
        dtype = z.dtype
        
        x = torch.zeros(batch_size, self.dim, device=device, dtype=dtype)
        log_det_jacobian = torch.zeros(batch_size, device=device, dtype=dtype)
        
        # Sequential computation for autoregressive structure
        for i in range(self.dim):
            # Get parameters based on current partial x
            params = self.conditioner(x)
            mu, alpha = params.chunk(2, dim=1)
            
            # Clamp alpha for numerical stability
            alpha = torch.clamp(alpha, min=-self.clamp_alpha, max=self.clamp_alpha)
            
            # Compute transformation for dimension i
            log_scale_i = alpha[:, i]
            log_scale_i = torch.clamp(log_scale_i, min=-self.clamp_log_scale, max=self.clamp_log_scale)
            scale_i = torch.exp(log_scale_i)
            
            # Update x in-place for dimension i
            x[:, i] = z[:, i] * scale_i + mu[:, i]
            log_det_jacobian += log_scale_i
        
        # Handle numerical issues
        x = torch.where(torch.isnan(x) | torch.isinf(x), torch.zeros_like(x), x)
        log_det_jacobian = torch.where(
            torch.isnan(log_det_jacobian) | torch.isinf(log_det_jacobian),
            torch.zeros_like(log_det_jacobian),
            log_det_jacobian
        )
        
        # Final clamping for stability
        log_det_jacobian = torch.clamp(log_det_jacobian, min=-100, max=100)
        
        return x, log_det_jacobian