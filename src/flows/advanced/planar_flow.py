"""
Planar Flow implementation.

This module implements Planar Flows, which are simple normalizing flows that
apply planar transformations to the input. The transformation is of the form:
f(z) = z + u * tanh(w^T z + b)

where u and w are learnable vectors and b is a learnable scalar.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
from ..flow.flow import Flow


class PlanarFlow(Flow):
    """
    Planar Flow implementation.
    
    The planar flow applies the transformation:
    f(z) = z + u * tanh(w^T z + b)
    
    where:
    - u, w are learnable vectors of dimension dim
    - b is a learnable scalar
    - tanh is the activation function
    
    The transformation is invertible when u^T w >= -1, which we enforce
    by reparameterizing u to ensure this constraint.
    """
    
    def __init__(self, dim: int):
        """
        Initialize Planar Flow.
        
        Args:
            dim: Dimensionality of the data
        """
        super().__init__()
        self.data_dim = dim
        self.dim = dim
        
        # Learnable parameters
        self.w = nn.Parameter(torch.randn(dim))
        self.u_hat = nn.Parameter(torch.randn(dim))
        self.b = nn.Parameter(torch.randn(1))
        
        # Initialize parameters
        self._initialize_parameters()
    
    def _initialize_parameters(self):
        """Initialize parameters for stability."""
        nn.init.normal_(self.w, mean=0, std=0.1)
        nn.init.normal_(self.u_hat, mean=0, std=0.1)
        nn.init.normal_(self.b, mean=0, std=0.1)
    
    def _get_u(self) -> torch.Tensor:
        """
        Get the constrained u parameter.
        
        To ensure invertibility, we need u^T w >= -1.
        We reparameterize u as:
        u = u_hat + (m(w^T u_hat) - w^T u_hat) * w / ||w||^2
        
        where m(x) = -1 + log(1 + exp(x)) is a smooth approximation to max(-1, x).
        
        Returns:
            u: Constrained u parameter
        """
        # Compute w^T u_hat
        wtu_hat = torch.dot(self.w, self.u_hat)
        
        # Compute m(w^T u_hat) = -1 + log(1 + exp(w^T u_hat))
        # Use log1p for numerical stability
        m_wtu = -1 + torch.log1p(torch.exp(wtu_hat))
        
        # Compute ||w||^2
        w_norm_sq = torch.sum(self.w ** 2)
        
        # Constrained u
        u = self.u_hat + (m_wtu - wtu_hat) * self.w / (w_norm_sq + 1e-8)
        
        return u
    
    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward transformation: x = f(z) = z + u * tanh(w^T z + b).
        
        Args:
            z: Input tensor from base distribution space [batch_size, dim]
            
        Returns:
            x: Transformed tensor in data space [batch_size, dim]
            log_det_jacobian: Log-determinant of Jacobian [batch_size]
        """
        # Get constrained u
        u = self._get_u()
        
        # Compute linear transformation
        linear_term = torch.mv(z, self.w) + self.b  # [batch_size]
        
        # Apply activation
        activation = torch.tanh(linear_term)  # [batch_size]
        
        # Compute transformation
        x = z + u.unsqueeze(0) * activation.unsqueeze(1)  # [batch_size, dim]
        
        # Compute log-determinant of Jacobian
        # J = I + u * (1 - tanh^2(w^T z + b)) * w^T
        # det(J) = 1 + u^T w * (1 - tanh^2(w^T z + b))
        psi = 1 - activation ** 2  # derivative of tanh
        uTw = torch.dot(u, self.w)
        det_jacobian = 1 + uTw * psi  # [batch_size]
        
        # Compute log-determinant with numerical stability
        log_det_jacobian = torch.log(torch.abs(det_jacobian) + 1e-8)
        
        # Handle numerical issues
        log_det_jacobian = torch.where(
            torch.isnan(log_det_jacobian) | torch.isinf(log_det_jacobian),
            torch.zeros_like(log_det_jacobian),
            log_det_jacobian
        )
        
        return x, log_det_jacobian
    
    def inverse(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Inverse transformation: z = f^(-1)(x).
        
        Since planar flows don't have a closed-form inverse, we use
        fixed-point iteration to solve: x = z + u * tanh(w^T z + b) for z.
        
        Args:
            x: Input tensor from data space [batch_size, dim]
            
        Returns:
            z: Transformed tensor in base distribution space [batch_size, dim]
            log_det_jacobian: Log-determinant of Jacobian [batch_size]
        """
        batch_size = x.size(0)
        device = x.device
        dtype = x.dtype
        
        # Get constrained u
        u = self._get_u()
        
        # Initialize z with x (good starting point)
        z = x.clone()
        
        # Fixed-point iteration: solve x = z + u * tanh(w^T z + b) for z
        max_iter = 50
        tol = 1e-6
        
        for i in range(max_iter):
            # Compute current transformation
            linear_term = torch.mv(z, self.w) + self.b
            activation = torch.tanh(linear_term)
            z_new = x - u.unsqueeze(0) * activation.unsqueeze(1)
            
            # Check convergence
            diff = torch.norm(z_new - z, dim=1)
            if torch.all(diff < tol):
                break
            
            z = z_new
        
        # Compute log-determinant (negative of forward)
        linear_term = torch.mv(z, self.w) + self.b
        activation = torch.tanh(linear_term)
        psi = 1 - activation ** 2
        uTw = torch.dot(u, self.w)
        det_jacobian = 1 + uTw * psi
        
        log_det_jacobian = -torch.log(torch.abs(det_jacobian) + 1e-8)
        
        # Handle numerical issues
        log_det_jacobian = torch.where(
            torch.isnan(log_det_jacobian) | torch.isinf(log_det_jacobian),
            torch.zeros_like(log_det_jacobian),
            log_det_jacobian
        )
        
        return z, log_det_jacobian