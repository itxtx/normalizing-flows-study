"""
Radial Flow implementation.

This module implements Radial Flows, which are simple normalizing flows that
apply radial transformations to the input. The transformation is of the form:
f(z) = z + β * h(α, r) * (z - z0)

where:
- z0 is a learnable reference point
- α and β are learnable scalars
- r = ||z - z0|| is the distance from the reference point
- h(α, r) = 1 / (α + r) is the radial function
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
from ..flow.flow import Flow


class RadialFlow(Flow):
    """
    Radial Flow implementation.
    
    The radial flow applies the transformation:
    f(z) = z + β * h(α, r) * (z - z0)
    
    where:
    - z0 is a learnable reference point (center) of dimension dim
    - α is a learnable positive scalar (controls the "sharpness")
    - β is a learnable scalar (controls the "strength")
    - r = ||z - z0|| is the distance from the reference point
    - h(α, r) = 1 / (α + r) is the radial function
    
    The transformation is invertible when β > -α, which we enforce
    by reparameterizing β.
    """
    
    def __init__(self, dim: int):
        """
        Initialize Radial Flow.
        
        Args:
            dim: Dimensionality of the data
        """
        super().__init__()
        self.data_dim = dim
        self.dim = dim
        
        # Learnable parameters
        self.z0 = nn.Parameter(torch.randn(dim))  # reference point
        self.alpha_hat = nn.Parameter(torch.randn(1))  # unconstrained alpha
        self.beta_hat = nn.Parameter(torch.randn(1))   # unconstrained beta
        
        # Initialize parameters
        self._initialize_parameters()
    
    def _initialize_parameters(self):
        """Initialize parameters for stability."""
        nn.init.normal_(self.z0, mean=0, std=0.1)
        nn.init.normal_(self.alpha_hat, mean=0, std=0.1)
        nn.init.normal_(self.beta_hat, mean=0, std=0.1)
    
    def _get_alpha(self) -> torch.Tensor:
        """
        Get the constrained alpha parameter.
        
        Alpha must be positive, so we use:
        α = log(1 + exp(α_hat))
        
        Returns:
            alpha: Positive alpha parameter
        """
        return F.softplus(self.alpha_hat)
    
    def _get_beta(self, alpha: torch.Tensor) -> torch.Tensor:
        """
        Get the constrained beta parameter.
        
        To ensure invertibility, we need β > -α.
        We reparameterize β as:
        β = -α + log(1 + exp(β_hat))
        
        Args:
            alpha: The alpha parameter
            
        Returns:
            beta: Constrained beta parameter
        """
        return -alpha + F.softplus(self.beta_hat)
    
    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward transformation: x = f(z) = z + β * h(α, r) * (z - z0).
        
        Args:
            z: Input tensor from base distribution space [batch_size, dim]
            
        Returns:
            x: Transformed tensor in data space [batch_size, dim]
            log_det_jacobian: Log-determinant of Jacobian [batch_size]
        """
        batch_size = z.size(0)
        
        # Get constrained parameters
        alpha = self._get_alpha()
        beta = self._get_beta(alpha)
        
        # Compute distance from reference point
        z_minus_z0 = z - self.z0.unsqueeze(0)  # [batch_size, dim]
        r = torch.norm(z_minus_z0, dim=1, keepdim=True)  # [batch_size, 1]
        
        # Compute radial function h(α, r) = 1 / (α + r)
        h = 1.0 / (alpha + r + 1e-8)  # [batch_size, 1]
        
        # Compute transformation
        x = z + beta * h * z_minus_z0  # [batch_size, dim]
        
        # Compute log-determinant of Jacobian
        # The Jacobian is: J = I + β * [h * I + h' * (z-z0)(z-z0)^T / r]
        # where h' = -1 / (α + r)^2
        # det(J) = (1 + β * h)^(d-1) * (1 + β * h + β * h' * r)
        # where d is the dimension
        
        h_prime = -1.0 / ((alpha + r) ** 2 + 1e-8)  # [batch_size, 1]
        
        # Compute determinant terms
        term1 = 1 + beta * h  # [batch_size, 1]
        term2 = 1 + beta * h + beta * h_prime * r  # [batch_size, 1]
        
        # Log-determinant
        log_det_jacobian = (self.dim - 1) * torch.log(torch.abs(term1) + 1e-8) + \
                          torch.log(torch.abs(term2) + 1e-8)
        log_det_jacobian = log_det_jacobian.squeeze(1)  # [batch_size]
        
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
        
        Since radial flows don't have a closed-form inverse, we use
        fixed-point iteration to solve: x = z + β * h(α, r) * (z - z0) for z.
        
        Args:
            x: Input tensor from data space [batch_size, dim]
            
        Returns:
            z: Transformed tensor in base distribution space [batch_size, dim]
            log_det_jacobian: Log-determinant of Jacobian [batch_size]
        """
        batch_size = x.size(0)
        device = x.device
        dtype = x.dtype
        
        # Get constrained parameters
        alpha = self._get_alpha()
        beta = self._get_beta(alpha)
        
        # Initialize z with x (good starting point)
        z = x.clone()
        
        # Fixed-point iteration: solve x = z + β * h(α, r) * (z - z0) for z
        max_iter = 50
        tol = 1e-6
        
        for i in range(max_iter):
            # Compute current transformation
            z_minus_z0 = z - self.z0.unsqueeze(0)
            r = torch.norm(z_minus_z0, dim=1, keepdim=True)
            h = 1.0 / (alpha + r + 1e-8)
            
            z_new = x - beta * h * z_minus_z0
            
            # Check convergence
            diff = torch.norm(z_new - z, dim=1)
            if torch.all(diff < tol):
                break
            
            z = z_new
        
        # Compute log-determinant (negative of forward)
        z_minus_z0 = z - self.z0.unsqueeze(0)
        r = torch.norm(z_minus_z0, dim=1, keepdim=True)
        h = 1.0 / (alpha + r + 1e-8)
        h_prime = -1.0 / ((alpha + r) ** 2 + 1e-8)
        
        term1 = 1 + beta * h
        term2 = 1 + beta * h + beta * h_prime * r
        
        log_det_jacobian = -((self.dim - 1) * torch.log(torch.abs(term1) + 1e-8) + \
                            torch.log(torch.abs(term2) + 1e-8))
        log_det_jacobian = log_det_jacobian.squeeze(1)
        
        # Handle numerical issues
        log_det_jacobian = torch.where(
            torch.isnan(log_det_jacobian) | torch.isinf(log_det_jacobian),
            torch.zeros_like(log_det_jacobian),
            log_det_jacobian
        )
        
        return z, log_det_jacobian