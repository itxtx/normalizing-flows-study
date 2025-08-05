"""
Residual Flow implementation with Lipschitz constraints.

This module implements a Residual Flow that ensures invertibility through
spectral normalization and Lipschitz constraints. The flow uses fixed-point
iteration for inverse computation and Neumann series approximation for
log-determinant calculation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from ..flow.flow import Flow


class SpectralNorm(nn.Module):
    """
    Spectral normalization layer to enforce Lipschitz constraints.
    
    This implementation uses power iteration to compute the spectral norm
    and normalizes the weight matrix to have spectral norm <= lipschitz_constant.
    """
    
    def __init__(self, module: nn.Module, lipschitz_constant: float = 0.9, n_power_iterations: int = 1):
        super().__init__()
        self.module = module
        self.lipschitz_constant = lipschitz_constant
        self.n_power_iterations = n_power_iterations
        
        # Get weight tensor
        if hasattr(module, 'weight'):
            weight = module.weight
        else:
            raise ValueError("Module must have a 'weight' attribute")
        
        # Initialize u and v vectors for power iteration
        self.register_buffer('u', torch.randn(weight.size(0)))
        self.register_buffer('v', torch.randn(weight.size(1)))
        
        # Normalize initial vectors
        self.u.data = F.normalize(self.u.data, dim=0)
        self.v.data = F.normalize(self.v.data, dim=0)
    
    def forward(self, *args, **kwargs):
        """Apply spectral normalization and forward through module."""
        weight = self.module.weight
        
        # Power iteration to compute spectral norm
        u = self.u
        v = self.v
        
        for _ in range(self.n_power_iterations):
            v = F.normalize(torch.mv(weight.t(), u), dim=0)
            u = F.normalize(torch.mv(weight, v), dim=0)
        
        # Compute spectral norm
        sigma = torch.dot(u, torch.mv(weight, v))
        
        # Normalize weight matrix
        if sigma > self.lipschitz_constant:
            weight_normalized = weight * (self.lipschitz_constant / sigma)
        else:
            weight_normalized = weight
        
        # Temporarily replace weight data
        original_weight_data = self.module.weight.data.clone()
        self.module.weight.data = weight_normalized
        
        # Forward pass
        output = self.module(*args, **kwargs)
        
        # Restore original weight data
        self.module.weight.data = original_weight_data
        
        # Update u and v buffers
        if self.training:
            self.u.data = u
            self.v.data = v
        
        return output


class ResidualBlock(nn.Module):
    """
    Residual block with spectral normalization for Lipschitz constraint.
    
    The block computes g(x) = x + f(x) where f has Lipschitz constant < 1,
    ensuring that g is invertible.
    """
    
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        lipschitz_constant: float = 0.9,
        activation: str = "relu"
    ):
        super().__init__()
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.lipschitz_constant = lipschitz_constant
        
        # Activation function
        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "elu":
            self.activation = nn.ELU()
        elif activation == "leaky_relu":
            self.activation = nn.LeakyReLU(0.1)
        elif activation == "tanh":
            self.activation = nn.Tanh()
        else:
            raise ValueError(f"Unsupported activation: {activation}")
        
        # Network layers with spectral normalization
        self.layer1 = SpectralNorm(
            nn.Linear(dim, hidden_dim),
            lipschitz_constant=lipschitz_constant / 2
        )
        self.layer2 = SpectralNorm(
            nn.Linear(hidden_dim, hidden_dim),
            lipschitz_constant=lipschitz_constant / 2
        )
        self.layer3 = SpectralNorm(
            nn.Linear(hidden_dim, dim),
            lipschitz_constant=lipschitz_constant / 2
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights for stability."""
        for module in [self.layer1.module, self.layer2.module, self.layer3.module]:
            nn.init.xavier_normal_(module.weight, gain=0.1)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: g(x) = x + f(x)."""
        residual = x
        
        out = self.layer1(x)
        out = self.activation(out)
        out = self.layer2(out)
        out = self.activation(out)
        out = self.layer3(out)
        
        return residual + out


class ResidualFlow(Flow):
    """
    Residual Flow with Lipschitz constraints for guaranteed invertibility.
    
    This flow implements the transformation g(x) = x + f(x) where f is a neural
    network with Lipschitz constant < 1. This ensures that g is invertible and
    can be inverted using fixed-point iteration.
    
    The log-determinant is approximated using the Neumann series:
    log|det(I + J_f)| ≈ tr(J_f) - tr(J_f^2)/2 + tr(J_f^3)/3 - ...
    """
    
    def __init__(
        self,
        dim: int,
        hidden_dim: int = 64,
        lipschitz_constant: float = 0.9,
        activation: str = "relu",
        max_iter: int = 100,
        tol: float = 1e-6,
        neumann_terms: int = 3
    ):
        """
        Initialize Residual Flow.
        
        Args:
            dim: Dimensionality of the data
            hidden_dim: Hidden layer dimension
            lipschitz_constant: Lipschitz constant for the residual function
            activation: Activation function
            max_iter: Maximum iterations for fixed-point iteration
            tol: Tolerance for fixed-point iteration
            neumann_terms: Number of terms in Neumann series approximation
        """
        super().__init__()
        self.data_dim = dim
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.lipschitz_constant = lipschitz_constant
        self.max_iter = max_iter
        self.tol = tol
        self.neumann_terms = neumann_terms
        
        # Residual block
        self.residual_block = ResidualBlock(
            dim=dim,
            hidden_dim=hidden_dim,
            lipschitz_constant=lipschitz_constant,
            activation=activation
        )
    
    def _residual_function(self, x: torch.Tensor) -> torch.Tensor:
        """Compute the residual function f(x)."""
        return self.residual_block(x) - x
    
    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward transformation: x = g(z) = z + f(z).
        
        Args:
            z: Input tensor from base distribution space
            
        Returns:
            x: Transformed tensor in data space
            log_det_jacobian: Log-determinant of Jacobian
        """
        # Forward pass is simple: x = z + f(z)
        f_z = self._residual_function(z)
        x = z + f_z
        
        # Compute log-determinant using Neumann series
        log_det_jacobian = self._compute_log_det_neumann(z)
        
        return x, log_det_jacobian
    
    def inverse(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Inverse transformation: z = g^(-1)(x) using fixed-point iteration.
        
        We solve z + f(z) = x for z using fixed-point iteration:
        z_{k+1} = x - f(z_k)
        
        Args:
            x: Input tensor from data space
            
        Returns:
            z: Transformed tensor in base distribution space
            log_det_jacobian: Log-determinant of Jacobian
        """
        batch_size = x.size(0)
        device = x.device
        dtype = x.dtype
        
        # Initialize z with x (good starting point)
        z = x.clone()
        
        # Fixed-point iteration: z = x - f(z)
        for i in range(self.max_iter):
            z_new = x - self._residual_function(z)
            
            # Check convergence
            diff = torch.norm(z_new - z, dim=1)
            if torch.all(diff < self.tol):
                break
            
            z = z_new
        
        # Compute log-determinant (negative of forward)
        log_det_jacobian = -self._compute_log_det_neumann(z)
        
        return z, log_det_jacobian
    
    def _compute_log_det_neumann(self, z: torch.Tensor) -> torch.Tensor:
        """
        Compute log-determinant using Neumann series approximation.
        
        For g(z) = z + f(z), we have J_g = I + J_f.
        The log-determinant is approximated as:
        log|det(I + J_f)| ≈ tr(J_f) - tr(J_f^2)/2 + tr(J_f^3)/3 - ...
        
        Args:
            z: Input tensor
            
        Returns:
            log_det: Log-determinant for each sample in the batch
        """
        batch_size = z.size(0)
        device = z.device
        
        # Compute Jacobian of f with respect to z
        z_flat = z.view(batch_size, -1)
        jacobian = self._compute_jacobian(z_flat)
        
        # Neumann series approximation
        log_det = torch.zeros(batch_size, device=device)
        
        # Current power of Jacobian
        jac_power = jacobian
        
        for k in range(1, self.neumann_terms + 1):
            # Add k-th term: (-1)^(k+1) * tr(J^k) / k
            trace_term = torch.diagonal(jac_power, dim1=-2, dim2=-1).sum(dim=-1)
            log_det += ((-1) ** (k + 1)) * trace_term / k
            
            # Update Jacobian power for next iteration
            if k < self.neumann_terms:
                jac_power = torch.bmm(jac_power, jacobian)
        
        return log_det
    
    def _compute_jacobian(self, z: torch.Tensor) -> torch.Tensor:
        """
        Compute Jacobian of the residual function f.
        
        Args:
            z: Input tensor [batch_size, dim]
            
        Returns:
            jacobian: Jacobian tensor [batch_size, dim, dim]
        """
        batch_size, dim = z.shape
        device = z.device
        
        jacobian = torch.zeros(batch_size, dim, dim, device=device)
        
        for i in range(dim):
            # Create unit vector for i-th dimension
            e_i = torch.zeros_like(z)
            e_i[:, i] = 1.0
            
            # Compute gradient of f_i with respect to z
            z_copy = z.clone().requires_grad_(True)
            f_z = self._residual_function(z_copy)
            
            grad_outputs = torch.zeros_like(f_z)
            grad_outputs[:, i] = 1.0
            
            grad = torch.autograd.grad(
                outputs=f_z,
                inputs=z_copy,
                grad_outputs=grad_outputs,
                create_graph=True,
                retain_graph=True
            )[0]
            
            jacobian[:, i, :] = grad
        
        return jacobian