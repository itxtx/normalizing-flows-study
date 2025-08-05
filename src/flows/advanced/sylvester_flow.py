"""
Sylvester Flow implementation.

This module implements Sylvester Normalizing Flows, which generalize planar flows
by using multiple transformation directions through orthogonal matrices constructed
from Householder reflections.

The transformation is of the form:
f(z) = z + QR^T tanh(RQz + b)

where:
- Q is an orthogonal matrix constructed from Householder reflections
- R is a learnable matrix
- b is a learnable bias vector
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List
from ..flow.flow import Flow


class HouseholderReflection(nn.Module):
    """
    Householder reflection matrix H = I - 2 * v * v^T / ||v||^2
    
    This creates an orthogonal matrix that reflects vectors across the
    hyperplane orthogonal to v.
    """
    
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        # Learnable reflection vector
        self.v = nn.Parameter(torch.randn(dim))
        self._initialize_parameters()
    
    def _initialize_parameters(self):
        """Initialize parameters for stability."""
        nn.init.normal_(self.v, mean=0, std=0.1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply Householder reflection: H * x = x - 2 * v * (v^T x) / ||v||^2
        
        Args:
            x: Input tensor [batch_size, dim] or [dim]
            
        Returns:
            Hx: Reflected tensor with same shape as input
        """
        # Normalize v to avoid numerical issues
        v_norm_sq = torch.sum(self.v ** 2) + 1e-8
        
        # Compute v^T x
        if x.dim() == 1:
            # Single vector case
            vTx = torch.dot(self.v, x)
            return x - 2 * self.v * vTx / v_norm_sq
        else:
            # Batch case
            vTx = torch.mv(x, self.v)  # [batch_size]
            return x - 2 * self.v.unsqueeze(0) * vTx.unsqueeze(1) / v_norm_sq
    
    def get_matrix(self) -> torch.Tensor:
        """
        Get the explicit Householder matrix H = I - 2 * v * v^T / ||v||^2
        
        Returns:
            H: Householder matrix [dim, dim]
        """
        v_norm_sq = torch.sum(self.v ** 2) + 1e-8
        I = torch.eye(self.dim, device=self.v.device, dtype=self.v.dtype)
        H = I - 2 * torch.outer(self.v, self.v) / v_norm_sq
        return H


class OrthogonalMatrix(nn.Module):
    """
    Orthogonal matrix constructed as a product of Householder reflections.
    
    Q = H_1 * H_2 * ... * H_k
    
    where each H_i is a Householder reflection.
    """
    
    def __init__(self, dim: int, num_householder: int = 8):
        super().__init__()
        self.dim = dim
        self.num_householder = min(num_householder, dim)  # Can't have more than dim reflections
        
        # Create Householder reflections
        self.reflections = nn.ModuleList([
            HouseholderReflection(dim) for _ in range(self.num_householder)
        ])
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply orthogonal transformation: Q * x
        
        Args:
            x: Input tensor [batch_size, dim] or [dim]
            
        Returns:
            Qx: Transformed tensor with same shape as input
        """
        result = x
        for reflection in self.reflections:
            result = reflection(result)
        return result
    
    def get_matrix(self) -> torch.Tensor:
        """
        Get the explicit orthogonal matrix Q = H_1 * H_2 * ... * H_k
        
        Returns:
            Q: Orthogonal matrix [dim, dim]
        """
        Q = torch.eye(self.dim, device=self.reflections[0].v.device, 
                     dtype=self.reflections[0].v.dtype)
        
        for reflection in self.reflections:
            H = reflection.get_matrix()
            Q = torch.mm(H, Q)
        
        return Q
    
    def log_det(self) -> torch.Tensor:
        """
        Compute log-determinant of the orthogonal matrix.
        
        For orthogonal matrices, |det(Q)| = 1, so log|det(Q)| = 0.
        However, det(Q) can be ±1, so we need to compute the actual determinant.
        
        Returns:
            log_det: Log-determinant (should be 0 for orthogonal matrices)
        """
        # For Householder reflections, det(H) = -1
        # So det(Q) = (-1)^num_householder
        det_sign = (-1) ** self.num_householder
        return torch.log(torch.abs(torch.tensor(det_sign, dtype=torch.float32)))


class SylvesterFlow(Flow):
    """
    Sylvester Normalizing Flow.
    
    This flow generalizes planar flows by using orthogonal matrices to create
    multiple transformation directions. The transformation is:
    
    f(z) = z + QR^T tanh(RQz + b)
    
    where:
    - Q is an orthogonal matrix (dim x dim) constructed from Householder reflections
    - R is a learnable matrix (M x dim) where M is the number of transformation directions
    - b is a learnable bias vector (M,)
    
    The log-determinant can be computed efficiently using the matrix determinant lemma.
    """
    
    def __init__(
        self,
        dim: int,
        num_householder: int = 8,
        num_transforms: int = None
    ):
        """
        Initialize Sylvester Flow.
        
        Args:
            dim: Dimensionality of the data
            num_householder: Number of Householder reflections for orthogonal matrix
            num_transforms: Number of transformation directions (default: dim)
        """
        super().__init__()
        self.data_dim = dim
        self.dim = dim
        self.num_householder = min(num_householder, dim)
        self.num_transforms = num_transforms if num_transforms is not None else dim
        
        # Orthogonal matrix Q constructed from Householder reflections
        self.Q = OrthogonalMatrix(dim, num_householder)
        
        # Learnable transformation matrix R
        self.R = nn.Parameter(torch.randn(self.num_transforms, dim))
        
        # Learnable bias vector
        self.b = nn.Parameter(torch.randn(self.num_transforms))
        
        # Initialize parameters
        self._initialize_parameters()
    
    def _initialize_parameters(self):
        """Initialize parameters for stability."""
        nn.init.xavier_normal_(self.R, gain=0.1)
        nn.init.normal_(self.b, mean=0, std=0.1)
    
    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward transformation: x = f(z) = z + QR^T tanh(RQz + b).
        
        Args:
            z: Input tensor from base distribution space [batch_size, dim]
            
        Returns:
            x: Transformed tensor in data space [batch_size, dim]
            log_det_jacobian: Log-determinant of Jacobian [batch_size]
        """
        batch_size = z.size(0)
        
        # Apply orthogonal transformation: Qz
        Qz = self.Q(z)  # [batch_size, dim]
        
        # Compute linear transformation: RQz + b
        linear_term = torch.mm(Qz, self.R.t()) + self.b.unsqueeze(0)  # [batch_size, num_transforms]
        
        # Apply activation
        activation = torch.tanh(linear_term)  # [batch_size, num_transforms]
        
        # Compute transformation: z + QR^T tanh(RQz + b)
        QRt_activation = torch.mm(activation, self.R)  # [batch_size, dim]
        QRt_activation = self.Q(QRt_activation)  # Apply Q again
        
        x = z + QRt_activation  # [batch_size, dim]
        
        # Compute log-determinant using matrix determinant lemma
        log_det_jacobian = self._compute_log_det(z, activation)
        
        return x, log_det_jacobian
    
    def inverse(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Inverse transformation: z = f^(-1)(x).
        
        Since Sylvester flows don't have a closed-form inverse, we use
        fixed-point iteration to solve: x = z + QR^T tanh(RQz + b) for z.
        
        Args:
            x: Input tensor from data space [batch_size, dim]
            
        Returns:
            z: Transformed tensor in base distribution space [batch_size, dim]
            log_det_jacobian: Log-determinant of Jacobian [batch_size]
        """
        batch_size = x.size(0)
        device = x.device
        dtype = x.dtype
        
        # Initialize z with x (good starting point)
        z = x.clone()
        
        # Fixed-point iteration
        max_iter = 50
        tol = 1e-6
        
        for i in range(max_iter):
            # Compute current transformation
            Qz = self.Q(z)
            linear_term = torch.mm(Qz, self.R.t()) + self.b.unsqueeze(0)
            activation = torch.tanh(linear_term)
            QRt_activation = torch.mm(activation, self.R)
            QRt_activation = self.Q(QRt_activation)
            
            z_new = x - QRt_activation
            
            # Check convergence
            diff = torch.norm(z_new - z, dim=1)
            if torch.all(diff < tol):
                break
            
            z = z_new
        
        # Compute log-determinant (negative of forward)
        Qz = self.Q(z)
        linear_term = torch.mm(Qz, self.R.t()) + self.b.unsqueeze(0)
        activation = torch.tanh(linear_term)
        log_det_jacobian = -self._compute_log_det(z, activation)
        
        return z, log_det_jacobian
    
    def _compute_log_det(self, z: torch.Tensor, activation: torch.Tensor) -> torch.Tensor:
        """
        Compute log-determinant of the Jacobian using matrix determinant lemma.
        
        For the transformation f(z) = z + QR^T tanh(RQz + b), the Jacobian is:
        J = I + QR^T diag(1 - tanh^2(RQz + b)) RQ
        
        Using the matrix determinant lemma:
        det(I + UV^T) = det(I + V^T U)
        
        where U = QR^T diag(psi) and V^T = RQ, so V^T U = RQ QR^T diag(psi) = RR^T diag(psi)
        
        Args:
            z: Input tensor [batch_size, dim]
            activation: Activation values tanh(RQz + b) [batch_size, num_transforms]
            
        Returns:
            log_det: Log-determinant for each sample [batch_size]
        """
        batch_size = z.size(0)
        
        # Compute derivative of tanh: psi = 1 - tanh^2
        psi = 1 - activation ** 2  # [batch_size, num_transforms]
        
        # For efficiency, we use the fact that Q is orthogonal (Q^T Q = I)
        # So RQ QR^T = RR^T, and we need det(I + RR^T diag(psi))
        
        # Compute RR^T
        RRt = torch.mm(self.R, self.R.t())  # [num_transforms, num_transforms]
        
        # For each sample in the batch, compute det(I + RR^T diag(psi_i))
        log_dets = []
        I = torch.eye(self.num_transforms, device=z.device, dtype=z.dtype)
        
        for i in range(batch_size):
            # Create diagonal matrix from psi[i]
            psi_diag = torch.diag(psi[i])  # [num_transforms, num_transforms]
            
            # Compute I + RR^T * diag(psi_i)
            matrix = I + torch.mm(RRt, psi_diag)
            
            # Compute determinant
            det = torch.det(matrix)
            log_det = torch.log(torch.abs(det) + 1e-8)
            log_dets.append(log_det)
        
        log_det_jacobian = torch.stack(log_dets)
        
        # Handle numerical issues
        log_det_jacobian = torch.where(
            torch.isnan(log_det_jacobian) | torch.isinf(log_det_jacobian),
            torch.zeros_like(log_det_jacobian),
            log_det_jacobian
        )
        
        return log_det_jacobian
    
    def get_orthogonal_matrix(self) -> torch.Tensor:
        """
        Get the explicit orthogonal matrix Q.
        
        Returns:
            Q: Orthogonal matrix [dim, dim]
        """
        return self.Q.get_matrix()
    
    def check_orthogonality(self) -> torch.Tensor:
        """
        Check orthogonality constraint: Q^T Q should be identity.
        
        Returns:
            error: Frobenius norm of (Q^T Q - I)
        """
        Q = self.get_orthogonal_matrix()
        QtQ = torch.mm(Q.t(), Q)
        I = torch.eye(self.dim, device=Q.device, dtype=Q.dtype)
        error = torch.norm(QtQ - I, p='fro')
        return error