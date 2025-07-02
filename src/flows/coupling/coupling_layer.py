import torch
import torch.nn as nn
from src.flows.flow.flow import Flow

class CouplingLayer(Flow):
    """
    Implements a single coupling layer from Real NVP.
    """
    def __init__(self, data_dim, hidden_dim, mask):
        super().__init__()
        
        self.register_buffer('mask', mask)

        # The conditioner networks for scale (s) and bias (b)
        # These should be simple MLPs that take the "control" part of the input
        # and output the parameters for the other part.
        self.s_net = nn.Sequential(
            nn.Linear(data_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, data_dim)
        )
        self.b_net = nn.Sequential(
            nn.Linear(data_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, data_dim)
        )
        
        # Initialize weights properly to prevent exploding gradients
        self._initialize_weights()

    def forward(self, z):
        """
        Computes the forward pass x = f(z).
        z -> x
        """
        # The mask determines which part is transformed (mask == 1) 
        # and which part is the identity (mask == 0).
        z_a = z * self.mask  # This is z_A in the equations, but with zeros for z_B
        
        # The conditioner networks only see the identity part
        s = torch.clamp(self.s_net(z_a), min=-10.0, max=10.0)
        b = torch.clamp(self.b_net(z_a), min=-10.0, max=10.0)
        
        # Apply the transformation to the other part
        # z_b is selected by (1 - mask)
        x = z_a + (1 - self.mask) * (z * torch.exp(s) + b)
        
        # The log-determinant of the Jacobian
        log_det_J = ((1 - self.mask) * s).sum(dim=1)

        # Safety check: ensure x doesn't contain NaN or infinite values
        x = torch.where(torch.isnan(x) | torch.isinf(x), torch.zeros_like(x), x)
        log_det_J = torch.where(
            torch.isnan(log_det_J) | torch.isinf(log_det_J),
            torch.zeros_like(log_det_J),
            log_det_J
        )

        return x, log_det_J

    def inverse(self, x):
        """
        Computes the inverse pass z = g(x).
        x -> z
        """
        # The mask determines which part was the identity
        x_a = x * self.mask # This is x_A in the equations
        
        # The conditioner networks see the identity part
        s = torch.clamp(self.s_net(x_a), min=-10.0, max=10.0)
        b = torch.clamp(self.b_net(x_a), min=-10.0, max=10.0)
        
        # Apply the inverse transformation to the other part
        z = x_a + (1 - self.mask) * ((x - b) * torch.exp(-s))
        
        # The log-determinant of the inverse Jacobian
        log_det_J_inv = ((1 - self.mask) * -s).sum(dim=1)

        # Safety check: ensure z doesn't contain NaN or infinite values
        z = torch.where(torch.isnan(z) | torch.isinf(z), torch.zeros_like(z), z)
        log_det_J_inv = torch.where(
            torch.isnan(log_det_J_inv) | torch.isinf(log_det_J_inv),
            torch.zeros_like(log_det_J_inv),
            log_det_J_inv
        )

        return z, log_det_J_inv
    
    def _initialize_weights(self):
        for net in [self.s_net, self.b_net]:
            # Initialize all but the final layer
            for layer in net[:-1]:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_normal_(layer.weight, gain=1.0) # Use a standard gain
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)
        
        # Initialize final layers to output zeros, making the flow initially identity
        nn.init.zeros_(self.s_net[-1].weight)
        nn.init.zeros_(self.s_net[-1].bias)
        nn.init.zeros_(self.b_net[-1].weight)
        nn.init.zeros_(self.b_net[-1].bias)
