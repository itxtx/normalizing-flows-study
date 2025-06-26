import torch
import torch.nn as nn
import numpy as np

class MaskedLinear(nn.Linear):
    """
    A linear layer with a mask to enforce autoregressive properties.
    """
    def __init__(self, in_features, out_features, bias=True):
        super().__init__(in_features, out_features, bias)
        self.register_buffer('mask', torch.ones(out_features, in_features))

    def set_mask(self, mask):
        """
        Sets the mask for the linear layer's weights.
        """
        self.mask.data.copy_(torch.from_numpy(mask.astype(np.uint8)))

    def forward(self, input):
        """
        Applies the masked linear transformation.
        """
        return nn.functional.linear(input, self.mask * self.weight, self.bias)

class MADE(nn.Module):
    """
    Masked Autoencoder for Distribution Estimation (Conditioner Network).
    This network is autoregressive. For a given input x, the output for
    dimension i is only dependent on inputs x_j where j < i.
    """
    def __init__(self, input_dim, hidden_dim, output_dim_multiplier=2):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim_multiplier = output_dim_multiplier

        # Assign degrees to input, hidden, and output neurons
        self.m = {}
        self.m[-1] = np.arange(self.input_dim)
        self.m[0] = np.random.randint(0, self.input_dim - 1, size=self.hidden_dim)
        self.m[1] = np.arange(self.input_dim)

        self.masks = self.create_masks()
        self.net = self.create_network()

    def create_masks(self):
        """
        Creates the masks for the linear layers.
        The mask for a connection from layer i to j ensures that an output unit
        can only be connected to input units with a strictly smaller degree.
        """
        masks = []
        # Mask for input to hidden layer: shape should be (hidden_dim, input_dim)
        # m_k(x_j) <= m_{k-1}(x_i)
        mask1 = (self.m[-1][:, np.newaxis] <= self.m[0][np.newaxis, :]).T
        masks.append(mask1)
        
        # Mask for hidden to output layer: shape should be (input_dim * output_dim_multiplier, hidden_dim)
        # m_D(y_j) > m_k(x_i)
        # We need to repeat the mask for each output dimension
        base_mask = (self.m[0][:, np.newaxis] < self.m[1][np.newaxis, :]).T
        # Repeat the mask for each output dimension multiplier
        mask2 = np.repeat(base_mask, self.output_dim_multiplier, axis=0)
        masks.append(mask2)
        return masks

    def create_network(self):
        """
        Creates the neural network with masked linear layers.
        """
        layers = [
            MaskedLinear(self.input_dim, self.hidden_dim),
            nn.ReLU(),
            MaskedLinear(self.hidden_dim, self.input_dim * self.output_dim_multiplier)
        ]
        layers[0].set_mask(self.masks[0])
        layers[2].set_mask(self.masks[1])
        return nn.Sequential(*layers)

    def forward(self, x):
        """
        The forward pass of the conditioner network.
        """
        # The output contains the parameters for the transformation (e.g., mu and alpha)
        return self.net(x)

class Flow(nn.Module):
    """
    Base class for normalizing flow layers.
    """
    def __init__(self):
        super(Flow, self).__init__()

    def forward(self, z):
        """
        Computes the forward transformation x = f(z) and the
        log-determinant of the Jacobian of the forward transformation.

        Args:
            z (torch.Tensor): The input tensor from the base distribution.

        Returns:
            torch.Tensor: The transformed tensor in the data space.
            torch.Tensor: The log-determinant of the Jacobian of f.
        """
        raise NotImplementedError

    def inverse(self, x):
        """
        Computes the inverse transformation z = g(x) and the
        log-determinant of the Jacobian of the inverse transformation.

        Args:
            x (torch.Tensor): The input tensor from the data space.

        Returns:
            torch.Tensor: The transformed tensor in the base distribution space.
            torch.Tensor: The log-determinant of the Jacobian of g.
        """
        raise NotImplementedError
    

class CouplingLayer(Flow):
    """
    Implements a single coupling layer from Real NVP.
    """
    def __init__(self, data_dim, hidden_dim, mask):
        super(CouplingLayer, self).__init__()
        
        self.mask = mask

        # The conditioner networks for scale (s) and bias (b)
        # These should be simple MLPs that take the "control" part of the input
        # and output the parameters for the other part.
        self.s_net = nn.Sequential(
            nn.Linear(data_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, data_dim)
        )
        self.b_net = nn.Sequential(
            nn.Linear(data_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, data_dim)
        )

    def forward(self, z):
        """
        Computes the forward pass x = f(z).
        z -> x
        """
        # The mask determines which part is transformed (mask == 1) 
        # and which part is the identity (mask == 0).
        z_a = z * self.mask  # This is z_A in the equations, but with zeros for z_B
        
        # The conditioner networks only see the identity part
        s = self.s_net(z_a)
        b = self.b_net(z_a)
        
        # Apply the transformation to the other part
        # z_b is selected by (1 - mask)
        x = z_a + (1 - self.mask) * (z * torch.exp(s) + b)
        
        # The log-determinant is the sum of s for the transformed dimensions
        log_det_J = ((1 - self.mask) * s).sum(dim=1)
        
        return x, log_det_J

    def inverse(self, x):
        """
        Computes the inverse pass z = g(x).
        x -> z
        """
        # The mask determines which part was the identity
        x_a = x * self.mask # This is x_A in the equations
        
        # The conditioner networks see the identity part
        s = self.s_net(x_a)
        b = self.b_net(x_a)
        
        # Apply the inverse transformation to the other part
        z = x_a + (1 - self.mask) * ((x - b) * torch.exp(-s))
        
        # The log-determinant of the inverse Jacobian
        log_det_J_inv = ((1 - self.mask) * -s).sum(dim=1)

        return z, log_det_J_inv
    
    



class MaskedAutoregressiveFlow(Flow):
    """
    Masked Autoregressive Flow (MAF). This flow is universal, meaning it can
    approximate any density.
    The inverse transformation (density evaluation) is fast and parallel.
    The forward transformation (sampling) is slow and sequential.
    """
    def __init__(self, dim, hidden_dim=64):
        super().__init__()
        self.dim = dim
        # The conditioner network that outputs mu and alpha
        self.conditioner = MADE(dim, hidden_dim, 2)
        
    def inverse(self, x):
        """
        Computes z = g(x). This direction is fast.
        z_i = (x_i - mu_i) * exp(-alpha_i), where mu_i and alpha_i are functions
        of x_1, ..., x_{i-1}.
        """
        # Get parameters mu and alpha from the conditioner
        params = self.conditioner(x)
        mu, alpha = params.chunk(2, dim=1)

        # Apply the transformation
        z = (x - mu) * torch.exp(-alpha)
        
        # The log-determinant of the Jacobian is the sum of the log of the scaling factors.
        log_det_jacobian = -torch.sum(alpha, dim=1)
        
        return z, log_det_jacobian

    def forward(self, z):
        """
        Computes x = f(z). This direction is slow as it must be done sequentially.
        x_i = z_i * exp(alpha_i) + mu_i
        """
        x = torch.zeros_like(z)
        log_det_jacobian = torch.zeros(z.size(0), device=z.device)

        # Iterate over each dimension to generate the sample
        for i in range(self.dim):
            # The conditioner's output for dim i depends only on x_1, ..., x_{i-1}
            params = self.conditioner(x)
            mu, alpha = params.chunk(2, dim=1)
            
            # Apply the transformation for the current dimension
            x[:, i] = z[:, i] * torch.exp(alpha[:, i]) + mu[:, i]

        # The log determinant is the sum of alpha, which we can get from one final pass
        final_params = self.conditioner(x)
        _, final_alpha = final_params.chunk(2, dim=1)
        log_det_jacobian = torch.sum(final_alpha, dim=1)
        
        return x, log_det_jacobian







    
class PlanarFlow(Flow):
    """
    Implements a planar flow layer as described in "Variational Inference with
    Normalizing Flows" by Rezende & Mohamed (2015).
    """
    def __init__(self, dim):
        super(PlanarFlow, self).__init__()
        self.u = nn.Parameter(torch.randn(1, dim))
        self.w = nn.Parameter(torch.randn(1, dim))
        self.b = nn.Parameter(torch.randn(1))

    def forward(self, z):
        """
        Computes x = z + u * h(w^T * z + b)
        """
        # Ensure invertibility condition
        uw = torch.matmul(self.u, self.w.t())
        u_hat = self.u + (-1 + torch.log(1 + torch.exp(uw))) - uw
        
        # Compute the transformation
        wz_b = torch.matmul(z, self.w.t()) + self.b
        x = z + u_hat * torch.tanh(wz_b)
        
        # Compute the log-determinant of the Jacobian
        psi = (1 - torch.tanh(wz_b)**2) * self.w
        log_det_jacobian = torch.log(torch.abs(1 + torch.matmul(psi, u_hat.t())))
        
        return x, log_det_jacobian

    def inverse(self, x):
        """
        Planar flow does not have a simple analytical inverse.
        """
        raise NotImplementedError

