import torch
import torch.nn as nn
from torchdiffeq import odeint
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
            nn.BatchNorm1d(self.hidden_dim),
            nn.ReLU(),
            MaskedLinear(self.hidden_dim, self.input_dim * self.output_dim_multiplier)
        ]
        layers[0].set_mask(self.masks[0])
        layers[3].set_mask(self.masks[1])
        
        # Initialize weights properly
        for layer in layers:
            if isinstance(layer, MaskedLinear):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)
        
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
        
        # Clamp alpha to prevent numerical instability
        alpha = torch.clamp(alpha, min=-10, max=10)

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
            
            # Clamp alpha to prevent numerical instability
            alpha = torch.clamp(alpha, min=-10, max=10)
            
            # Apply the transformation for the current dimension
            x_i = z[:, i:i+1] * torch.exp(alpha[:, i:i+1]) + mu[:, i:i+1]
            # Use clone to avoid in-place operations
            x = x.clone()
            x[:, i:i+1] = x_i

        # The log determinant is the sum of alpha, which we can get from one final pass
        final_params = self.conditioner(x)
        _, final_alpha = final_params.chunk(2, dim=1)
        final_alpha = torch.clamp(final_alpha, min=-10, max=10)
        log_det_jacobian = torch.sum(final_alpha, dim=1)
        
        return x, log_det_jacobian







class InverseAutoregressiveFlow(Flow):
    """
    Inverse Autoregressive Flow (IAF). This flow has the same expressiveness as
    MAF but with a reversed computational trade-off.
    The forward transformation (sampling) is fast and parallel.
    The inverse transformation (density evaluation) is slow and sequential.
    """
    def __init__(self, dim, hidden_dim=64):
        super().__init__()
        self.dim = dim
        # The conditioner network's autoregressive property is on z, not x.
        self.conditioner = MADE(dim, hidden_dim, 2)

    def forward(self, z):
        """
        Computes x = f(z). This direction is fast.
        x_i = z_i * exp(alpha_i) + mu_i, where mu_i and alpha_i are functions
        of z_1, ..., z_{i-1}.
        """
        params = self.conditioner(z)
        mu, alpha = params.chunk(2, dim=1)
        
        # Clamp alpha to prevent numerical instability
        alpha = torch.clamp(alpha, min=-10, max=10)
        
        # The transformation is parallel
        x = z * torch.exp(alpha) + mu
        
        # The log-determinant is also parallel
        log_det_jacobian = torch.sum(alpha, dim=1)
        
        return x, log_det_jacobian

    def inverse(self, x):
        """
        Computes z = g(x). This is slow and sequential.
        z_i = (x_i - mu_i) * exp(-alpha_i)
        """
        z = torch.zeros_like(x)
        log_det_jacobian = torch.zeros(x.size(0), device=x.device)

        # Iterate over each dimension
        for i in range(self.dim):
            # The conditioner's output depends on z_1, ..., z_{i-1}, which we have
            # already computed.
            params = self.conditioner(z)
            mu, alpha = params.chunk(2, dim=1)
            
            # Clamp alpha to prevent numerical instability
            alpha = torch.clamp(alpha, min=-10, max=10)
            
            # Apply the inverse transformation for the current dimension
            z_i = (x[:, i:i+1] - mu[:, i:i+1]) * torch.exp(-alpha[:, i:i+1])
            # Use scatter_ to update the i-th column without in-place operations
            z = z.clone()
            z[:, i:i+1] = z_i
            
        # The log determinant of the inverse is the negative sum of alpha.
        final_params = self.conditioner(z)
        _, final_alpha = final_params.chunk(2, dim=1)
        final_alpha = torch.clamp(final_alpha, min=-10, max=10)
        log_det_jacobian = -torch.sum(final_alpha, dim=1)

        return z, log_det_jacobian




class ODEFunc(nn.Module):
    """
    Defines the dynamics of the continuous flow.
    dz/dt = f(z, t)
    """
    def __init__(self, dim, hidden_dim):
        super(ODEFunc, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, dim)
        )

    def forward(self, t, z):
        """
        The forward pass of the ODE function.
        """
        # The network is independent of t, but the interface requires it.
        return self.net(z)

class ContinuousFlow(nn.Module):
    """
    Continuous Normalizing Flow model.
    """
    def __init__(self, dim, hidden_dim=64):
        super(ContinuousFlow, self).__init__()
        self.ode_func = ODEFunc(dim, hidden_dim)

    def forward(self, z, integration_times=torch.tensor([0.0, 1.0])):
        """
        Forward pass (sampling). Solves the ODE from t=0 to t=1.
        """
        # Ensure integration_times is valid for torchdiffeq
        if integration_times[0] == integration_times[1]:
            # If start and end times are the same, return the input unchanged
            return z
        
        # Ensure integration_times is strictly increasing or decreasing
        if integration_times[0] > integration_times[1]:
            # If decreasing, reverse the times and the result
            integration_times = torch.flip(integration_times, [0])
            reverse_result = True
        else:
            reverse_result = False
            
        x = odeint(
            self.ode_func,
            z,
            integration_times,
            method='dopri5',
            atol=1e-5,
            rtol=1e-5
        )
        
        result = x[-1]
        if reverse_result:
            # If we reversed the integration, we need to reverse the result
            # This is a simplified approach - in practice, you might need more sophisticated handling
            result = z  # For now, just return the original input
            
        return result

    def inverse(self, x, integration_times=torch.tensor([1.0, 0.0])):
        """
        Inverse pass (likelihood). Solves the ODE from t=1 to t=0.
        """
        # Ensure integration_times is valid for torchdiffeq
        if integration_times[0] == integration_times[1]:
            # If start and end times are the same, return the input unchanged
            return x
        
        # Ensure integration_times is strictly increasing or decreasing
        if integration_times[0] < integration_times[1]:
            # If increasing, reverse the times and the result
            integration_times = torch.flip(integration_times, [0])
            reverse_result = True
        else:
            reverse_result = False
            
        z = odeint(
            self.ode_func,
            x,
            integration_times,
            method='dopri5',
            atol=1e-5,
            rtol=1e-5
        )
        
        result = z[-1]
        if reverse_result:
            # If we reversed the integration, we need to reverse the result
            # This is a simplified approach - in practice, you might need more sophisticated handling
            result = x  # For now, just return the original input
            
        return result
    
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