import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.utils import spectral_norm
from torchdiffeq import odeint
import numpy as np

class MaskedLinear(nn.Linear):
    """A linear layer with a fixed binary mask."""
    
    def __init__(self, in_features, out_features, mask, bias=True):
        # Call the parent constructor
        super().__init__(in_features, out_features, bias)
        
        # Register the mask as a buffer to ensure it moves with the model (e.g., to GPU)
        self.register_buffer('mask', mask)

    def forward(self, input):
        # Apply the mask to the weights during the forward pass
        # Ensure mask has the same dtype as weights
        mask = self.mask.to(dtype=self.weight.dtype)
        return F.linear(input, self.weight * mask, self.bias)

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

        # Debug: print degree assignments and masks for input_dim=2
        # if self.input_dim == 2:
        #     print("[MADE DEBUG] Degree assignments:")
        #     print("  Input:", self.m[-1])
        #     print("  Hidden:", self.m[0])
        #     print("  Output:", self.m[1])

        # Ensure proper autoregressive structure
        # Each output dimension i should only depend on input dimensions < i
        self.masks = self.create_masks()

        # if self.input_dim == 2:
        #     print("[MADE DEBUG] Input-to-hidden mask (shape {}):".format(self.masks[0].shape))
        #     print(self.masks[0].numpy())
        #     print("[MADE DEBUG] Hidden-to-output mask (shape {}):".format(self.masks[1].shape))
        #     print(self.masks[1].numpy())

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
        Creates a deeper neural network with masked linear layers.
        """
        # Create a deeper network for better expressiveness
        layers = []
        
        # First layer: input to hidden
        first_layer = MaskedLinear(
            self.input_dim, 
            self.hidden_dim, 
            mask=self.masks[0]
        )
        layers.append(first_layer)
        layers.append(nn.ReLU())
        
        # Second hidden layer (fully connected within the autoregressive constraint)
        second_layer = nn.Linear(self.hidden_dim, self.hidden_dim)
        layers.append(second_layer)
        layers.append(nn.ReLU())
        
        # Third hidden layer
        third_layer = nn.Linear(self.hidden_dim, self.hidden_dim)
        layers.append(third_layer)
        layers.append(nn.ReLU())
        
        # Final layer: hidden to output (masked)
        final_layer = MaskedLinear(
            self.hidden_dim, 
            self.input_dim * self.output_dim_multiplier, 
            mask=self.masks[1]
        )
        layers.append(final_layer)
        
        # Better initialization for learning meaningful transformations
        nn.init.xavier_normal_(first_layer.weight, gain=1.0)
        if first_layer.bias is not None:
            nn.init.zeros_(first_layer.bias)
        
        nn.init.xavier_normal_(second_layer.weight, gain=1.0)
        if second_layer.bias is not None:
            nn.init.zeros_(second_layer.bias)
            
        nn.init.xavier_normal_(third_layer.weight, gain=1.0)
        if third_layer.bias is not None:
            nn.init.zeros_(third_layer.bias)
        
        # Initialize final layer with larger weights to allow expressive transformations
        nn.init.normal_(final_layer.weight, mean=0.0, std=0.05)  # Smaller std for stability
        if final_layer.bias is not None:
            # Use small random values instead of zeros to allow learning
            nn.init.normal_(final_layer.bias, mean=0.0, std=0.005)  # Smaller std for stability
        
        return nn.Sequential(*layers)
    
    def forward(self, x):
        """
        The forward pass of the conditioner network.
        """
        # The output contains the parameters for the transformation (e.g., mu and alpha)
        output = self.net(x)
        # Much tighter clamping to prevent extreme values that cause numerical instability
        return torch.clamp(output, min=-3.0, max=3.0)

class Flow(nn.Module):
    """
    Base class for normalizing flow layers.
    """
    def __init__(self):
        super().__init__()

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
        super().__init__()
        
        self.register_buffer('mask', mask)

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
        
        # Much tighter clamping to prevent numerical instability
        alpha = torch.clamp(alpha, min=-3, max=3)

        # Apply the transformation with better numerical stability
        # Use log-space operations to avoid overflow
        log_scale = -alpha
        scale = torch.exp(torch.clamp(log_scale, min=-5, max=5))
        z = (x - mu) * scale
        
        # The log-determinant of the Jacobian is the sum of the log of the scaling factors.
        log_det_jacobian = -torch.sum(alpha, dim=1)
        
        # Safety check: ensure z doesn't contain NaN or infinite values
        z = torch.where(torch.isnan(z) | torch.isinf(z), torch.zeros_like(z), z)
        log_det_jacobian = torch.where(
            torch.isnan(log_det_jacobian) | torch.isinf(log_det_jacobian),
            torch.zeros_like(log_det_jacobian),
            log_det_jacobian
        )
        
        # Additional safety: clamp log_det to prevent extreme values
        log_det_jacobian = torch.clamp(log_det_jacobian, min=-100, max=100)
        
        return z, log_det_jacobian

    def forward(self, z):
        """
        Computes x = f(z). This direction is slow as it must be done sequentially.
        x_i = z_i * exp(alpha_i) + mu_i
        """
        batch_size = z.size(0)
        x = torch.zeros(batch_size, self.dim, device=z.device, dtype=z.dtype)
        log_det_jacobian = torch.zeros(batch_size, device=z.device, dtype=z.dtype)

        # Iterate over each dimension to generate the sample
        for i in range(self.dim):
            # Get parameters from the conditioner using the current x (which is partially filled)
            params = self.conditioner(x)
            mu, alpha = params.chunk(2, dim=1)
            
            # Much tighter clamping to prevent numerical instability
            alpha = torch.clamp(alpha, min=-3, max=3)
            
            # Apply the transformation for the current dimension with better numerical stability
            log_scale = alpha[:, i]
            scale = torch.exp(torch.clamp(log_scale, min=-5, max=5))
            # Use clone to avoid inplace operations
            x_new = x.clone()
            x_new[:, i] = z[:, i] * scale + mu[:, i]
            x = x_new
            
            log_det_jacobian += alpha[:, i]

        # Safety check: ensure x doesn't contain NaN or infinite values
        x = torch.where(torch.isnan(x) | torch.isinf(x), torch.zeros_like(x), x)
        log_det_jacobian = torch.where(
            torch.isnan(log_det_jacobian) | torch.isinf(log_det_jacobian),
            torch.zeros_like(log_det_jacobian),
            log_det_jacobian
        )
        
        # Additional safety: clamp log_det to prevent extreme values
        log_det_jacobian = torch.clamp(log_det_jacobian, min=-100, max=100)
        
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
        Computes x = f(z). This direction is fast and parallel.
        x_i = z_i * exp(alpha_i) + mu_i, where mu_i and alpha_i are functions
        of z_1, ..., z_{i-1}.
        """
        params = self.conditioner(z)
        mu, alpha = params.chunk(2, dim=1)
        
        # Much tighter clamping to prevent numerical instability
        alpha = torch.clamp(alpha, min=-3, max=3)
        
        # The transformation is parallel with better numerical stability
        log_scale = alpha
        scale = torch.exp(torch.clamp(log_scale, min=-5, max=5))
        x = z * scale + mu
        
        # The log-determinant is also parallel
        log_det_jacobian = torch.sum(alpha, dim=1)
        
        # Safety check: ensure x doesn't contain NaN or infinite values
        x = torch.where(torch.isnan(x) | torch.isinf(x), torch.zeros_like(x), x)
        log_det_jacobian = torch.where(
            torch.isnan(log_det_jacobian) | torch.isinf(log_det_jacobian),
            torch.zeros_like(log_det_jacobian),
            log_det_jacobian
        )
        
        # Additional safety: clamp log_det to prevent extreme values
        log_det_jacobian = torch.clamp(log_det_jacobian, min=-100, max=100)
        
        return x, log_det_jacobian

    def inverse(self, x):
        """
        Computes z = g(x). This is slow and sequential.
        z_i = (x_i - mu_i) * exp(-alpha_i)
        """
        batch_size = x.size(0)
        z = torch.zeros(batch_size, self.dim, device=x.device, dtype=x.dtype)
        log_det_jacobian = torch.zeros(batch_size, device=x.device, dtype=x.dtype)

        # Iterate over each dimension
        for i in range(self.dim):
            # Get parameters from the conditioner using the current z (which is partially filled)
            params = self.conditioner(z)
            mu, alpha = params.chunk(2, dim=1)
            
            # Much tighter clamping to prevent numerical instability
            alpha = torch.clamp(alpha, min=-3, max=3)
            
            # Apply the inverse transformation for the current dimension with better numerical stability
            log_scale = -alpha[:, i]
            scale = torch.exp(torch.clamp(log_scale, min=-5, max=5))
            # Use scatter to avoid inplace operations
            z_new = z.clone()
            z_new[:, i] = (x[:, i] - mu[:, i]) * scale
            z = z_new
            
            # Accumulate the log-determinant from this step's alpha
            log_det_jacobian -= alpha[:, i]

        # Safety check: ensure z doesn't contain NaN or infinite values
        z = torch.where(torch.isnan(z) | torch.isinf(z), torch.zeros_like(z), z)
        log_det_jacobian = torch.where(
            torch.isnan(log_det_jacobian) | torch.isinf(log_det_jacobian),
            torch.zeros_like(log_det_jacobian),
            log_det_jacobian
        )

        # Additional safety: clamp log_det to prevent extreme values
        log_det_jacobian = torch.clamp(log_det_jacobian, min=-100, max=100)

        return z, log_det_jacobian




class ODEFunc(nn.Module):
    """
    Defines the dynamics of the continuous flow with augmented state.
    The augmented state includes both the input z and the log-determinant.
    """
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.dim = dim
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, dim)
        )
        
        # Initialize weights properly to prevent exploding gradients
        self._initialize_weights()

    def forward(self, t, augmented_state):
        """
        The forward pass of the ODE function.
        
        Args:
            t: time (not used in this implementation)
            augmented_state: concatenated tensor [z, log_det] where z is the state
                           and log_det is the accumulated log-determinant
        
        Returns:
            concatenated tensor [dz/dt, dlog_det/dt]
        """
        # Split the augmented state
        z = augmented_state[:, :self.dim]
        log_det = augmented_state[:, self.dim:]
        
        # Compute the dynamics dz/dt = f(z, t)
        dz_dt = self.net(z)
        
        # Compute the trace of the Jacobian using Hutchinson's estimator
        # Use multiple random vectors for better accuracy
        batch_size = z.size(0)
        device = z.device
        
        # Use 3 random vectors for better accuracy (trade-off between speed and accuracy)
        num_vectors = 3
        trace_estimate = 0.0
        
        for _ in range(num_vectors):
            # Sample random vector from standard normal
            epsilon = torch.randn_like(z)
            
            # Compute the Jacobian-vector product using autograd
            # Create a copy of z to avoid modifying the original
            z_copy = z.clone().detach().requires_grad_(True)
            f_z = self.net(z_copy)
            jvp = torch.autograd.grad(
                f_z.sum(), z_copy, create_graph=True, retain_graph=True
            )[0]
            
            # Compute the trace contribution
            trace_contribution = (jvp * epsilon).sum(dim=1, keepdim=True)
            trace_estimate += trace_contribution
        
        # Average the trace estimates
        trace = trace_estimate / num_vectors
        
        # The change in log-determinant is the trace
        dlog_det_dt = trace
        
        # Return the concatenated result
        return torch.cat([dz_dt, dlog_det_dt], dim=1)
    
    def _initialize_weights(self):
        """Initialize network weights using Xavier initialization with proper scaling."""
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                # Use Xavier normal initialization for better stability
                nn.init.xavier_normal_(layer.weight, gain=0.1)  # Small gain for stability
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)
        
        # Special initialization for the final layer to start with near-zero dynamics
        final_layer = self.net[-1]
        if isinstance(final_layer, nn.Linear):
            nn.init.zeros_(final_layer.weight)
            if final_layer.bias is not None:
                nn.init.zeros_(final_layer.bias)

class ContinuousFlow(nn.Module):
    """
    Continuous Normalizing Flow model with proper log-determinant calculation.
    """
    def __init__(self, dim, hidden_dim=64):
        super().__init__()
        self.dim = dim
        self.ode_func = ODEFunc(dim, hidden_dim)

    def forward(self, z, integration_times=torch.tensor([0.0, 1.0])):
        """
        Forward pass (sampling). Solves the ODE from t=0 to t=1.
        
        Args:
            z: input tensor from base distribution
            integration_times: time points for integration
            
        Returns:
            tuple (x, log_det_J) where x is the transformed tensor and
            log_det_J is the log-determinant of the Jacobian
        """
        batch_size = z.size(0)
        device = z.device
        
        # Create augmented state: [z, log_det]
        # Initialize log_det to zero
        log_det_init = torch.zeros(batch_size, 1, device=device)
        augmented_state = torch.cat([z, log_det_init], dim=1)
        
        # Ensure integration_times is valid for torchdiffeq
        if integration_times[0] == integration_times[1]:
            # If start and end times are the same, return the input unchanged
            return z, torch.zeros(batch_size, device=device)
        
        # Ensure integration_times is on the correct device
        integration_times = integration_times.to(device)
        
        # Solve the ODE with augmented state
        try:
            solution = odeint(
                self.ode_func,
                augmented_state,
                integration_times,
                method='dopri5',
                atol=1e-5,
                rtol=1e-5
            )
        except Exception as e:
            # Fallback to simpler solver if dopri5 fails
            solution = odeint(
                self.ode_func,
                augmented_state,
                integration_times,
                method='rk4',
                options={'step_size': 0.01}
            )
        
        # Extract the final state
        final_augmented_state = solution[-1]
        x = final_augmented_state[:, :self.dim]
        log_det_J = final_augmented_state[:, self.dim:].squeeze(-1)
        
        # Safety check: ensure x doesn't contain NaN or infinite values
        x = torch.where(torch.isnan(x) | torch.isinf(x), z, x)
        log_det_J = torch.where(
            torch.isnan(log_det_J) | torch.isinf(log_det_J),
            torch.zeros_like(log_det_J),
            log_det_J
        )
        
        return x, log_det_J

    def inverse(self, x, integration_times=torch.tensor([1.0, 0.0])):
        """
        Inverse pass (likelihood). Solves the ODE from t=1 to t=0.
        
        Args:
            x: input tensor from data distribution
            integration_times: time points for integration
            
        Returns:
            tuple (z, log_det_J_inv) where z is the reconstructed tensor and
            log_det_J_inv is the log-determinant of the inverse Jacobian
        """
        batch_size = x.size(0)
        device = x.device
        
        # Create augmented state: [x, log_det]
        # Initialize log_det to zero
        log_det_init = torch.zeros(batch_size, 1, device=device)
        augmented_state = torch.cat([x, log_det_init], dim=1)
        
        # Ensure integration_times is valid for torchdiffeq
        if integration_times[0] == integration_times[1]:
            # If start and end times are the same, return the input unchanged
            return x, torch.zeros(batch_size, device=device)
        
        # Ensure integration_times is on the correct device
        integration_times = integration_times.to(device)
        
        # Solve the ODE with augmented state
        try:
            solution = odeint(
                self.ode_func,
                augmented_state,
                integration_times,
                method='dopri5',
                atol=1e-5,
                rtol=1e-5
            )
        except Exception as e:
            # Fallback to simpler solver if dopri5 fails
            solution = odeint(
                self.ode_func,
                augmented_state,
                integration_times,
                method='rk4',
                options={'step_size': 0.01}
            )
        
        # Extract the final state
        final_augmented_state = solution[-1]
        z = final_augmented_state[:, :self.dim]
        log_det_J_inv = final_augmented_state[:, self.dim:].squeeze(-1)
        
        # Safety check: ensure z doesn't contain NaN or infinite values
        z = torch.where(torch.isnan(z) | torch.isinf(z), x, z)
        log_det_J_inv = torch.where(
            torch.isnan(log_det_J_inv) | torch.isinf(log_det_J_inv),
            torch.zeros_like(log_det_J_inv),
            log_det_J_inv
        )
        
        return z, log_det_J_inv

class SplineCouplingLayer(Flow):
    """
    Implements a coupling layer with a rational-quadratic spline transformation.
    This layer follows the implementation described in the Neural Spline Flows paper
    (https://arxiv.org/abs/1906.04032), which builds upon the method of
    Gregory and Delbourgo.
    """
    def __init__(self, 
                 data_dim, 
                 hidden_dim, 
                 mask, 
                 num_bins=10, 
                 bound=5.0,
                 min_bin_width=1e-3,
                 min_bin_height=1e-3,
                 min_derivative=1e-3):
        """
        Initializes the SplineCouplingLayer.

        Args:
            data_dim (int): The dimensionality of the data.
            hidden_dim (int): The number of hidden units in the conditioner network.
            mask (torch.Tensor): A binary mask of shape (data_dim,) to separate
                                 identity from transformed dimensions.
            num_bins (int): The number of bins to use for the spline.
            bound (float): The bound of the interval [-B, B] on which the spline is defined.
                           Values outside this interval are identity-mapped.
        """
        super().__init__()
        
        self.data_dim = data_dim
        self.num_bins = num_bins
        self.bound = bound
        self.min_bin_width = min_bin_width
        self.min_bin_height = min_bin_height
        self.min_derivative = min_derivative

        # The mask determines which dimensions are transformed (mask == 0) 
        # and which are passed through as identity (mask == 1).
        self.register_buffer('mask', mask)

        # The conditioner network outputs the parameters for the splines.
        # For each transformed dimension, we need K widths, K heights, and K-1 derivatives.
        # Total parameters per transformed dimension = 3K - 1.
        self.param_net = nn.Sequential(
            nn.Linear(data_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, data_dim * (3 * num_bins - 1))
        )
        
        # Initialize weights properly to prevent exploding gradients
        self._initialize_weights()

    def _get_spline_params(self, z_a):
        """
        Computes and reshapes the spline parameters from the conditioner network.
        
        Args:
            z_a (torch.Tensor): The masked input tensor.

        Returns:
            A tuple of (unnormalized_widths, unnormalized_heights, unnormalized_derivatives).
        """
        params = self.param_net(z_a)
        params = params.view(-1, self.data_dim, 3 * self.num_bins - 1)
        
        # Split the output into the three parameter types
        unnormalized_widths, unnormalized_heights, unnormalized_derivatives = torch.split(
            params, [self.num_bins, self.num_bins, self.num_bins - 1], dim=-1
        )
        return unnormalized_widths, unnormalized_heights, unnormalized_derivatives

    def forward(self, z):
        """
        Computes the forward pass x = f(z). (z -> x)
        
        Args:
            z (torch.Tensor): The input tensor from the base distribution.
        
        Returns:
            A tuple (x, log_det_J) where x is the transformed tensor and
            log_det_J is the log of the absolute value of the Jacobian determinant.
        """
        # The identity part of the input is used to condition the transformation.
        z_a = z * self.mask
        
        # Get spline parameters from the conditioner
        unnormalized_widths, unnormalized_heights, unnormalized_derivatives = self._get_spline_params(z_a)
        
        # The transformation is applied to the other part of the input.
        # We use (1 - mask) to select the dimensions to transform.
        num_transformed_dims = int((self.mask == 0).sum())

        # Extract the transformed dimensions without flattening batch
        z_b = z[:, self.mask == 0]  # Shape: (batch_size, num_transformed_dims)
        
        # Extract parameters for transformed dimensions
        un_widths_b = unnormalized_widths[:, self.mask == 0]  # Shape: (batch_size, num_transformed_dims, num_bins)
        un_heights_b = unnormalized_heights[:, self.mask == 0]  # Shape: (batch_size, num_transformed_dims, num_bins)
        un_derivs_b = unnormalized_derivatives[:, self.mask == 0]  # Shape: (batch_size, num_transformed_dims, num_bins-1)

        # Apply the spline transformation on the batched data
        x_b_transformed, log_det_b = self._rational_quadratic_spline(
            inputs=z_b,
            unnormalized_widths=un_widths_b,
            unnormalized_heights=un_heights_b,
            unnormalized_derivatives=un_derivs_b,
            inverse=False
        )

        # Reconstruct the output tensor
        x = torch.empty_like(z)
        x[:, self.mask == 1] = z[:, self.mask == 1] # Identity part
        x[:, self.mask == 0] = x_b_transformed         # Transformed part
        
        # Sum the log determinant across transformed dimensions
        log_det_J = log_det_b.sum(dim=1)

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
        Computes the inverse pass z = g(x). (x -> z)

        Args:
            x (torch.Tensor): The input tensor from the data distribution.

        Returns:
            A tuple (z, log_det_J_inv) where z is the reconstructed tensor and
            log_det_J_inv is the log of the absolute value of the inverse Jacobian determinant.
        """
        # The identity part is used for conditioning.
        x_a = x * self.mask

        # Get spline parameters
        unnormalized_widths, unnormalized_heights, unnormalized_derivatives = self._get_spline_params(x_a)
        
        num_transformed_dims = int((self.mask == 0).sum())
        
        # Extract the transformed dimensions without flattening batch
        x_b = x[:, self.mask == 0]  # Shape: (batch_size, num_transformed_dims)
        
        # Extract parameters for transformed dimensions
        un_widths_b = unnormalized_widths[:, self.mask == 0]  # Shape: (batch_size, num_transformed_dims, num_bins)
        un_heights_b = unnormalized_heights[:, self.mask == 0]  # Shape: (batch_size, num_transformed_dims, num_bins)
        un_derivs_b = unnormalized_derivatives[:, self.mask == 0]  # Shape: (batch_size, num_transformed_dims, num_bins-1)

        # Apply the inverse spline transformation
        z_b_transformed, log_det_inv_b = self._rational_quadratic_spline(
            inputs=x_b,
            unnormalized_widths=un_widths_b,
            unnormalized_heights=un_heights_b,
            unnormalized_derivatives=un_derivs_b,
            inverse=True
        )

        # Reconstruct the output tensor
        z = torch.empty_like(x)
        z[:, self.mask == 1] = x[:, self.mask == 1] # Identity part
        z[:, self.mask == 0] = z_b_transformed          # Transformed part

        # Sum the log determinants across transformed dimensions
        log_det_J_inv = log_det_inv_b.sum(dim=1)

        # Safety check: ensure z doesn't contain NaN or infinite values
        z = torch.where(torch.isnan(z) | torch.isinf(z), torch.zeros_like(z), z)
        log_det_J_inv = torch.where(
            torch.isnan(log_det_J_inv) | torch.isinf(log_det_J_inv),
            torch.zeros_like(log_det_J_inv),
            log_det_J_inv
        )

        return z, log_det_J_inv

    def _rational_quadratic_spline(self, inputs, unnormalized_widths, unnormalized_heights, unnormalized_derivatives, inverse=False):
        """
        Apply rational quadratic spline transformation to batched inputs.
        Args:
            inputs: tensor of shape (batch_size, num_transformed_dims)
            unnormalized_widths: tensor of shape (batch_size, num_transformed_dims, num_bins)
            unnormalized_heights: tensor of shape (batch_size, num_transformed_dims, num_bins)
            unnormalized_derivatives: tensor of shape (batch_size, num_transformed_dims, num_bins-1)
            inverse: whether to apply inverse transformation
        Returns:
            tuple (outputs, logabsdet) where both have shape (batch_size, num_transformed_dims)
        """
        batch_size, num_transformed_dims = inputs.shape
        device = inputs.device
        
        # --- Handle points outside the defined interval ---
        inside_interval_mask = (inputs >= -self.bound) & (inputs <= self.bound)
        outside_interval_mask = ~inside_interval_mask

        outputs = torch.zeros_like(inputs)
        logabsdet = torch.zeros_like(inputs)

        # Identity map for points outside the interval
        outputs[outside_interval_mask] = inputs[outside_interval_mask]
        logabsdet[outside_interval_mask] = 0

        # Proceed with points inside the interval
        if not inside_interval_mask.any():
            return outputs, logabsdet

        # --- Normalize parameters for all batch elements ---
        widths = F.softmax(unnormalized_widths, dim=-1)
        widths = self.min_bin_width + (1 - self.min_bin_width * self.num_bins) * widths
        cum_widths = torch.cumsum(widths, dim=-1)
        cum_widths = F.pad(cum_widths, pad=(0, 0, 1, 0), mode='constant', value=0.0)
        cum_widths = (2 * self.bound) * cum_widths + (-self.bound)
        cum_widths[..., 0] = -self.bound
        cum_widths[..., -1] = self.bound
        widths = cum_widths[..., 1:] - cum_widths[..., :-1]

        heights = F.softmax(unnormalized_heights, dim=-1)
        heights = self.min_bin_height + (1 - self.min_bin_height * self.num_bins) * heights
        cum_heights = torch.cumsum(heights, dim=-1)
        cum_heights = F.pad(cum_heights, pad=(0, 0, 1, 0), mode='constant', value=0.0)
        cum_heights = (2 * self.bound) * cum_heights + (-self.bound)
        cum_heights[..., 0] = -self.bound
        cum_heights[..., -1] = self.bound
        heights = cum_heights[..., 1:] - cum_heights[..., :-1]

        # Normalize derivatives and pad with 1s at boundaries
        derivatives = self.min_derivative + F.softplus(unnormalized_derivatives)
        # The unnormalized_derivatives is already the subset for transformed dimensions
        # We need to pad it to get num_bins+1 elements (original had num_bins-1)
        derivatives = F.pad(derivatives, pad=(1, 1), mode='constant', value=1.0)
        
        # Debug: print shapes to understand what's happening
        #print(f"Debug - derivatives shape after padding: {derivatives.shape}")
        #print(f"Debug - num_bins: {self.num_bins}")
        #print(f"Debug - unnormalized_derivatives shape: {unnormalized_derivatives.shape}")

        # Only process inside-interval points
        idx = torch.nonzero(inside_interval_mask, as_tuple=False)
        b_idx, f_idx = idx[:, 0], idx[:, 1]
        x_in = inputs[b_idx, f_idx]

        if inverse:
            # For each (b, f), search in cum_heights[b, f]
            bin_idx = torch.stack([
                torch.searchsorted(cum_heights[b, f], x, right=True) - 1
                for b, f, x in zip(b_idx, f_idx, x_in)
            ])
            bin_idx = bin_idx.clamp(min=0, max=self.num_bins - 1)  # Derivatives are padded, so valid range is 0 to num_bins-1
            # Gather bin params
            widths_sel = widths[b_idx, f_idx, bin_idx]
            cumwidths_sel = cum_widths[b_idx, f_idx, bin_idx]
            heights_sel = heights[b_idx, f_idx, bin_idx]
            cumheights_sel = cum_heights[b_idx, f_idx, bin_idx]
            derivatives_sel = derivatives[b_idx, f_idx, bin_idx]
            # Ensure bin_idx + 1 doesn't exceed the actual derivatives tensor bounds
            max_bin_idx = derivatives.shape[-1] - 1
            bin_idx_plus1 = torch.clamp(bin_idx + 1, max=max_bin_idx)
            derivatives_plus1_sel = derivatives[b_idx, f_idx, bin_idx_plus1]
            s_k = heights_sel / widths_sel

            # Inverse: solve for xi in [0, 1] using quadratic formula
            y = x_in
            y_k = cumheights_sel
            w_k = widths_sel
            z_k = cumwidths_sel
            d_k = derivatives_sel
            d_k1 = derivatives_plus1_sel
            h_k = heights_sel
            s_k = h_k / w_k

            a = (y - y_k) * (d_k + d_k1 - 2 * s_k) + h_k * (s_k - d_k)
            b = h_k * d_k - (y - y_k) * (d_k + d_k1 - 2 * s_k)
            c = -s_k * (y - y_k)
            discriminant = b.pow(2) - 4 * a * c
            discriminant = torch.clamp(discriminant, min=0)
            numerator = -b + torch.sqrt(discriminant)
            denominator = 2 * a
            eps = 1e-8
            xi = numerator / (torch.sign(denominator) * torch.clamp(torch.abs(denominator), min=eps))
            xi = torch.clamp(xi, 0, 1)

            z = xi * w_k + z_k
            outputs[b_idx, f_idx] = z

            # Logabsdet
            denominator_ld = s_k + (d_k1 + d_k - 2 * s_k) * xi * (1 - xi)
            numerator_ld = s_k.pow(2) * (d_k1 * xi.pow(2) + 2 * s_k * xi * (1 - xi) + d_k * (1 - xi).pow(2))
            numerator_ld = torch.clamp(numerator_ld, min=1e-8)
            denominator_ld = torch.clamp(denominator_ld, min=1e-8)
            logabsdet_inside = -torch.log(numerator_ld) + 2 * torch.log(denominator_ld)
            logabsdet[b_idx, f_idx] = logabsdet_inside
        else:
            # For each (b, f), search in cum_widths[b, f]
            bin_idx = torch.stack([
                torch.searchsorted(cum_widths[b, f], x, right=True) - 1
                for b, f, x in zip(b_idx, f_idx, x_in)
            ])
            bin_idx = bin_idx.clamp(min=0, max=self.num_bins - 1)  # Derivatives are padded, so valid range is 0 to num_bins-1
            # Gather bin params
            widths_sel = widths[b_idx, f_idx, bin_idx]
            cumwidths_sel = cum_widths[b_idx, f_idx, bin_idx]
            heights_sel = heights[b_idx, f_idx, bin_idx]
            cumheights_sel = cum_heights[b_idx, f_idx, bin_idx]
            derivatives_sel = derivatives[b_idx, f_idx, bin_idx]
            # Ensure bin_idx + 1 doesn't exceed the actual derivatives tensor bounds
            max_bin_idx = derivatives.shape[-1] - 1
            bin_idx_plus1 = torch.clamp(bin_idx + 1, max=max_bin_idx)
            derivatives_plus1_sel = derivatives[b_idx, f_idx, bin_idx_plus1]
            s_k = heights_sel / widths_sel

            # Forward: compute xi = (x - z_k) / w_k
            x = x_in
            z_k = cumwidths_sel
            w_k = widths_sel
            h_k = heights_sel
            d_k = derivatives_sel
            d_k1 = derivatives_plus1_sel
            s_k = h_k / w_k
            xi = (x - z_k) / w_k
            xi = torch.clamp(xi, 0, 1)

            # Output
            denominator = s_k + (d_k1 + d_k - 2 * s_k) * xi * (1 - xi)
            numerator = h_k * (s_k * xi.pow(2) + d_k * xi * (1 - xi))
            y = cumheights_sel + numerator / denominator
            outputs[b_idx, f_idx] = y

            # Logabsdet
            numerator_ld = s_k.pow(2) * (d_k1 * xi.pow(2) + 2 * s_k * xi * (1 - xi) + d_k * (1 - xi).pow(2))
            denominator_ld = s_k + (d_k1 + d_k - 2 * s_k) * xi * (1 - xi)
            numerator_ld = torch.clamp(numerator_ld, min=1e-8)
            denominator_ld = torch.clamp(denominator_ld, min=1e-8)
            logabsdet_inside = torch.log(numerator_ld) - 2 * torch.log(denominator_ld)
            logabsdet[b_idx, f_idx] = logabsdet_inside

        # Additional safety check for NaN or infinite values
        outputs = torch.where(torch.isnan(outputs) | torch.isinf(outputs), torch.zeros_like(outputs), outputs)
        logabsdet = torch.where(torch.isnan(logabsdet) | torch.isinf(logabsdet), torch.zeros_like(logabsdet), logabsdet)

        return outputs, logabsdet
    
    def _initialize_weights(self):
        """Initialize network weights."""
        # Initialize all but the final layer
        for layer in self.param_net[:-1]:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight, gain=1.0) # Standard gain is often fine
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)
        
        # Special initialization for the final layer
        final_layer = self.param_net[-1]
        if isinstance(final_layer, nn.Linear):
            # Initialize weights to be very small and biases to zero
            # to start close to an identity transformation.
            nn.init.zeros_(final_layer.weight)
            if final_layer.bias is not None:
                nn.init.zeros_(final_layer.bias)