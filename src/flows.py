import torch
import torch.nn as nn
from torch.nn import functional as F
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
        # Mask for input to hidden layer: m_k(x_j) <= m_{k-1}(x_i)
        # Transpose to get shape (hidden_dim, input_dim)
        mask1 = (self.m[-1][:, np.newaxis] <= self.m[0][np.newaxis, :]).T
        masks.append(mask1)

        # Mask for hidden to output layer: m_D(y_j) > m_k(x_i)
        # Transpose to get shape (output_dim, hidden_dim)
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
        batch_size = z.size(0)
        x_list = [torch.zeros(batch_size, device=z.device) for _ in range(self.dim)]
        log_det_jacobian = torch.zeros(batch_size, device=z.device)

        # Iterate over each dimension to generate the sample
        for i in range(self.dim):
            # Build the current x tensor from the list
            x_current = torch.stack(x_list, dim=1)
            
            # The conditioner's output for dim i depends only on x_1, ..., x_{i-1}
            params = self.conditioner(x_current) # Pass the partially generated x
            mu, alpha = params.chunk(2, dim=1)
            
            # Clamp alpha to prevent numerical instability
            alpha = torch.clamp(alpha, min=-10, max=10)
            
            # Apply the transformation for the current dimension
            x_list[i] = z[:, i] * torch.exp(alpha[:, i]) + mu[:, i]
            log_det_jacobian += alpha[:, i]

        # Stack the final result
        x = torch.stack(x_list, dim=1)
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
        batch_size = x.size(0)
        z_list = [torch.zeros(batch_size, device=x.device) for _ in range(self.dim)]
        log_det_jacobian = torch.zeros(batch_size, device=x.device)

        # Iterate over each dimension
        for i in range(self.dim):
            # Build the current z tensor from the list
            z_current = torch.stack(z_list, dim=1)
            
            # The conditioner's output depends on z_1, ..., z_{i-1}, which we have
            # already computed.
            params = self.conditioner(z_current)
            mu, alpha = params.chunk(2, dim=1)
            
            # Clamp alpha to prevent numerical instability
            alpha = torch.clamp(alpha, min=-10, max=10)
            
            # Apply the inverse transformation for the current dimension
            z_list[i] = (x[:, i] - mu[:, i]) * torch.exp(-alpha[:, i])
            
            # Accumulate the log-determinant from this step's alpha
            log_det_jacobian -= alpha[:, i]

        # Stack the final result
        z = torch.stack(z_list, dim=1)
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
        super(SplineCouplingLayer, self).__init__()
        
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

        # The spline function requires inputs to be 1D, so we flatten the batch/dim axes.
        z_b_flat = z[:, self.mask == 0].reshape(-1)

        # We also need the parameters corresponding to these dimensions, also flattened.
        un_widths_flat = unnormalized_widths[:, self.mask == 0].reshape(-1, self.num_bins)
        un_heights_flat = unnormalized_heights[:, self.mask == 0].reshape(-1, self.num_bins)
        un_derivs_flat = unnormalized_derivatives[:, self.mask == 0].reshape(-1, self.num_bins - 1)

        # Apply the spline transformation on the flattened data
        x_b_transformed_flat, log_det_b_flat = self._rational_quadratic_spline(
            inputs=z_b_flat,
            unnormalized_widths=un_widths_flat,
            unnormalized_heights=un_heights_flat,
            unnormalized_derivatives=un_derivs_flat,
            inverse=False
        )
        
        # Reshape the transformed part back to (batch_size, num_transformed_dims)
        x_b_reshaped = x_b_transformed_flat.view(-1, num_transformed_dims)

        # Reconstruct the output tensor
        x = torch.empty_like(z)
        x[:, self.mask == 1] = z[:, self.mask == 1] # Identity part
        x[:, self.mask == 0] = x_b_reshaped         # Transformed part
        
        # Reshape and sum the log determinant
        log_det_b_reshaped = log_det_b_flat.view(-1, num_transformed_dims)
        log_det_J = log_det_b_reshaped.sum(dim=1)

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
        
        # Flatten the inputs and parameters for the spline function.
        x_b_flat = x[:, self.mask == 0].reshape(-1)
        un_widths_flat = unnormalized_widths[:, self.mask == 0].reshape(-1, self.num_bins)
        un_heights_flat = unnormalized_heights[:, self.mask == 0].reshape(-1, self.num_bins)
        un_derivs_flat = unnormalized_derivatives[:, self.mask == 0].reshape(-1, self.num_bins - 1)

        # Apply the inverse spline transformation
        z_b_transformed_flat, log_det_inv_b_flat = self._rational_quadratic_spline(
            inputs=x_b_flat,
            unnormalized_widths=un_widths_flat,
            unnormalized_heights=un_heights_flat,
            unnormalized_derivatives=un_derivs_flat,
            inverse=True
        )
        
        # Reshape the transformed part back
        z_b_reshaped = z_b_transformed_flat.view(-1, num_transformed_dims)

        # Reconstruct the output tensor
        z = torch.empty_like(x)
        z[:, self.mask == 1] = x[:, self.mask == 1] # Identity part
        z[:, self.mask == 0] = z_b_reshaped          # Transformed part

        # Reshape and sum the log determinants for each sample in the batch
        log_det_inv_b_reshaped = log_det_inv_b_flat.view(-1, num_transformed_dims)
        log_det_J_inv = log_det_inv_b_reshaped.sum(dim=1)

        return z, log_det_J_inv

    def _rational_quadratic_spline(self, inputs, unnormalized_widths, unnormalized_heights, unnormalized_derivatives, inverse=False):
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
            
        inputs_inside = inputs[inside_interval_mask]
        
        # --- Normalize parameters ---
        # Select params for inside points. These will have shape (num_inside, num_bins) etc.
        un_widths = unnormalized_widths[inside_interval_mask, :]
        un_heights = unnormalized_heights[inside_interval_mask, :]
        un_derivs = unnormalized_derivatives[inside_interval_mask, :]
        
        # Normalize widths and heights to sum to 2*bound and enforce minimums
        widths = F.softmax(un_widths, dim=-1)
        widths = self.min_bin_width + (1 - self.min_bin_width * self.num_bins) * widths
        cum_widths = torch.cumsum(widths, dim=-1)
        cum_widths = F.pad(cum_widths, pad=(1, 0), mode='constant', value=0.0)
        cum_widths = (2 * self.bound) * cum_widths + (-self.bound)
        cum_widths[..., 0] = -self.bound
        cum_widths[..., -1] = self.bound
        widths = cum_widths[..., 1:] - cum_widths[..., :-1]

        heights = F.softmax(un_heights, dim=-1)
        heights = self.min_bin_height + (1 - self.min_bin_height * self.num_bins) * heights
        cum_heights = torch.cumsum(heights, dim=-1)
        cum_heights = F.pad(cum_heights, pad=(1, 0), mode='constant', value=0.0)
        cum_heights = (2 * self.bound) * cum_heights + (-self.bound)
        cum_heights[..., 0] = -self.bound
        cum_heights[..., -1] = self.bound
        heights = cum_heights[..., 1:] - cum_heights[..., :-1]

        # Normalize derivatives and pad with 1s at boundaries
        derivatives = self.min_derivative + F.softplus(un_derivs)
        derivatives = F.pad(derivatives, pad=(1, 1), mode='constant', value=1.0)
        
        # --- Find bin for each input and gather params ---
        if inverse:
            bin_idx = torch.searchsorted(cum_heights, inputs_inside[..., None]).squeeze(-1)
        else:
            bin_idx = torch.searchsorted(cum_widths, inputs_inside[..., None]).squeeze(-1)
        
        # Clamp to avoid out-of-bounds due to precision
        bin_idx = torch.clamp(bin_idx, 0, self.num_bins - 1)

        def gather_params(params, idx):
            return torch.gather(params, -1, idx[..., None]).squeeze(-1)

        input_cum_widths = gather_params(cum_widths, bin_idx)
        input_widths = gather_params(widths, bin_idx)
        input_cum_heights = gather_params(cum_heights, bin_idx)
        input_heights = gather_params(heights, bin_idx)
        input_delta_k = gather_params(derivatives, bin_idx)
        input_delta_k_plus_1 = gather_params(derivatives, bin_idx + 1)
        
        s_k = input_heights / input_widths

        # --- Apply Spline Transformation ---
        if inverse:
            # Solve for ξ in [0, 1] using Eq. 25-32
            y = inputs_inside
            y_k = input_cum_heights
            
            # Coefficients for the quadratic equation aξ^2 + bξ + c = 0
            a = (input_heights * (s_k - input_delta_k)) + (y - y_k) * (input_delta_k_plus_1 + input_delta_k - 2 * s_k)
            b = (input_heights * input_delta_k) - (y - y_k) * (input_delta_k_plus_1 + input_delta_k - 2 * s_k) * 2
            c = -s_k * (y - y_k)

            discriminant = b.pow(2) - 4 * a * c
            discriminant = F.relu(discriminant) # Ensure non-negative

            # Use numerically stable form for the root corresponding to ξ
            xi = (2 * c) / (-b - torch.sqrt(discriminant))
            xi = torch.clamp(xi, 0, 1) # Clamp to handle precision errors

            # Compute z (output) from ξ
            outputs_inside = xi * input_widths + input_cum_widths
            
            # --- CORRECTED LOG-DETERMINANT CALCULATION ---
            # The log-determinant of the inverse is -log(forward_derivative).
            # The forward derivative is a function of ξ. We must compute this ξ
            # from the *output* of this inverse function (`outputs_inside`).
            
            # Re-calculate xi based on the output z
            xi_for_logdet = (outputs_inside - input_cum_widths) / input_widths
            xi_for_logdet = torch.clamp(xi_for_logdet, 0, 1)

            # Denominator of the forward derivative
            denominator = s_k + (input_delta_k_plus_1 + input_delta_k - 2 * s_k) * xi_for_logdet * (1 - xi_for_logdet)
            # Numerator of the forward derivative
            numerator_logdet = s_k.pow(2) * (input_delta_k_plus_1 * xi_for_logdet.pow(2) + 2 * s_k * xi_for_logdet * (1 - xi_for_logdet) + input_delta_k * (1 - xi_for_logdet).pow(2))
            
            # log|det(J_inv)| = -log|det(J_forward)|
            logabsdet_inside = - (torch.log(numerator_logdet) - 2 * torch.log(denominator))


        else: # Forward pass
            # Compute ξ = (z - z_k) / w_k
            xi = (inputs_inside - input_cum_widths) / input_widths
            xi = torch.clamp(xi, 0, 1)

            # Compute x (output) using Eq. 19
            denominator = s_k + (input_delta_k_plus_1 + input_delta_k - 2 * s_k) * xi * (1 - xi)
            numerator = input_heights * (s_k * xi.pow(2) + input_delta_k * xi * (1 - xi))
            outputs_inside = input_cum_heights + (numerator / denominator)

            # Compute log derivative using Eq. 22
            numerator_logdet = s_k.pow(2) * (input_delta_k_plus_1 * xi.pow(2) + 2 * s_k * xi * (1 - xi) + input_delta_k * (1 - xi).pow(2))
            logabsdet_inside = torch.log(numerator_logdet) - 2 * torch.log(denominator)

        outputs[inside_interval_mask] = outputs_inside
        logabsdet[inside_interval_mask] = logabsdet_inside
        
        return outputs, logabsdet