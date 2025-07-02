import torch
import torch.nn as nn
from torch.nn import functional as F
from .flow import Flow

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
                 min_derivative=1e-3,
                 data_min=None,
                 data_max=None):
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
            data_min (float or torch.Tensor): Minimum value(s) of the data (for rescaling).
            data_max (float or torch.Tensor): Maximum value(s) of the data (for rescaling).
        """
        super().__init__()
        
        self.data_dim = data_dim
        self.num_bins = num_bins
        self.bound = bound
        self.min_bin_width = min_bin_width
        self.min_bin_height = min_bin_height
        self.min_derivative = min_derivative
        self.data_min = data_min
        self.data_max = data_max

        self.register_buffer('mask', mask)

        self.param_net = nn.Sequential(
            nn.Linear(data_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, data_dim * (3 * num_bins - 1))
        )
        
        self._initialize_weights()

    def _get_spline_params(self, z_a):
        """
        Computes and reshapes the spline parameters from the conditioner network.
        """
        params = self.param_net(z_a)
        params = params.view(-1, self.data_dim, 3 * self.num_bins - 1)
        
        unnormalized_widths, unnormalized_heights, unnormalized_derivatives = torch.split(
            params, [self.num_bins, self.num_bins, self.num_bins - 1], dim=-1
        )
        return unnormalized_widths, unnormalized_heights, unnormalized_derivatives

    def _rescale_to_spline(self, x):
        """
        Rescale data from [data_min, data_max] to [-bound, bound].
        """
        if self.data_min is None or self.data_max is None:
            return x
        scale = (2 * self.bound) / (self.data_max - self.data_min)
        return scale * (x - self.data_min) - self.bound

    def _rescale_from_spline(self, x):
        """
        Rescale data from [-bound, bound] back to [data_min, data_max].
        """
        if self.data_min is None or self.data_max is None:
            return x
        scale = (self.data_max - self.data_min) / (2 * self.bound)
        return (x + self.bound) * scale + self.data_min

    def forward(self, z):
        """
        Computes the forward pass x = f(z). (z -> x)
        """
        # Rescale input to spline interval
        z_rescaled = self._rescale_to_spline(z)
        z_a = z_rescaled * self.mask
        
        unnormalized_widths, unnormalized_heights, unnormalized_derivatives = self._get_spline_params(z_a)
        
        num_transformed_dims = int((self.mask == 0).sum())

        z_b = z_rescaled[:, self.mask == 0]
        
        un_widths_b = unnormalized_widths[:, self.mask == 0]
        un_heights_b = unnormalized_heights[:, self.mask == 0]
        un_derivs_b = unnormalized_derivatives[:, self.mask == 0]

        x_b_transformed, log_det_b = self._rational_quadratic_spline(
            inputs=z_b,
            unnormalized_widths=un_widths_b,
            unnormalized_heights=un_heights_b,
            unnormalized_derivatives=un_derivs_b,
            inverse=False
        )
        # Rescale output back to original data range
        x_b_transformed = self._rescale_from_spline(x_b_transformed)

        # Assign transformed values to the correct positions using advanced indexing
        x = z.clone()
        x[:, self.mask == 0] = x_b_transformed
        
        log_det_J = log_det_b.sum(dim=1)

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
        """
        # Rescale input to spline interval
        x_rescaled = self._rescale_to_spline(x)
        x_a = x_rescaled * self.mask

        unnormalized_widths, unnormalized_heights, unnormalized_derivatives = self._get_spline_params(x_a)
        
        num_transformed_dims = int((self.mask == 0).sum())
        
        x_b = x_rescaled[:, self.mask == 0]
        
        un_widths_b = unnormalized_widths[:, self.mask == 0]
        un_heights_b = unnormalized_heights[:, self.mask == 0]
        un_derivs_b = unnormalized_derivatives[:, self.mask == 0]

        z_b_transformed, log_det_inv_b = self._rational_quadratic_spline(
            inputs=x_b,
            unnormalized_widths=un_widths_b,
            unnormalized_heights=un_heights_b,
            unnormalized_derivatives=un_derivs_b,
            inverse=True
        )
        # Rescale output back to original data range
        z_b_transformed = self._rescale_from_spline(z_b_transformed)

        # Assign transformed values to the correct positions using advanced indexing
        z = x.clone()
        z[:, self.mask == 0] = z_b_transformed

        log_det_J_inv = log_det_inv_b.sum(dim=1)

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
        Fully vectorized implementation avoiding all in-place operations.
        """
        batch_size, num_transformed_dims = inputs.shape
        device = inputs.device
        eps = 1e-8  # Numerical stability constant

        # Create masks for inside/outside interval
        inside_interval_mask = (inputs >= -self.bound) & (inputs <= self.bound)
        outside_interval_mask = ~inside_interval_mask

        # Initialize outputs - identity mapping for outside interval
        outputs = torch.where(outside_interval_mask, inputs, torch.zeros_like(inputs))
        logabsdet = torch.zeros_like(inputs)

        # If no elements are inside the interval, return early
        if not inside_interval_mask.any():
            return outputs, logabsdet

        # Process spline parameters with better numerical stability
        widths = F.softmax(unnormalized_widths, dim=-1)
        widths = self.min_bin_width + (1 - self.min_bin_width * self.num_bins) * widths
        widths = torch.clamp(widths, min=eps)  # Ensure positive widths

        cum_widths = torch.cumsum(widths, dim=-1)
        cum_widths = F.pad(cum_widths, pad=(0, 0, 1, 0), mode='constant', value=0.0)
        cum_widths = (2 * self.bound) * cum_widths + (-self.bound)
        cum_widths[..., 0] = -self.bound
        cum_widths[..., -1] = self.bound
        widths = cum_widths[..., 1:] - cum_widths[..., :-1]
        widths = torch.clamp(widths, min=eps)  # Ensure positive widths

        heights = F.softmax(unnormalized_heights, dim=-1)
        heights = self.min_bin_height + (1 - self.min_bin_height * self.num_bins) * heights
        heights = torch.clamp(heights, min=eps)  # Ensure positive heights

        cum_heights = torch.cumsum(heights, dim=-1)
        cum_heights = F.pad(cum_heights, pad=(0, 0, 1, 0), mode='constant', value=0.0)
        cum_heights = (2 * self.bound) * cum_heights + (-self.bound)
        cum_heights[..., 0] = -self.bound
        cum_heights[..., -1] = self.bound
        heights = cum_heights[..., 1:] - cum_heights[..., :-1]
        heights = torch.clamp(heights, min=eps)  # Ensure positive heights

        derivatives = self.min_derivative + F.softplus(unnormalized_derivatives)
        derivatives = torch.clamp(derivatives, min=eps)  # Ensure positive derivatives
        derivatives = F.pad(derivatives, pad=(1, 1), mode='constant', value=1.0)

        # Vectorized searchsorted for [batch, dim] pairs
        flat_inputs = inputs.contiguous().view(-1)  # [B * D]
        if inverse:
            boundaries = cum_heights.contiguous().view(-1, cum_heights.shape[-1])  # [B * D, num_bins+1]
        else:
            boundaries = cum_widths.contiguous().view(-1, cum_widths.shape[-1])  # [B * D, num_bins+1]
        # Row-wise searchsorted for each [B*D] pair
        bin_idx = torch.stack([
            torch.searchsorted(boundaries[i], flat_inputs[i], right=True) - 1
            for i in range(flat_inputs.shape[0])
        ])
        bin_idx = torch.clamp(bin_idx, 0, self.num_bins - 1)

        # Gather all parameters for each [batch, dim] pair
        widths_flat = widths.contiguous().view(-1, widths.shape[-1])
        cumwidths_flat = cum_widths.contiguous().view(-1, cum_widths.shape[-1])
        heights_flat = heights.contiguous().view(-1, heights.shape[-1])
        cumheights_flat = cum_heights.contiguous().view(-1, cum_heights.shape[-1])
        derivatives_flat = derivatives.contiguous().view(-1, derivatives.shape[-1])

        # Gather for each input
        w_k = torch.gather(widths_flat, 1, bin_idx.unsqueeze(-1)).squeeze(-1)
        z_k = torch.gather(cumwidths_flat, 1, bin_idx.unsqueeze(-1)).squeeze(-1)
        h_k = torch.gather(heights_flat, 1, bin_idx.unsqueeze(-1)).squeeze(-1)
        y_k = torch.gather(cumheights_flat, 1, bin_idx.unsqueeze(-1)).squeeze(-1)
        d_k = torch.gather(derivatives_flat, 1, bin_idx.unsqueeze(-1)).squeeze(-1)
        d_k1 = torch.gather(derivatives_flat, 1, (bin_idx + 1).clamp(max=derivatives_flat.shape[1]-1).unsqueeze(-1)).squeeze(-1)
        s_k = h_k / torch.clamp(w_k, min=eps)

        xi = None
        if inverse:
            y = flat_inputs
            # Solve quadratic equation for xi
            a = (y - y_k) * (d_k + d_k1 - 2 * s_k) + h_k * (s_k - d_k)
            b = h_k * d_k - (y - y_k) * (d_k + d_k1 - 2 * s_k)
            c = -s_k * (y - y_k)
            linear_mask = torch.abs(a) < eps
            quadratic_mask = ~linear_mask
            xi = torch.zeros_like(y)
            # Linear case
            if linear_mask.any():
                xi[linear_mask] = -c[linear_mask] / torch.clamp(b[linear_mask], min=eps)
            # Quadratic case
            if quadratic_mask.any():
                a_quad = a[quadratic_mask]
                b_quad = b[quadratic_mask]
                c_quad = c[quadratic_mask]
                discriminant = b_quad.pow(2) - 4 * a_quad * c_quad
                discriminant = torch.clamp(discriminant, min=0)
                sqrt_disc = torch.sqrt(discriminant)
                xi[quadratic_mask] = (-b_quad - sqrt_disc) / (2 * torch.clamp(a_quad, min=eps))
            xi = torch.clamp(xi, 0, 1)
            transformed_outputs = xi * w_k + z_k
            denominator_ld = s_k + (d_k1 + d_k - 2 * s_k) * xi * (1 - xi)
            numerator_ld = s_k.pow(2) * (d_k1 * xi.pow(2) + 2 * s_k * xi * (1 - xi) + d_k * (1 - xi).pow(2))
            numerator_ld = torch.clamp(numerator_ld, min=eps)
            denominator_ld = torch.clamp(denominator_ld, min=eps)
            transformed_logabsdet = -torch.log(numerator_ld) + 2 * torch.log(denominator_ld)
        else:
            x = flat_inputs
            xi = (x - z_k) / torch.clamp(w_k, min=eps)
            xi = torch.clamp(xi, 0, 1)
            denominator = s_k + (d_k1 + d_k - 2 * s_k) * xi * (1 - xi)
            denominator = torch.clamp(denominator, min=eps)
            numerator = h_k * (s_k * xi.pow(2) + d_k * xi * (1 - xi))
            transformed_outputs = y_k + numerator / denominator
            numerator_deriv = s_k.pow(2) * (d_k1 * xi.pow(2) + 2*s_k*xi*(1-xi) + d_k*(1-xi).pow(2))
            denominator_deriv = denominator.pow(2)
            derivative = numerator_deriv / torch.clamp(denominator_deriv, min=eps)
            transformed_logabsdet = torch.log(torch.clamp(derivative, min=eps))

        # Reshape back to [batch_size, num_transformed_dims]
        outputs_flat = transformed_outputs
        logabsdet_flat = transformed_logabsdet
        outputs = outputs.clone().view(-1)
        logabsdet = logabsdet.clone().view(-1)
        outputs[inside_interval_mask.view(-1)] = outputs_flat[inside_interval_mask.view(-1)]
        logabsdet[inside_interval_mask.view(-1)] = logabsdet_flat[inside_interval_mask.view(-1)]
        outputs = outputs.view(batch_size, num_transformed_dims)
        logabsdet = logabsdet.view(batch_size, num_transformed_dims)

        # Handle numerical issues more aggressively
        outputs = torch.where(torch.isnan(outputs) | torch.isinf(outputs), inputs, outputs)
        logabsdet = torch.where(torch.isnan(logabsdet) | torch.isinf(logabsdet), torch.zeros_like(logabsdet), logabsdet)

        return outputs, logabsdet
    
    def _initialize_weights(self):
        """Initialize network weights."""
        for layer in self.param_net[:-1]:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight, gain=1.0)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)
        
        final_layer = self.param_net[-1]
        if isinstance(final_layer, nn.Linear):
            nn.init.zeros_(final_layer.weight)
            if final_layer.bias is not None:
                nn.init.zeros_(final_layer.bias)

class MLP(nn.Module):
    """
    A simple Multi-Layer Perceptron (MLP).
    """
    def __init__(self, input_dim, output_dim, hidden_dim, num_layers, activation='relu'):
        super().__init__()
        
        if activation == 'relu':
            self.activation = F.relu
        elif activation == 'tanh':
            self.activation = torch.tanh
        else:
            raise ValueError(f"Unknown activation function: {activation}")
            
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
        layers.append(nn.Linear(hidden_dim, output_dim))
        
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        """
        Forward pass through the MLP.
        """
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                x = self.activation(x)
        return x

from .flow import Flow, SequentialFlow
from .autoregressive import MADE

class ARQS(Flow):
    """
    Autoregressive flow with rational-quadratic splines (IAF variant).
    This implementation follows the Inverse Autoregressive Flow (IAF) structure,
    where the forward pass is fast (parallel) and the inverse is slow (sequential).
    """
    def __init__(self, dim, hidden_dim=128, num_bins=8, layers=2, data_min=None, data_max=None):
        super().__init__()
        self.dim = dim
        self.num_bins = num_bins
        self.data_min = data_min
        self.data_max = data_max
        output_dim_per_dim = 3 * self.num_bins - 1
        # Use MADE as the conditioner to enforce autoregressive property
        self.conditioner = MADE(
            input_dim=dim,
            hidden_dim=hidden_dim,
            output_dim_multiplier=output_dim_per_dim
        )

    def _rescale_to_unit(self, x):
        """
        Rescale data from [data_min, data_max] to [0, 1].
        """
        if self.data_min is None or self.data_max is None:
            return x
        return (x - self.data_min) / (self.data_max - self.data_min)

    def _rescale_from_unit(self, x):
        """
        Rescale data from [0, 1] back to [data_min, data_max].
        """
        if self.data_min is None or self.data_max is None:
            return x
        return x * (self.data_max - self.data_min) + self.data_min

    def forward(self, z):
        """
        Forward pass (sampling), z -> x. This is slow and sequential (true autoregressive).
        """
        # Rescale input to [0, 1]
        z_rescaled = self._rescale_to_unit(z)
        x_rescaled = torch.zeros_like(z_rescaled)
        log_det_jacobian = torch.zeros(z.size(0), device=z.device)
        # Sequentially compute each dimension of x
        for i in range(self.dim):
            # The conditioner's output for all dimensions depends on the input `x_rescaled`
            params = self.conditioner(x_rescaled)
            b, d = z.shape
            output_dim_per_dim = 3 * self.num_bins - 1
            params = params.view(b, d, output_dim_per_dim)
            # Select parameters for the current dimension
            widths_i = params[:, i, :self.num_bins]
            heights_i = params[:, i, self.num_bins:2*self.num_bins]
            derivatives_i = params[:, i, 2*self.num_bins:]
            # Compute the forward transformation for the current dimension
            x_i_rescaled, log_det_i = rational_quadratic_spline(
                inputs=z_rescaled[:, i],
                widths=widths_i,
                heights=heights_i,
                derivatives=derivatives_i,
                inverse=False
            )
            # Update the input for the next iteration without in-place modification
            x_new = x_rescaled.clone()
            x_new[:, i] = x_i_rescaled
            x_rescaled = x_new
            log_det_jacobian += log_det_i
        # Rescale output back to original data range
        x = self._rescale_from_unit(x_rescaled)
        return x, log_det_jacobian

    def inverse(self, x):
        """
        Inverse pass (density estimation), x -> z. This is slow and sequential.
        """
        # Rescale input to [0, 1]
        x_rescaled = self._rescale_to_unit(x)
        z_rescaled = torch.zeros_like(x_rescaled)
        log_det_jacobian = torch.zeros(x.size(0), device=x.device)
        # Sequentially compute each dimension of z
        for i in range(self.dim):
            # The conditioner's output for all dimensions depends on the input `z_rescaled`
            params = self.conditioner(z_rescaled)
            b, d = x.shape
            output_dim_per_dim = 3 * self.num_bins - 1
            params = params.view(b, d, output_dim_per_dim)
            # Select parameters for the current dimension
            widths_i = params[:, i, :self.num_bins]
            heights_i = params[:, i, self.num_bins:2*self.num_bins]
            derivatives_i = params[:, i, 2*self.num_bins:]
            # Compute the inverse transformation for the current dimension
            z_i_rescaled, log_det_i = rational_quadratic_spline(
                inputs=x_rescaled[:, i],
                widths=widths_i,
                heights=heights_i,
                derivatives=derivatives_i,
                inverse=True
            )
            # Update the input for the next iteration without in-place modification
            z_new = z_rescaled.clone()
            z_new[:, i] = z_i_rescaled
            z_rescaled = z_new
            log_det_jacobian += log_det_i
        # Rescale output back to original data range
        z = self._rescale_from_unit(z_rescaled)
        return z, log_det_jacobian

def rational_quadratic_spline(
    inputs,
    widths,
    heights,
    derivatives,
    inverse=False,
    min_bin_width=1e-3,
    min_bin_height=1e-3,
    min_derivative=1e-3,
    epsilon=1e-6
):
    """
    Implements the rational quadratic spline transformation with robust batching.
    """
    # Add a small epsilon for numerical stability
    epsilon = 1e-6

    # Normalize widths and heights to be positive and sum to 1
    widths = F.softmax(widths, dim=-1)
    heights = F.softmax(heights, dim=-1)
    
    # Add minimums to prevent collapse and clamp for stability
    widths = min_bin_width + (1 - min_bin_width * widths.shape[-1]) * widths
    heights = min_bin_height + (1 - min_bin_height * heights.shape[-1]) * heights
    widths = torch.clamp(widths, min=epsilon)
    heights = torch.clamp(heights, min=epsilon)

    # Ensure derivatives are positive
    derivatives = F.softplus(derivatives) + min_derivative
    derivatives = torch.clamp(derivatives, min=epsilon)
    
    # Precompute knot positions
    x_knots = F.pad(torch.cumsum(widths, dim=-1), (1, 0), 'constant', 0.0)
    y_knots = F.pad(torch.cumsum(heights, dim=-1), (1, 0), 'constant', 0.0)
    
    # Ensure boundary derivatives are 1
    derivatives = F.pad(derivatives, (1, 1), 'constant', 1.0)

    # Find the correct bin for each input value
    if inverse:
        # For inverse, search in y_knots (cum_heights)
        boundaries = y_knots.contiguous()
        if boundaries.dim() > 2:
            boundaries = boundaries.view(-1, boundaries.shape[-1])
        bin_idx = torch.searchsorted(boundaries, inputs.unsqueeze(-1), right=True) - 1
    else:
        # For forward, search in x_knots (cum_widths)
        boundaries = x_knots.contiguous()
        if boundaries.dim() > 2:
            boundaries = boundaries.view(-1, boundaries.shape[-1])
        bin_idx = torch.searchsorted(boundaries, inputs.unsqueeze(-1), right=True) - 1
    
    bin_idx = torch.clamp(bin_idx, 0, widths.shape[-1] - 1)

    # Gather the parameters for each input's bin
    x_k = torch.gather(x_knots, -1, bin_idx)
    y_k = torch.gather(y_knots, -1, bin_idx)
    w_k = torch.gather(widths, -1, bin_idx)
    h_k = torch.gather(heights, -1, bin_idx)
    d_k = torch.gather(derivatives, -1, bin_idx)
    d_k_plus_1 = torch.gather(derivatives, -1, bin_idx + 1)
    
    s_k = h_k / torch.clamp(w_k, min=epsilon)

    inputs_r = inputs.unsqueeze(-1)

    if inverse:
        term1 = (inputs_r - y_k) * (d_k + d_k_plus_1 - 2 * s_k)
        a = h_k * (s_k - d_k) + term1
        b = h_k * d_k - term1
        c = -s_k * (inputs_r - y_k)
        
        discriminant = b.pow(2) - 4 * a * c
        discriminant = torch.clamp(discriminant, min=0)
        
        theta = (2 * c) / (-b - torch.sqrt(discriminant))
        theta = torch.clamp(theta, 0, 1)
        outputs = theta * w_k + x_k
        
        theta_one_minus_theta = theta * (1 - theta)
        nominator = s_k.pow(2) * (d_k_plus_1 * theta.pow(2) + 2 * s_k * theta_one_minus_theta + d_k * (1 - theta).pow(2))
        denominator = (s_k + (d_k + d_k_plus_1 - 2 * s_k) * theta_one_minus_theta).pow(2)
        derivative = nominator / torch.clamp(denominator, min=epsilon)
        log_det = -torch.log(torch.clamp(derivative, min=epsilon))

    else:
        theta = (inputs_r - x_k) / torch.clamp(w_k, min=epsilon)
        theta = torch.clamp(theta, 0, 1)
        theta_one_minus_theta = theta * (1 - theta)
        
        nominator = h_k * (s_k * theta.pow(2) + d_k * theta_one_minus_theta)
        denominator = s_k + (d_k + d_k_plus_1 - 2 * s_k) * theta_one_minus_theta
        
        outputs = y_k + nominator / torch.clamp(denominator, min=epsilon)
        
        nominator_deriv = s_k.pow(2) * (d_k_plus_1 * theta.pow(2) + 2*s_k*theta_one_minus_theta + d_k*(1-theta).pow(2))
        denominator_deriv = denominator.pow(2)
        derivative = nominator_deriv / torch.clamp(denominator_deriv, min=epsilon)
        log_det = torch.log(torch.clamp(derivative, min=epsilon))
        
    return outputs.squeeze(-1), log_det.squeeze(-1)