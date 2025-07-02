import torch
import torch.nn as nn
from torch.nn import functional as F
from src.flows.flow.flow import Flow

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
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
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
