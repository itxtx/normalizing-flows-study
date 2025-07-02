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

    def forward(self, z):
        """
        Computes the forward pass x = f(z). (z -> x)
        """
        z_a = z * self.mask
        
        unnormalized_widths, unnormalized_heights, unnormalized_derivatives = self._get_spline_params(z_a)
        
        num_transformed_dims = int((self.mask == 0).sum())

        z_b = z[:, self.mask == 0]
        
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

        x = torch.empty_like(z)
        x[:, self.mask == 1] = z[:, self.mask == 1]
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
        x_a = x * self.mask

        unnormalized_widths, unnormalized_heights, unnormalized_derivatives = self._get_spline_params(x_a)
        
        num_transformed_dims = int((self.mask == 0).sum())
        
        x_b = x[:, self.mask == 0]
        
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

        z = torch.empty_like(x)
        z[:, self.mask == 1] = x[:, self.mask == 1]
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
        """
        batch_size, num_transformed_dims = inputs.shape
        device = inputs.device
        
        inside_interval_mask = (inputs >= -self.bound) & (inputs <= self.bound)
        outside_interval_mask = ~inside_interval_mask

        outputs = torch.zeros_like(inputs)
        logabsdet = torch.zeros_like(inputs)

        outputs[outside_interval_mask] = inputs[outside_interval_mask]
        logabsdet[outside_interval_mask] = 0

        if not inside_interval_mask.any():
            return outputs, logabsdet

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

        derivatives = self.min_derivative + F.softplus(unnormalized_derivatives)
        derivatives = F.pad(derivatives, pad=(1, 1), mode='constant', value=1.0)
        
        idx = torch.nonzero(inside_interval_mask, as_tuple=False)
        b_idx, f_idx = idx[:, 0], idx[:, 1]
        x_in = inputs[b_idx, f_idx]

        if inverse:
            bin_idx = torch.stack([
                torch.searchsorted(cum_heights[b, f], x, right=True) - 1
                for b, f, x in zip(b_idx, f_idx, x_in)
            ])
            bin_idx = bin_idx.clamp(min=0, max=self.num_bins - 1)
            widths_sel = widths[b_idx, f_idx, bin_idx]
            cumwidths_sel = cum_widths[b_idx, f_idx, bin_idx]
            heights_sel = heights[b_idx, f_idx, bin_idx]
            cumheights_sel = cum_heights[b_idx, f_idx, bin_idx]
            derivatives_sel = derivatives[b_idx, f_idx, bin_idx]
            max_bin_idx = derivatives.shape[-1] - 1
            bin_idx_plus1 = torch.clamp(bin_idx + 1, max=max_bin_idx)
            derivatives_plus1_sel = derivatives[b_idx, f_idx, bin_idx_plus1]
            s_k = heights_sel / widths_sel

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

            denominator_ld = s_k + (d_k1 + d_k - 2 * s_k) * xi * (1 - xi)
            numerator_ld = s_k.pow(2) * (d_k1 * xi.pow(2) + 2 * s_k * xi * (1 - xi) + d_k * (1 - xi).pow(2))
            numerator_ld = torch.clamp(numerator_ld, min=1e-8)
            denominator_ld = torch.clamp(denominator_ld, min=1e-8)
            logabsdet_inside = -torch.log(numerator_ld) + 2 * torch.log(denominator_ld)
            logabsdet[b_idx, f_idx] = logabsdet_inside
        else:
            bin_idx = torch.stack([
                torch.searchsorted(cum_widths[b, f], x, right=True) - 1
                for b, f, x in zip(b_idx, f_idx, x_in)
            ])
            bin_idx = bin_idx.clamp(min=0, max=self.num_bins - 1)
            widths_sel = widths[b_idx, f_idx, bin_idx]
            cumwidths_sel = cum_widths[b_idx, f_idx, bin_idx]
            heights_sel = heights[b_idx, f_idx, bin_idx]
            cumheights_sel = cum_heights[b_idx, f_idx, bin_idx]
            derivatives_sel = derivatives[b_idx, f_idx, bin_idx]
            max_bin_idx = derivatives.shape[-1] - 1
            bin_idx_plus1 = torch.clamp(bin_idx + 1, max=max_bin_idx)
            derivatives_plus1_sel = derivatives[b_idx, f_idx, bin_idx_plus1]
            s_k = heights_sel / widths_sel

            x = x_in
            z_k = cumwidths_sel
            w_k = widths_sel
            h_k = heights_sel
            d_k = derivatives_sel
            d_k1 = derivatives_plus1_sel
            s_k = h_k / w_k
            xi = (x - z_k) / w_k
            xi = torch.clamp(xi, 0, 1)

            denominator = s_k + (d_k1 + d_k - 2 * s_k) * xi * (1 - xi)
            numerator = h_k * (s_k * xi.pow(2) + d_k * xi * (1 - xi))
            y = cumheights_sel + numerator / denominator
            outputs[b_idx, f_idx] = y

            numerator_ld = s_k.pow(2) * (d_k1 * xi.pow(2) + 2 * s_k * xi * (1 - xi) + d_k * (1 - xi).pow(2))
            denominator_ld = s_k + (d_k1 + d_k - 2 * s_k) * xi * (1 - xi)
            numerator_ld = torch.clamp(numerator_ld, min=1e-8)
            denominator_ld = torch.clamp(denominator_ld, min=1e-8)
            logabsdet_inside = torch.log(numerator_ld) - 2 * torch.log(denominator_ld)
            logabsdet[b_idx, f_idx] = logabsdet_inside

        outputs = torch.where(torch.isnan(outputs) | torch.isinf(outputs), torch.zeros_like(outputs), outputs)
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

class ARQS(nn.Module):
    def __init__(self, dim, num_bins=8, hidden_dim=128, num_layers=2):
        super().__init__()
        self.dim = dim
        self.num_bins = num_bins
        
        output_dim_per_dim = 3 * self.num_bins - 1
        
        self.conditioner = MLP(
            input_dim=self.dim,
            output_dim=self.dim * output_dim_per_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            activation='relu'
        )

    def forward(self, x):
        params = self.conditioner(x)
        
        b, d = x.shape
        output_dim_per_dim = 3 * self.num_bins - 1
        params = params.view(b, d, output_dim_per_dim)
        
        widths = params[..., :self.num_bins]
        heights = params[..., self.num_bins:2*self.num_bins]
        derivatives = params[..., 2*self.num_bins:]
        
        z, log_det = rational_quadratic_spline(
            inputs=x,
            widths=widths,
            heights=heights,
            derivatives=derivatives
        )
        
        log_det_jacobian = torch.sum(log_det, dim=1)

        return z, log_det_jacobian

    def inverse(self, z):
        x = torch.zeros_like(z)
        
        for i in range(self.dim):
            params = self.conditioner(x)
            
            b, d = z.shape
            output_dim_per_dim = 3 * self.num_bins - 1
            params = params.view(b, d, output_dim_per_dim)
            
            widths_i = params[..., i, :self.num_bins]
            heights_i = params[..., i, self.num_bins:2*self.num_bins]
            derivatives_i = params[..., i, 2*self.num_bins:]
            
            x_i, _ = rational_quadratic_spline(
                inputs=z[:, i],
                widths=widths_i,
                heights=heights_i,
                derivatives=derivatives_i,
                inverse=True
            )
            x[:, i] = x_i
        
        _, log_det_jacobian = self.forward(x)
        
        return x, log_det_jacobian

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
    Implements the rational quadratic spline transformation.
    """
    inputs = torch.clamp(inputs, 0, 1)

    widths = F.softmax(widths, dim=-1)
    heights = F.softmax(heights, dim=-1)
    
    widths = min_bin_width + (1 - min_bin_width * widths.shape[-1]) * widths
    heights = min_bin_height + (1 - min_bin_height * heights.shape[-1]) * heights

    derivatives = F.softplus(derivatives) + min_derivative
    
    x_knots = F.pad(torch.cumsum(widths, dim=-1), (1, 0), 'constant', 0.0)
    y_knots = F.pad(torch.cumsum(heights, dim=-1), (1, 0), 'constant', 0.0)
    
    derivatives = F.pad(derivatives, (1, 1), 'constant', 1.0)
    
    if inverse:
        bin_idx = torch.searchsorted(y_knots, inputs)
    else:
        bin_idx = torch.searchsorted(x_knots, inputs)

    bin_idx = torch.clamp(bin_idx, 1, widths.shape[-1]).unsqueeze(-1)

    x_k = torch.gather(x_knots, -1, bin_idx - 1)
    y_k = torch.gather(y_knots, -1, bin_idx - 1)
    
    w_k = torch.gather(widths, -1, bin_idx - 1)
    h_k = torch.gather(heights, -1, bin_idx - 1)
    
    d_k = torch.gather(derivatives, -1, bin_idx - 1)
    d_k_plus_1 = torch.gather(derivatives, -1, bin_idx)
    
    s_k = h_k / w_k

    inputs_r = inputs.unsqueeze(-1)
    
    if inverse:
        term1 = (inputs_r - y_k) * (d_k + d_k_plus_1 - 2 * s_k)
        term2 = h_k * (s_k - d_k)
        
        a = h_k * (s_k - d_k) + term1
        b = h_k * d_k - term1
        c = -s_k * (inputs_r - y_k)
        
        discriminant = b.pow(2) - 4 * a * c
        discriminant = torch.clamp(discriminant, min=0)
        
        theta = (2 * c) / (-b - discriminant.sqrt())
        outputs = theta * w_k + x_k
        
        theta_one_minus_theta = theta * (1 - theta)
        nominator = s_k.pow(2) * (d_k_plus_1 * theta.pow(2) + 2 * s_k * theta_one_minus_theta + d_k * (1 - theta).pow(2))
        denominator = (s_k + (d_k + d_k_plus_1 - 2 * s_k) * theta_one_minus_theta).pow(2)
        derivative = nominator / (denominator + epsilon)
        log_det = -torch.log(derivative)

    else:
        theta = (inputs_r - x_k) / w_k
        theta_one_minus_theta = theta * (1 - theta)
        
        nominator = h_k * (s_k * theta.pow(2) + d_k * theta_one_minus_theta)
        denominator = s_k + (d_k + d_k_plus_1 - 2 * s_k) * theta_one_minus_theta
        
        outputs = y_k + nominator / (denominator + epsilon)
        
        nominator_deriv = s_k.pow(2) * (d_k_plus_1 * theta.pow(2) + 2*s_k*theta_one_minus_theta + d_k*(1-theta).pow(2))
        denominator_deriv = denominator.pow(2)
        derivative = nominator_deriv / (denominator_deriv + epsilon)
        log_det = torch.log(derivative)
        
    return outputs.squeeze(-1), log_det.squeeze(-1)