import torch
import torch.nn.functional as F

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
