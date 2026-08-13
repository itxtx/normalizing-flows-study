import torch
import torch.nn.functional as F
from ..flow.flow import Flow
from ..autoregressive.made import MADE
from src.flows.spline.rational_quadratic_spline import rational_quadratic_spline

class ARQS(Flow):
    """
    Autoregressive flow with rational-quadratic splines (IAF variant).
    This implementation follows the Inverse Autoregressive Flow (IAF) structure,
    where the forward pass is fast (parallel) and the inverse is slow (sequential).
    """
    def __init__(self, dim, hidden_dim=128, num_bins=8, layers=2, data_min=None, data_max=None, use_batch_norm=False):
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
            output_dim_multiplier=output_dim_per_dim,
            use_batch_norm=use_batch_norm
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
        log_det_jacobian = torch.zeros(z.size(0), device=z.device, dtype=z.dtype)
        # Sequentially compute each dimension of x
        for i in range(self.dim):
            # The conditioner's output for all dimensions depends on the input `x_rescaled`
            params = self.conditioner(x_rescaled)
            b, d = z.shape
            output_dim_per_dim = 3 * self.num_bins - 1
            # MADE groups outputs by parameter, then by dimension. Move the
            # dimension axis ahead of the parameter axis for spline slicing.
            params = params.view(b, output_dim_per_dim, d).transpose(1, 2)
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
        Inverse pass (density estimation), x -> z. This is fast and parallel.
        """
        # Rescale input to [0, 1]
        x_rescaled = self._rescale_to_unit(x)
        batch_size, dim = x.shape
        output_dim_per_dim = 3 * self.num_bins - 1
        params = self.conditioner(x_rescaled)
        params = params.view(batch_size, output_dim_per_dim, dim).transpose(1, 2)
        widths = params[..., :self.num_bins]
        heights = params[..., self.num_bins:2 * self.num_bins]
        derivatives = params[..., 2 * self.num_bins:]
        z_rescaled, log_det = rational_quadratic_spline(
            inputs=x_rescaled,
            widths=widths,
            heights=heights,
            derivatives=derivatives,
            inverse=True,
        )
        log_det_jacobian = log_det.sum(dim=1)
        # Rescale output back to original data range
        z = self._rescale_from_unit(z_rescaled)
        return z, log_det_jacobian
