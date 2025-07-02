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
