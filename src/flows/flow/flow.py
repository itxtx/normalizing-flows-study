import torch.nn as nn

class Flow(nn.Module):
    """
    Base class for normalizing flow layers.
    """
    def __init__(self):
        super().__init__()
        self.data_dim = None

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

    def sample(self, num_samples, base_dist, device='cpu'):
        """
        Samples from the flow.

        Args:
            num_samples (int): The number of samples to generate.
            base_dist (torch.distributions.Distribution): The base distribution.
            device (str): The device to put samples on.

        Returns:
            torch.Tensor: Samples from the transformed distribution.
        """
        z = base_dist.sample((num_samples,)).to(device)
        x, _ = self.forward(z)
        return x

    def log_prob(self, x, base_dist):
        """
        Computes the log probability of a batch of samples.

        Args:
            x (torch.Tensor): A batch of samples from the data space.
            base_dist (torch.distributions.Distribution): The base distribution.

        Returns:
            torch.Tensor: The log probability of each sample.
        """
        z, log_det_inv = self.inverse(x)
        # The log probability of the base distribution is summed over the dimensions
        log_p_z = base_dist.log_prob(z)
        if len(log_p_z.shape) > 1:
            log_p_z = log_p_z.sum(dim=1)
        # The log probability of x is log p(z) + log|det(J_inv)|
        return log_p_z + log_det_inv
