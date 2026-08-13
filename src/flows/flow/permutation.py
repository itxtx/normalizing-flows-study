import torch

from .flow import Flow


class Permutation(Flow):
    """Permute feature order between flow layers without changing density."""

    def __init__(self, permutation):
        super().__init__()
        permutation = torch.as_tensor(permutation, dtype=torch.long)
        if permutation.ndim != 1:
            raise ValueError("permutation must be one-dimensional")
        expected = torch.arange(permutation.numel())
        if not torch.equal(torch.sort(permutation).values.cpu(), expected):
            raise ValueError("permutation must contain each feature index exactly once")

        self.data_dim = permutation.numel()
        self.register_buffer("permutation", permutation)
        self.register_buffer("inverse_permutation", torch.argsort(permutation))

    def forward(self, z):
        return z[:, self.permutation], z.new_zeros(z.size(0))

    def inverse(self, x):
        return x[:, self.inverse_permutation], x.new_zeros(x.size(0))
