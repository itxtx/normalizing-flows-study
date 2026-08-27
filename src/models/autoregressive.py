import torch

from src.flows.autoregressive.inverse_autoregressive_flow import (
    InverseAutoregressiveFlow,
)
from src.flows.autoregressive.masked_autoregressive_flow import (
    MaskedAutoregressiveFlow,
)
from src.flows.flow.permutation import Permutation
from src.models.normalizing_flow_model import NormalizingFlowModel


class _AutoregressiveFlowModel(NormalizingFlowModel):
    """Compose autoregressive transforms with feature mixing between layers."""

    transform_type = None

    def __init__(
        self,
        data_dim,
        n_layers,
        hidden_dim=64,
        use_permutations=True,
        use_batch_norm=False,
    ):
        if not isinstance(data_dim, int) or data_dim < 1:
            raise ValueError("data_dim must be a positive integer")
        if not isinstance(n_layers, int) or n_layers < 1:
            raise ValueError("n_layers must be a positive integer")
        if not isinstance(hidden_dim, int) or hidden_dim < 1:
            raise ValueError("hidden_dim must be a positive integer")

        transforms = []
        reverse_order = torch.arange(data_dim - 1, -1, -1)
        for index in range(n_layers):
            transforms.append(
                self.transform_type(
                    dim=data_dim,
                    hidden_dim=hidden_dim,
                    use_batch_norm=use_batch_norm,
                )
            )
            if use_permutations and data_dim > 1 and index < n_layers - 1:
                transforms.append(Permutation(reverse_order))

        super().__init__(transforms)
        self.data_dim = data_dim
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.use_permutations = use_permutations


class MAF(_AutoregressiveFlowModel):
    """Canonical Masked Autoregressive Flow model with feature mixing."""

    transform_type = MaskedAutoregressiveFlow


class IAF(_AutoregressiveFlowModel):
    """Canonical Inverse Autoregressive Flow model with feature mixing."""

    transform_type = InverseAutoregressiveFlow
