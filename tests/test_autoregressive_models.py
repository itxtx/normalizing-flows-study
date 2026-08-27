import pytest
import torch

from src.flows import (
    InverseAutoregressiveFlow,
    MaskedAutoregressiveFlow,
    Permutation,
)
from src.models import IAF, MAF


@pytest.mark.parametrize(
    ("model_type", "transform_type"),
    [
        (MAF, MaskedAutoregressiveFlow),
        (IAF, InverseAutoregressiveFlow),
    ],
)
def test_default_stack_mixes_features_between_transforms(model_type, transform_type):
    model = model_type(data_dim=4, n_layers=3, hidden_dim=16)
    transforms = list(model.flows)

    assert [type(transform) for transform in transforms] == [
        transform_type,
        Permutation,
        transform_type,
        Permutation,
        transform_type,
    ]
    assert torch.equal(transforms[1].permutation, torch.tensor([3, 2, 1, 0]))
    assert torch.equal(transforms[3].permutation, torch.tensor([3, 2, 1, 0]))


@pytest.mark.parametrize("model_type", [MAF, IAF])
def test_permutations_can_be_disabled_explicitly(model_type):
    model = model_type(
        data_dim=3,
        n_layers=2,
        hidden_dim=8,
        use_permutations=False,
    )

    assert len(model.flows) == 2
    assert not any(isinstance(transform, Permutation) for transform in model.flows)


@pytest.mark.parametrize("model_type", [MAF, IAF])
def test_model_round_trip_and_logdet_cancellation(model_type):
    torch.manual_seed(0)
    model = model_type(data_dim=3, n_layers=3, hidden_dim=16).eval()
    latent = torch.randn(12, 3)

    samples, forward_logdet = model(latent)
    reconstructed, inverse_logdet = model.inverse(samples)

    assert torch.allclose(reconstructed, latent, atol=1e-5, rtol=1e-5)
    assert torch.allclose(
        forward_logdet + inverse_logdet,
        torch.zeros_like(forward_logdet),
        atol=1e-5,
        rtol=1e-5,
    )


@pytest.mark.parametrize(
    ("argument", "value"),
    [("data_dim", 0), ("n_layers", 0), ("hidden_dim", 0)],
)
def test_model_rejects_invalid_dimensions(argument, value):
    kwargs = {"data_dim": 2, "n_layers": 2, "hidden_dim": 8}
    kwargs[argument] = value

    with pytest.raises(ValueError):
        MAF(**kwargs)
