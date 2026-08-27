import torch
import torch.nn as nn
import pytest

from plots import _common as C


class ShiftFlow(nn.Module):
    def __init__(self):
        super().__init__()
        self.shift = nn.Parameter(torch.tensor([0.0, 0.0]))

    def forward(self, latent):
        return latent + self.shift, latent.new_zeros(len(latent))

    def inverse(self, data):
        return data - self.shift, data.new_zeros(len(data))


class BrokenFlow(ShiftFlow):
    def forward(self, latent):
        return latent + self.shift + 1.0, latent.new_ones(len(latent))


def test_train_validation_split_is_deterministic_and_disjoint():
    data = torch.arange(40).reshape(20, 2)
    first_train, first_validation = C.train_validation_split(data, seed=7)
    second_train, second_validation = C.train_validation_split(data, seed=7)

    assert torch.equal(first_train, second_train)
    assert torch.equal(first_validation, second_validation)
    assert len(first_train) == 16
    assert len(first_validation) == 4
    assert set(map(tuple, first_train.tolist())).isdisjoint(
        map(tuple, first_validation.tolist())
    )


def test_training_restores_an_improving_validation_checkpoint():
    torch.manual_seed(0)
    data = torch.randn(200, 2) * 0.25 + 2.0
    train_data, validation_data = C.train_validation_split(data, seed=0)
    model = ShiftFlow()

    result = C.train(model, train_data, validation_data, epochs=60, lr=0.1)
    quality = C.validate_cache_candidate(model, validation_data, result)

    assert result["best_epoch"] is not None
    assert result["best_validation_nll"] < result["baseline_validation_nll"]
    assert quality["ok"]
    assert quality["max_roundtrip_error"] < 1e-6
    assert quality["max_logdet_error"] < 1e-6


def test_quality_gate_rejects_a_broken_contract():
    data = torch.randn(32, 2)
    training_result = {
        "baseline_validation_nll": 3.0,
        "best_validation_nll": 2.0,
        "stopped_nonfinite": False,
    }

    quality = C.validate_cache_candidate(BrokenFlow(), data, training_result)

    assert not quality["ok"]
    assert "round-trip error exceeds tolerance" in quality["reasons"]
    assert "log-determinant cancellation error exceeds tolerance" in quality["reasons"]


def test_rejected_candidate_does_not_overwrite_existing_cache(tmp_path, monkeypatch):
    destination = tmp_path / "existing.pt"
    destination.write_bytes(b"existing cache")
    monkeypatch.setattr(C, "cache_path", lambda dataset, flow: str(destination))

    with pytest.raises(ValueError, match="refusing cache candidate"):
        C.save_cache(
            "moons",
            "broken",
            BrokenFlow(),
            {"train_curve": [], "validation_curve": []},
            {"ok": False, "reasons": ["failed quality gate"]},
            train_time=0.0,
        )

    assert destination.read_bytes() == b"existing cache"
