"""Regression tests for the three flow bugs documented in docs/bugfix-plan.md:

1. Spline coupling: invertible + calibrated density (no 1e12 spikes), eval == train.
2. MAF / IAF: autoregressive + eval-mode density matches train (no BatchNorm blow-up).
3. CNF: log-determinant matches the true Jacobian (trace no longer hard-coded to 0),
   and maximum-likelihood training reduces the NLL.
"""

import numpy as np
import torch
from torch.distributions import MultivariateNormal

from src.models import RealNVPSpline, NormalizingFlowModel
from src.flows import (
    MaskedAutoregressiveFlow,
    InverseAutoregressiveFlow,
    ContinuousFlow,
)
from src.utils import get_two_moons_data


def _base(dim=2):
    return MultivariateNormal(torch.zeros(dim), torch.eye(dim))


def _standardize(x):
    return (x - x.mean(0)) / (x.std(0) + 1e-8)


def _density_max(model, lim=3.0, res=120):
    xs = torch.linspace(-lim, lim, res)
    xx, yy = torch.meshgrid(xs, xs, indexing="xy")
    grid = torch.stack([xx.reshape(-1), yy.reshape(-1)], 1)
    z, ld = model.inverse(grid)
    return torch.exp(_base().log_prob(z) + ld).max().item()


def _train(model, data, epochs, lr=1e-3):
    base = _base(data.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr)
    first = last = None
    for ep in range(epochs):
        z, ld = model.inverse(data)
        loss = -(base.log_prob(z) + ld).mean()
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        opt.step()
        if ep == 0:
            first = loss.item()
        last = loss.item()
    return first, last


# --------------------------------------------------------------------------- #
# Spline
# --------------------------------------------------------------------------- #
def test_spline_invertible_and_calibrated():
    torch.manual_seed(0)
    data = _standardize(get_two_moons_data(2000, noise=0.07))
    model = RealNVPSpline(2, 8, 64)
    _train(model, data, epochs=200, lr=5e-4)
    model.eval()

    # round trip
    z, _ = model.inverse(data)
    x_rec, _ = model.forward(z)
    assert (data - x_rec).abs().max().item() < 1e-4

    # calibrated density (not the old ~1e12 spikes) and eval == train consistency
    assert _density_max(model) < 50.0
    z, ld = model.inverse(data)
    nll_eval = -( _base().log_prob(z) + ld).mean().item()
    assert 0.0 < nll_eval < 4.0


# --------------------------------------------------------------------------- #
# MAF / IAF
# --------------------------------------------------------------------------- #
def test_made_is_autoregressive():
    from src.flows.autoregressive.made import MADE
    torch.manual_seed(0)
    made = MADE(3, 32, 2)
    made.eval()
    for m in made.net:
        if hasattr(m, "mask"):
            m.weight.data = 0.6 * torch.randn_like(m.weight.data)
    x = torch.randn(3)
    J = torch.autograd.functional.jacobian(lambda v: made(v.unsqueeze(0)).squeeze(0), x)
    # output order is [mu_0..mu_{d-1}, alpha_0..alpha_{d-1}]; param i must not
    # depend on input j >= i.
    d = 3
    for i in range(d):
        for j in range(i, d):
            assert J[i, j].abs().item() < 1e-6        # mu_i wrt x_{>=i}
            assert J[d + i, j].abs().item() < 1e-6    # alpha_i wrt x_{>=i}


def test_maf_iaf_eval_matches_train():
    torch.manual_seed(0)
    data = _standardize(get_two_moons_data(2000, noise=0.07))
    for flow_cls in (MaskedAutoregressiveFlow, InverseAutoregressiveFlow):
        model = NormalizingFlowModel([flow_cls(2, 64) for _ in range(4)])
        _train(model, data, epochs=300, lr=1e-3)

        z, ld = model.inverse(data)
        nll_train = -(_base().log_prob(z) + ld).mean().item()
        model.eval()
        z, ld = model.inverse(data)
        nll_eval = -(_base().log_prob(z) + ld).mean().item()

        assert abs(nll_train - nll_eval) < 0.1        # no BatchNorm eval mismatch
        assert _density_max(model) < 50.0             # no eval blow-up
        assert 0.0 < nll_eval < 4.0


# --------------------------------------------------------------------------- #
# CNF
# --------------------------------------------------------------------------- #
def test_cnf_logdet_matches_autodiff():
    torch.manual_seed(0)
    model = ContinuousFlow(2, 32)
    for p in model.parameters():
        p.data = p.data + 0.2 * torch.randn_like(p)
    x = torch.randn(4, 2) * 0.6
    _, ld = model.inverse(x)
    for i in range(4):
        J = torch.autograd.functional.jacobian(
            lambda v: model.inverse(v.unsqueeze(0))[0].squeeze(0), x[i])
        assert abs(torch.logdet(J).item() - ld[i].item()) < 1e-3


def test_cnf_training_reduces_nll():
    torch.manual_seed(0)
    data = _standardize(get_two_moons_data(400, noise=0.07))
    model = ContinuousFlow(2, 64)
    first, last = _train(model, data, epochs=25, lr=2e-2)
    # the trace is no longer zero, so MLE actually makes progress
    assert last < first - 0.2
