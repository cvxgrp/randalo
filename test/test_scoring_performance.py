"""Numerical and callback contracts for the optimized scoring path."""

import linops as lo
import numpy as np
import pytest
import scipy.stats
import torch
from randalo import RandALO, truncnorm, utils
from randalo import modeling_layer as ml


def legacy_mean(mu, sigma, a, b):
    """Previous endpoint-stacking path, using unchanged density helpers."""
    mu, sigma, a, b = torch.broadcast_tensors(mu, sigma, a, b)
    alpha, beta = (a - mu) / sigma, (b - mu) / sigma
    nan = torch.isnan(alpha) | torch.isnan(beta)
    valid = ~nan
    alpha, beta = alpha[valid], beta[valid]
    pa, pb = truncnorm._truncnorm_pdf(
        torch.stack((alpha, beta)), alpha[None, :], beta[None, :]
    )
    out = torch.full_like(mu, float("nan"))
    out[nan] = torch.clamp(mu[nan], min=a[nan], max=b[nan])
    out[valid] = mu[valid] - sigma[valid] * (pb - pa)
    return out


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize(
    "device", ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
)
def test_endpoint_and_fallback_parity(dtype, device):
    mu = torch.tensor(
        [
            -0.5,
            0.0,
            0.2,
            1.0,
            1.5,
            -30.0,
            30.0,
            0.0,
            1.0,
            0.5,
            0.5,
            float("nan"),
            float("inf"),
            0.5,
        ],
        dtype=dtype,
        device=device,
    )
    sigma = torch.tensor(
        [
            0.2,
            0.2,
            0.2,
            0.2,
            0.2,
            1.0,
            1.0,
            0.0,
            0.0,
            0.0,
            float("inf"),
            0.1,
            0.1,
            -0.2,
        ],
        dtype=dtype,
        device=device,
    )
    a, b = (
        torch.tensor(0.0, dtype=dtype, device=device),
        torch.tensor(1.0, dtype=dtype, device=device),
    )
    torch.testing.assert_close(
        truncnorm.truncnorm_mean(mu, sigma, a, b),
        legacy_mean(mu, sigma, a, b),
        rtol=0,
        atol=0,
        equal_nan=True,
    )
    # All-zero variance and unbounded intervals exercise distinct branches.
    for case_sigma, case_a, case_b in [
        (torch.zeros_like(mu), a, b),
        (torch.ones_like(mu), a * 0 - float("inf"), b * 0 + float("inf")),
    ]:
        torch.testing.assert_close(
            truncnorm.truncnorm_mean(mu, case_sigma, case_a, case_b),
            legacy_mean(mu, case_sigma, case_a, case_b),
            rtol=0,
            atol=0,
            equal_nan=True,
        )


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_broadcast_strided_and_empty_parity(dtype):
    rng = torch.Generator().manual_seed(12)
    mu = torch.randn(8, 13, 7, dtype=dtype, generator=rng).transpose(0, 2)
    sigma = torch.rand(7, 1, 8, dtype=dtype, generator=rng) + 0.05
    a = torch.full((1, 13, 1), -1.0, dtype=dtype)
    b = torch.tensor(2.0, dtype=dtype)
    for args in [
        (mu, sigma, a, b),
        (mu[:0], sigma[:0], a, b),
        (torch.tensor(0.2, dtype=dtype), torch.tensor(0.1, dtype=dtype), a, b),
    ]:
        torch.testing.assert_close(
            truncnorm.truncnorm_mean(*args),
            legacy_mean(*args),
            rtol=0,
            atol=0,
            equal_nan=True,
        )


def test_mean_against_scipy():
    rng = np.random.default_rng(13)
    mu = rng.uniform(-5, 5, (31, 17))
    sigma = rng.uniform(0.2, 3, (31, 1))
    a, b = -1.0, 2.0
    expected = scipy.stats.truncnorm.mean(
        (a - mu) / sigma, (b - mu) / sigma, loc=mu, scale=sigma
    )
    actual = truncnorm.truncnorm_mean(
        *(torch.as_tensor(x, dtype=torch.float64) for x in (mu, sigma, a, b))
    )
    np.testing.assert_allclose(actual.numpy(), expected, rtol=1e-10, atol=1e-10)


def test_mean_gradient_parity():
    args = [
        torch.tensor(x, dtype=torch.float64, requires_grad=True)
        for x in ([-0.2, 0.2, 0.9, 1.2], [0.2, 0.1, 0.2, 0.3], 0.0, 1.0)
    ]
    old = legacy_mean(*args)
    old_grad = torch.autograd.grad(old.sum(), args)
    new = truncnorm.truncnorm_mean(*args)
    new_grad = torch.autograd.grad(new.sum(), args)
    torch.testing.assert_close(new, old, rtol=0, atol=0)
    for actual, expected in zip(new_grad, old_grad):
        torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("task", ["regression", "logistic"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_subset_predictions_and_callback_contract(task, dtype):
    previous = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        rng = torch.Generator().manual_seed(2)
        n = 39
        x = torch.randn(n, 5, dtype=dtype, generator=rng) / 10
        y = torch.randn(n, dtype=dtype, generator=rng)
        if task == "logistic":
            y = torch.where(y > 0, 1.0, -1.0)
        fitted = torch.randn(n, dtype=dtype, generator=rng)
        loss = ml.MSELoss() if task == "regression" else ml.LogisticLoss()
        ds = utils.compute_derivatives(loss, y, fitted)
        normalized = lo.MatrixOperator(x @ x.T)
        jac = normalized @ lo.DiagonalOperator(-ds.d2loss_dboth / ds.d2loss_dy_hat2)
        ra = RandALO(loss=loss, jac=jac, y=y, y_hat=fitted, dtype=dtype, rng=rng)
        ra.evaluate_bks(lambda y, z: loss(y, z).item(), n_matvecs=17)
        subsets = [list(range(k)) for k in [8, 8, 12, 17]]
        mixer = utils.create_mixing_matrix(17, subsets)
        h = ra._uniform_map_estimates(
            ra._normalized_diag_jac_estims @ mixer, (mixer > 0).sum(0, keepdim=True)
        )
        expected = [
            ra._y_tilde_from_normalized_jac(h[:, j]) for j in range(len(subsets))
        ]
        observed = []

        def risk(labels, predictions):
            assert labels is ra._y
            assert predictions.shape == (n,) and predictions.is_contiguous()
            observed.append(predictions.clone())
            # A nonseparable risk prevents treating columns as one larger batch.
            return ((labels - predictions).mean() ** 2 + predictions.var()).item()

        score = ra.evaluate(risk, n_matvecs=17, subsets=subsets)
        assert len(observed) == len(subsets)
        for actual, wanted in zip(observed, expected):
            torch.testing.assert_close(actual, wanted, rtol=0, atol=0)
        risks = [((y - z).mean() ** 2 + z.var()).item() for z in expected]
        wanted = utils.robust_y_intercept(
            1 / torch.tensor([[len(s) for s in subsets]]), risks
        )
        assert score == wanted
    finally:
        torch.set_default_dtype(previous)
