from functools import partial

import numpy as np
import pytest
import torch

# CausalSHAP depends on the optional `shap` extra; skip the whole module cleanly
# (rather than erroring at collection) when it is not installed.
pytest.importorskip("shap")

from causalig.attr._core.causal_shap import CausalMasker, CausalSHAP, gaussian_noise
from causalig.models.scm import DifferentiableSCM


def _build_simple_scm():
    # DAG: x0 -> x1 -> x2 and x0 -> x2
    adjacency = np.array(
        [
            [0, 1, 1],
            [0, 0, 1],
            [0, 0, 0],
        ]
    )
    assignments = [
        None,
        lambda Xp: 2.0 * Xp[:, 0],
        lambda Xp: Xp[:, 0] + Xp[:, 1],
    ]
    return DifferentiableSCM(
        adjacency_matrix=adjacency, assignments=assignments, topological_order=[0, 1, 2]
    )


def test_gaussian_noise_shape_and_zero_sigma():
    noise = gaussian_noise(batch=7, p=4, sigma=0.0)
    assert noise.shape == (7, 4)
    assert torch.allclose(noise, torch.zeros_like(noise))


def test_causal_masker_returns_expected_shape():
    scm = _build_simple_scm()
    noise = partial(gaussian_noise, sigma=0.1)
    masker = CausalMasker(
        scm=scm,
        noise_dist=noise,
        n_samples=5,
        device=torch.device("cpu"),
    )

    mask = np.array([True, False, True])
    x = np.array([1.0, 2.0, 3.0], dtype=np.float32)

    out = masker(mask, x)

    assert isinstance(out, np.ndarray)
    assert out.shape == (5, 3)


def test_causal_masker_intervened_features_are_fixed():
    scm = _build_simple_scm()
    noise = partial(gaussian_noise, sigma=0.0)
    masker = CausalMasker(
        scm=scm,
        noise_dist=noise,
        n_samples=6,
        device=torch.device("cpu"),
    )

    mask = np.array([True, False, True])
    x = np.array([1.5, -4.0, 7.25], dtype=np.float32)
    out = masker(mask, x)

    assert np.allclose(out[:, 0], x[0])
    assert np.allclose(out[:, 2], x[2])


def test_causal_masker_empty_coalition_path():
    scm = _build_simple_scm()
    noise = partial(gaussian_noise, sigma=0.0)
    masker = CausalMasker(
        scm=scm,
        noise_dist=noise,
        n_samples=4,
        device=torch.device("cpu"),
    )

    mask = np.array([False, False, False])
    x = np.array([10.0, 11.0, 12.0], dtype=np.float32)
    out = masker(mask, x)

    # With no intervention + zero noise + zero inputs for roots, SCM yields zeros here.
    assert out.shape == (4, 3)
    assert np.allclose(out, 0.0)


def test_causal_masker_reproducible_with_manual_seed():
    scm = _build_simple_scm()
    masker = CausalMasker(
        scm=scm, noise_dist=gaussian_noise, n_samples=5, device=torch.device("cpu")
    )

    mask = np.array([True, False, False])
    x = np.array([0.5, 0.0, 0.0], dtype=np.float32)

    torch.manual_seed(123)
    out_1 = masker(mask, x)
    torch.manual_seed(123)
    out_2 = masker(mask, x)

    assert np.allclose(out_1, out_2)


def test_causal_shap_attribute_smoke():
    scm = _build_simple_scm()

    class LinearModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(3, 1, bias=False)
            with torch.no_grad():
                self.linear.weight.copy_(
                    torch.tensor([[1.0, 0.5, -0.25]], dtype=torch.float32)
                )

        def forward(self, x):
            return self.linear(x).squeeze(-1)

    model = LinearModel().to(torch.device("cpu"))
    explainer = CausalSHAP(model, scm)

    inputs = torch.tensor([[1.0, 2.0, 3.0], [0.5, -1.0, 4.0]], dtype=torch.float32)
    attrs = explainer.attribute(inputs)

    attrs_arr = np.asarray(attrs)
    assert attrs_arr.shape[0] == inputs.shape[0]
    assert attrs_arr.shape[1] == inputs.shape[1]
    assert np.isfinite(attrs_arr).all()


def test_causal_shap_reproducible_with_seed():
    # shap reseeds NumPy's global RNG when the explainer is built (from fresh
    # entropy if no seed is given), so the seed has to reach the explainer.
    scm = _build_simple_scm()
    model = torch.nn.Linear(3, 1)
    background = torch.randn(100, 3)
    inputs = torch.tensor([[1.0, 2.0, 3.0], [0.5, -1.0, 4.0]])

    def run():
        np.random.seed(None)  # scramble global state between the two runs
        torch.manual_seed(0)  # SCM noise comes from torch's global RNG
        explainer = CausalSHAP(
            model, scm, background=background, n_samples=10, noise_sigma=0.5, seed=0
        )
        return np.asarray(explainer.attribute(inputs))

    np.testing.assert_array_equal(run(), run())
