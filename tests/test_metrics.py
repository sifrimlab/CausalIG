"""Tests for experiments.evaluation.metrics.

These tests use small, deterministic tensors and check the public metric
functions against hand-computed expected values wherever feasible. The autouse
seed fixture in ``tests/conftest.py`` guarantees a deterministic RNG state.
"""

import numpy as np
import pytest
import torch

from experiments.evaluation.metrics import (
    _normalize,
    compare_to_cate,
    evaluate_model_performance,
    mse,
    pearson,
    spearman,
)
from experiments.models.lasso import LassoRegression

# ---------------------------------------------------------------------------
# evaluate_model_performance
# ---------------------------------------------------------------------------


def test_evaluate_model_performance_linear_weights():
    n_features = 4
    model = LassoRegression(n_features=n_features)
    # Set explicit weights: features 0 and 2 are "learned" (non-zero).
    with torch.no_grad():
        model.w.copy_(torch.tensor([1.0, 0.0, 2.0, 0.0]))
        model.b.copy_(torch.tensor([0.0]))

    features = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ]
    )
    # predictions = X @ w + b = [1.0, 2.0]
    target = torch.tensor([1.0, 2.0])
    causal_indices = np.array(
        [0, 1]
    )  # feature 0 causal & learned, 1 causal not learned

    res = evaluate_model_performance(model, features, target, causal_indices)
    assert res["mse"] == pytest.approx(0.0, abs=1e-6)
    assert res["num_nonzero_weights"] == 2  # features 0 and 2
    assert res["num_causal_learned"] == 1  # only feature 0 in causal_indices
    assert res["total_causal"] == 2
    assert res["non_zero_indices"] == {0, 2}
    np.testing.assert_allclose(
        res["learned_weights"], np.array([1.0, 0.0, 2.0, 0.0]), atol=1e-6
    )


def test_evaluate_model_performance_threshold_filters_small_weights():
    model = LassoRegression(n_features=3)
    with torch.no_grad():
        model.w.copy_(torch.tensor([1e-9, 0.5, 1e-3]))
    features = torch.zeros(2, 3)
    target = torch.zeros(2)
    causal_indices = np.array([1])
    # Default threshold 1e-6: 1e-9 is filtered out, 0.5 and 1e-3 kept.
    res = evaluate_model_performance(model, features, target, causal_indices)
    assert res["non_zero_indices"] == {1, 2}
    assert res["num_nonzero_weights"] == 2
    assert res["num_causal_learned"] == 1


def test_evaluate_model_performance_no_weights_attr():
    # Edge case: a model without a ``w`` attribute.
    class NoWeightModel(torch.nn.Module):
        def forward(self, x):
            return x.sum(dim=1)

    model = NoWeightModel()
    features = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    target = torch.tensor([3.0, 7.0])
    res = evaluate_model_performance(model, features, target, np.array([0, 1]))
    assert res["mse"] == pytest.approx(0.0, abs=1e-6)
    assert res["num_nonzero_weights"] == 0
    assert res["num_causal_learned"] == 0
    assert res["total_causal"] == 2
    assert res["learned_weights"] is None


# ---------------------------------------------------------------------------
# helper functions (exercised directly)
# ---------------------------------------------------------------------------


def test_mse_helper():
    a = torch.tensor([[1.0, 2.0]])
    b = torch.tensor([[1.0, 4.0]])
    # ((0)^2 + (2)^2) / 2 == 2.0
    assert mse(a, b) == pytest.approx(2.0, abs=1e-6)


def test_pearson_perfect_positive_correlation():
    a = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    b = 3.0 * a + 5.0  # positive affine transform -> corr == 1
    assert pearson(a, b) == pytest.approx(1.0, abs=1e-5)


def test_pearson_perfect_negative_correlation():
    a = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    b = -2.0 * a
    assert pearson(a, b) == pytest.approx(-1.0, abs=1e-5)


def test_pearson_flatten_mode():
    a = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    b = 2.0 * a
    assert pearson(a, b, row_wise=False) == pytest.approx(1.0, abs=1e-5)


def test_spearman_monotonic_is_one():
    a = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    # Strictly monotonic (non-linear) transform -> spearman == 1.
    b = torch.tensor([[1.0, 4.0, 9.0, 16.0]])
    assert spearman(a, b) == pytest.approx(1.0, abs=1e-5)


def test_normalize_l1_sums_to_one_in_magnitude():
    x = torch.tensor([[1.0, -3.0]])
    n = _normalize(x)
    # L1 normalization: divides by sum of abs (+1e-8) -> abs values sum ~1.
    assert n.abs().sum().item() == pytest.approx(1.0, abs=1e-6)
    torch.testing.assert_close(n, torch.tensor([[0.25, -0.75]]), rtol=0, atol=1e-6)


# ---------------------------------------------------------------------------
# compare_to_cate
# ---------------------------------------------------------------------------


def test_compare_to_cate_keys_and_perfect_correlation():
    # cate is a positive scalar multiple of attr -> perfect row-wise corr.
    attr = torch.tensor([[1.0, 2.0, 3.0, 4.0], [4.0, 3.0, 2.0, 1.0]])
    cate = 2.5 * attr

    res = compare_to_cate(attr, cate)
    assert set(res.keys()) == {"mse", "pearson", "spearman"}

    # Normalized attr and normalized cate are identical (positive scaling),
    # so the MSE between them is ~0.
    assert res["mse"] == pytest.approx(0.0, abs=1e-6)
    # Positive linear relationship -> pearson & spearman ~1.
    assert res["pearson"] == pytest.approx(1.0, abs=1e-5)
    assert res["spearman"] == pytest.approx(1.0, abs=1e-5)


def test_compare_to_cate_nodes_to_compare_subset():
    # Restrict the comparison to a subset of feature indices.
    # NOTE: the source normalizes over ALL features *before* subsetting
    # (see metrics.py: ``_normalize`` runs on the full tensor, then
    # ``attr_n = attr_n[:, nodes_to_compare]``). The raw pearson is computed on
    # the subset only. Here attr == cate on nodes [0, 1] so the *raw*
    # correlation on the subset is perfect (pearson == 1.0), even though the
    # normalized MSE is non-zero because the normalization denominators differ.
    attr = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    cate = torch.tensor([[1.0, 2.0, 30.0, 40.0]])
    res = compare_to_cate(attr, cate, nodes_to_compare=[0, 1])
    assert set(res.keys()) == {"mse", "pearson", "spearman"}
    # Raw values on nodes [0, 1] are perfectly (positively) correlated.
    assert res["pearson"] == pytest.approx(1.0, abs=1e-5)
    # MSE is on the pre-subset-normalized values, which differ in scale, so it
    # is strictly positive (documents the normalize-before-subset ordering).
    assert res["mse"] > 0.0


def test_compare_to_cate_subset_matches_full_when_all_nodes():
    # When nodes_to_compare covers all features, results equal the no-subset
    # call: this confirms subsetting is the only difference.
    attr = torch.tensor([[1.0, 2.0, 3.0, 4.0], [4.0, 3.0, 2.0, 1.0]])
    cate = torch.tensor([[1.0, 3.0, 2.0, 5.0], [2.0, 3.0, 1.0, 4.0]])
    full = compare_to_cate(attr, cate)
    subset = compare_to_cate(attr, cate, nodes_to_compare=[0, 1, 2, 3])
    assert full["mse"] == pytest.approx(subset["mse"], abs=1e-6)
    assert full["pearson"] == pytest.approx(subset["pearson"], abs=1e-6)
    assert full["spearman"] == pytest.approx(subset["spearman"], abs=1e-6)


def test_compare_to_cate_negative_relationship():
    attr = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    cate = -1.0 * attr
    res = compare_to_cate(attr, cate)
    # Raw pearson/spearman on a negatively related signal -> ~ -1.
    assert res["pearson"] == pytest.approx(-1.0, abs=1e-5)
    assert res["spearman"] == pytest.approx(-1.0, abs=1e-5)
