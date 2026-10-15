"""Gradient-correctness tests for ``MaskedGlobalLinear`` and its basis expansion.

``MaskedGlobalLinear`` ships a *hand-derived* closed-form input Jacobian
(:meth:`MaskedGlobalLinear.compute_input_jacobian`), built on the analytic
:meth:`BasisExpansion.derivative`. This is the gradient path that
``CausalIntegratedGradients`` / ``CausalExpectedGradients`` actually use through
``compute_surrogate_gradients(..., has_closed_form_jacobian=True)``. A wrong derivative
here would silently corrupt every attribution computed with this surrogate, so
these tests pin the analytic gradients against autograd at three levels:

1. the ``BasisExpansion.derivative`` building block,
2. the model-level ``compute_input_jacobian`` (with and without a basis, plus
   chunking consistency), and
3. the end-to-end closed-form vs. autograd paths in
   ``compute_surrogate_gradients``.

All comparisons run in float64 so tolerances can be tight. Most tests keep
inputs inside the basis ``activation_clip`` range; the saturation tests at the
end of the file deliberately push inputs past it (but never onto the kink) to
check that the analytic derivative is zero where the clamp is flat.
"""

import numpy as np
import pytest
import torch

from causalig.attr._utils.surrogate import compute_surrogate_gradients
from causalig.models.surrogate import BasisExpansion, MaskedGlobalLinear


def _make_dag(n=6):
    """A small fixed DAG. Rows are parents, columns are children (mask[i, j]=1
    means i -> j). Node 0 has no parents (not a predictable child)."""
    A = np.zeros((n, n), dtype=np.float32)
    edges = [(0, 1), (0, 2), (1, 2), (1, 3), (2, 4), (3, 4), (2, 5), (4, 5)]
    for i, j in edges:
        A[i, j] = 1.0
    return A


def _autograd_basis_derivative(basis, x):
    """d phi[b, n, k] / d x[b, n] via autograd.

    Each basis element phi[b, n, k] is a function of x[b, n] only, so the
    gradient of ``phi[..., k].sum()`` w.r.t. ``x`` recovers the per-element
    derivative for column ``k``.
    """
    x = x.clone().requires_grad_(True)
    phi = basis(x)  # [B, N, K]
    n_basis = phi.shape[-1]
    cols = [
        torch.autograd.grad(phi[..., k].sum(), x, retain_graph=True)[0]
        for k in range(n_basis)
    ]
    return torch.stack(cols, dim=-1)  # [B, N, K]


def _autograd_input_jacobian(model, x):
    """d model(x)[b, j] / d x[b, n] via autograd -> [B, N_out, N_in]."""
    x = x.clone().requires_grad_(True)
    out = model(x)  # [B, N]
    n_out = out.shape[1]
    cols = []
    for j in range(n_out):
        g = torch.autograd.grad(
            out[:, j].sum(), x, retain_graph=True, allow_unused=True
        )[0]
        cols.append(torch.zeros_like(x) if g is None else g)
    return torch.stack(cols, dim=1)  # [B, N_out, N_in]


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(polynomial_degree=3, n_knots=0, fourier_features=0),
        dict(polynomial_degree=0, n_knots=4, knot_power=2, fourier_features=0),
        dict(polynomial_degree=0, n_knots=4, knot_power=1, fourier_features=0),
        dict(polynomial_degree=2, n_knots=3, knot_power=3, fourier_features=2),
    ],
)
def test_basis_expansion_derivative_matches_autograd(kwargs):
    n_features = 5
    # Non-trivial standardization exercises the chain rule through x_loc/x_scale.
    x_loc = torch.linspace(-1.0, 1.0, n_features)
    x_scale = torch.full((n_features,), 1.2)
    basis = BasisExpansion(
        n_features=n_features, x_loc=x_loc, x_scale=x_scale, **kwargs
    ).double()

    # Keep z = (x - loc) / scale within +/-8 so the activation clip is inactive.
    x = (torch.randn(16, n_features) * 1.5).clamp(-4.0, 4.0).double()

    analytic = basis.derivative(x)
    numeric = _autograd_basis_derivative(basis, x)

    assert analytic.shape == numeric.shape
    torch.testing.assert_close(analytic, numeric, atol=1e-9, rtol=1e-6)


def test_input_jacobian_matches_autograd_linear_no_basis():
    A = _make_dag(6)
    d = A.shape[0]
    model = MaskedGlobalLinear(A, use_basis_expansion=False).double()
    with torch.no_grad():
        model.weight.normal_()
        model.bias.normal_()

    x = torch.randn(7, d, dtype=torch.float64)
    j_model = model.compute_input_jacobian(x, 0, d)
    j_auto = _autograd_input_jacobian(model, x)

    assert j_model.shape == (x.shape[0], d, d)
    torch.testing.assert_close(j_model, j_auto, atol=1e-8, rtol=1e-6)


def test_input_jacobian_matches_autograd_with_basis():
    A = _make_dag(6)
    d = A.shape[0]
    X_train = torch.randn(300, d)
    model = MaskedGlobalLinear(
        A,
        use_basis_expansion=True,
        X_train=X_train,
        n_knots=4,
        knot_power=2,
        polynomial_degree=2,
        fourier_features=1,
    ).double()
    with torch.no_grad():
        model.weight.normal_(0, 0.5)
        model.bias.normal_(0, 0.1)

    x = torch.randn(8, d, dtype=torch.float64)  # loc~0, scale~1 => z within clip
    j_model = model.compute_input_jacobian(x, 0, d)
    j_auto = _autograd_input_jacobian(model, x)

    assert j_model.shape == (x.shape[0], d, d)
    torch.testing.assert_close(j_model, j_auto, atol=1e-7, rtol=1e-5)


def test_input_jacobian_chunking_consistency():
    A = _make_dag(6)
    d = A.shape[0]
    X_train = torch.randn(200, d)
    model = MaskedGlobalLinear(
        A, use_basis_expansion=True, X_train=X_train, n_knots=3, polynomial_degree=2
    ).double()

    x = torch.randn(5, d, dtype=torch.float64)
    full = model.compute_input_jacobian(x, 0, d)
    chunked = torch.cat(
        [
            model.compute_input_jacobian(x, 0, 3),
            model.compute_input_jacobian(x, 3, d),
        ],
        dim=1,
    )
    torch.testing.assert_close(full, chunked, atol=1e-10, rtol=0)


def test_closed_form_jacobian_path_matches_autograd_path():
    """The fast closed-form Jacobian path used by CIG/CEG must agree with the
    generic autograd path of ``compute_surrogate_gradients`` for the same model."""
    A = _make_dag(6)
    d = A.shape[0]
    X_train = torch.randn(200, d)
    model = MaskedGlobalLinear(
        A,
        use_basis_expansion=True,
        X_train=X_train,
        n_knots=4,
        knot_power=2,
        polynomial_degree=2,
    ).double()
    with torch.no_grad():
        model.weight.normal_(0, 0.3)
        model.bias.normal_(0, 0.1)

    device = torch.device("cpu")
    x = torch.randn(10, d, dtype=torch.float64)
    adjacency = torch.as_tensor(A)  # [parents, children]
    ordering = list(range(d))

    grads_lin, sel_lin = compute_surrogate_gradients(
        model,
        x,
        ordering,
        d,
        device,
        adjacency_matrix=adjacency,
        has_closed_form_jacobian=True,
    )
    grads_ag, sel_ag = compute_surrogate_gradients(
        model,
        x,
        ordering,
        d,
        device,
        adjacency_matrix=adjacency,
        has_closed_form_jacobian=False,
    )

    torch.testing.assert_close(sel_lin, sel_ag, atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(grads_lin, grads_ag, atol=1e-6, rtol=1e-5)


def test_closed_form_jacobian_rejects_dependence_on_non_parents():
    """The closed-form path does not mask inputs, so a surrogate built on a
    graph with an edge missing from the attribution graph must be rejected."""
    A = _make_dag(6)
    d = A.shape[0]
    model = MaskedGlobalLinear(A, use_basis_expansion=False).double()
    with torch.no_grad():
        model.weight.normal_(0, 0.3)

    A_missing = A.copy()
    A_missing[3, 4] = 0.0  # the surrogate still reads 3 -> 4

    with pytest.raises(ValueError, match="output 4 depends on feature 3"):
        compute_surrogate_gradients(
            model,
            torch.randn(5, d, dtype=torch.float64),
            list(range(d)),
            d,
            torch.device("cpu"),
            adjacency_matrix=torch.as_tensor(A_missing),
            has_closed_form_jacobian=True,
        )


def test_closed_form_jacobian_requires_compute_input_jacobian():
    d = 4
    with pytest.raises(TypeError, match="compute_input_jacobian"):
        compute_surrogate_gradients(
            lambda x: x,
            torch.randn(5, d),
            list(range(d)),
            d,
            torch.device("cpu"),
            has_closed_form_jacobian=True,
        )


# ---------------------------------------------------------------------------- #
# Saturated inputs: the derivative must respect the ``activation_clip`` clamp.
# ---------------------------------------------------------------------------- #


def _mixed_saturation_inputs(n_rows, n_features, clip, x_loc, x_scale):
    """Inputs whose standardized value z is well inside, well above and well
    below the clip range (never on the boundary, where the clamp is kinked)."""
    z = torch.randn(n_rows, n_features, dtype=torch.float64).clamp(-3.0, 3.0)
    z[0::3] = clip * 2.5  # saturated high
    z[1::3, ::2] = -clip * 2.5  # saturated low (every other feature)
    return z * x_scale.double() + x_loc.double()


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(polynomial_degree=3, n_knots=0, fourier_features=0),
        dict(polynomial_degree=0, n_knots=4, knot_power=2, fourier_features=0),
        dict(polynomial_degree=2, n_knots=3, knot_power=3, fourier_features=2),
    ],
)
def test_basis_expansion_derivative_zero_where_clipped(kwargs):
    """Regression: ``derivative`` used to ignore the clamp in ``_z`` and return
    the unclipped basis slope for saturated inputs, while the forward pass is
    flat there."""
    n_features = 5
    clip = 4.0
    x_loc = torch.linspace(-1.0, 1.0, n_features)
    x_scale = torch.full((n_features,), 1.2)
    basis = BasisExpansion(
        n_features=n_features,
        x_loc=x_loc,
        x_scale=x_scale,
        activation_clip=clip,
        **kwargs,
    ).double()

    x = _mixed_saturation_inputs(12, n_features, clip, x_loc, x_scale)
    saturated = ((x - basis.x_loc) / basis.x_scale).abs() > clip
    assert saturated.any() and (~saturated).any()

    analytic = basis.derivative(x)
    numeric = _autograd_basis_derivative(basis, x)

    torch.testing.assert_close(analytic, numeric, atol=1e-9, rtol=1e-6)
    assert torch.all(analytic[saturated] == 0)
    # Unsaturated entries must not be gated away.
    assert analytic[~saturated].abs().sum() > 0


def test_basis_expansion_derivative_no_clip_is_ungated():
    """With ``activation_clip=None`` large inputs keep their full slope."""
    basis = BasisExpansion(
        n_features=3, polynomial_degree=2, n_knots=0, activation_clip=None
    ).double()
    x = torch.full((2, 3), 50.0, dtype=torch.float64)

    analytic = basis.derivative(x)
    numeric = _autograd_basis_derivative(basis, x)

    torch.testing.assert_close(analytic, numeric, atol=1e-9, rtol=1e-6)
    assert torch.all(analytic != 0)


def test_input_jacobian_matches_autograd_with_saturated_inputs():
    """End-to-end: the closed-form Jacobian of ``MaskedGlobalLinear`` must match
    autograd when some parents fall outside the basis clip range."""
    A = _make_dag(6)
    d = A.shape[0]
    X_train = torch.randn(300, d)
    model = MaskedGlobalLinear(
        A,
        use_basis_expansion=True,
        X_train=X_train,
        n_knots=4,
        knot_power=2,
        polynomial_degree=2,
        fourier_features=1,
    ).double()
    with torch.no_grad():
        model.weight.normal_(0, 0.5)
        model.bias.normal_(0, 0.1)

    basis = model.basis
    clip = basis.activation_clip
    x = _mixed_saturation_inputs(9, d, clip, basis.x_loc, basis.x_scale)

    j_model = model.compute_input_jacobian(x, 0, d)
    j_auto = _autograd_input_jacobian(model, x)
    torch.testing.assert_close(j_model, j_auto, atol=1e-7, rtol=1e-5)

    # Columns for saturated parents must vanish entirely.
    saturated = ((x - basis.x_loc) / basis.x_scale).abs() > clip  # [B, N_in]
    assert saturated.any()
    assert torch.all(j_model.permute(0, 2, 1)[saturated] == 0)
