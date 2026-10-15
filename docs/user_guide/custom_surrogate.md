# Adding a new surrogate

Any model of the causal mechanisms can serve as the surrogate. You might to add a new one if:

- your mechanisms are interactive (which the additive {class}`~causalig.models.MaskedGlobalLinear`
  cannot model);
- you already have a fitted mechanistic model;
- or a different function class that suits your data better.

This page specifies exactly what CIG/CEG need from a surrogate, then builds a working example. It ends
with a checklist for validating a new surrogate and for contributing it to the library.

## The contract

A surrogate is a callable mapping a float tensor `(N, d)` to `(N, d)`, where **output column `j` predicts
feature `j` from the parents of `j` only**.
{class}`~causalig.attr._utils.surrogate.ModelInterface` documents the interface, but subclassing it is
optional.

How the surrogate is called depends on `has_closed_form_jacobian`:

| | Autograd path (`has_closed_form_jacobian=False`, default) | Closed-form path (`has_closed_form_jacobian=True`) |
|---|---|---|
| **What is called** | `surrogate(x_masked)` | `surrogate(x)` and `surrogate.compute_input_jacobian(x, dim_start, dim_end)` |
| **Input** | For each non-root child `j`, a copy of every path point where all features that are *not* parents of `j` are set to `0` ({data}`~causalig.attr._utils.surrogate.MASK_VAL`) | The unmasked path points, `(N, d)` |
| **What is read** | Column `j` of the output row built for child `j` | Every non-root column of `surrogate(x)`, plus the Jacobian block |
| **Jacobian** | Computed by autograd, then masked to `j`'s parents | Returned by your method, then masked to `j`'s parents |
| **Who enforces "parents only"** | The masking, so a generic network over all `d` inputs works | **Your model.** `surrogate(x)[:, j]` must not depend on non-parents |
| **Requirements** | Differentiable in its input; rows processed independently | `compute_input_jacobian` returns `(N, dim_end - dim_start, d)` |
| **Cost** | Runs the surrogate on `N × (#children in chunk)` rows | One forward pass, plus your Jacobian |

A few more rules apply on both paths:

- **Roots are ignored.** Outputs for features without parents are never read. They are treated as
  `0`, so $h = x$ for roots.
- **Rows must be independent.** The surrogate sees large batches built from many path points.
  Batch-norm in training mode or any cross-row operation would leak information between samples, so
  call `.eval()` first.
- **Same graph everywhere.** Pass CIG/CEG the same `adjacency_matrix` the surrogate was built for.

## Example: a linear SEM with a closed-form Jacobian

The simplest useful surrogate is a linear structural equation model,
$x_j \approx b_j + \sum_{i \in \mathrm{pa}(j)} W_{ij} x_i$. It can be fit by ordinary least squares, with
no training loop. Its Jacobian is the constant $W_{ij}$, so it can implement the closed-form path:

```python
import torch
from torch import nn


class LinearSEMSurrogate(nn.Module):
    """Linear structural equation surrogate: x_j ≈ b_j + sum_{i in pa(j)} W[i, j] x_i."""

    def __init__(self, adjacency_matrix):
        super().__init__()
        adjacency_matrix = torch.as_tensor(adjacency_matrix).float()
        d = adjacency_matrix.shape[0]
        self.register_buffer("mask", adjacency_matrix)
        self.register_buffer("predictable_children", torch.where(adjacency_matrix.sum(dim=0) > 0)[0])
        self.weight = nn.Parameter(torch.zeros(d, d))
        self.bias = nn.Parameter(torch.zeros(d))

    @torch.no_grad()
    def fit(self, X):
        """Ordinary least squares, one regression per child on its parents."""
        for j in self.predictable_children.tolist():
            parents = torch.where(self.mask[:, j] > 0)[0]
            design = torch.cat([X[:, parents], torch.ones(len(X), 1)], dim=1)
            coef = torch.linalg.lstsq(design, X[:, j : j + 1]).solution.squeeze(1)
            self.weight[parents, j] = coef[:-1]
            self.bias[j] = coef[-1]
        return self

    def forward(self, x):
        out = x @ (self.weight * self.mask) + self.bias
        recon = torch.zeros_like(x)  # parentless nodes are not predicted
        recon[:, self.predictable_children] = out[:, self.predictable_children]
        return recon

    @torch.no_grad()
    def compute_input_jacobian(self, x, dim_start, dim_end):
        # d out_j / d x_i = W[i, j] for every sample: shape (N, chunk, d)
        J = (self.weight * self.mask)[:, dim_start:dim_end].T
        return J.unsqueeze(0).expand(x.shape[0], -1, -1)
```

Fit it and use either path:

```python
from causalig.attr import CausalIntegratedGradients

surrogate = LinearSEMSurrogate(adjacency_matrix).fit(X_train).eval()

cig = CausalIntegratedGradients(model)
direct, indirect = cig.attribute(
    inputs, surrogate, baseline=baseline,
    adjacency_matrix=adjacency_matrix, separate_effects=True,
    has_closed_form_jacobian=True,     # uses compute_input_jacobian
)
```

The weight multiplication by `self.mask` in `forward` is what makes the closed-form path valid: column
`j` can only ever see the parents of `j`.

(validate-surrogate)=

## Validating a new surrogate

Run these checks before trusting attributions from a new surrogate. `X` is a batch of real data.

**1. Shapes and roots.** `surrogate(X)` returns `(N, d)`, and the root columns are irrelevant.

**2. Parents only.** No output may depend on a non-parent. This matters most on the closed-form path:

```python
x = X[:16].clone().requires_grad_(True)
out = surrogate(x)
for j in range(d):
    (g,) = torch.autograd.grad(out[:, j].sum(), x, retain_graph=True, allow_unused=True)
    if g is not None:
        assert torch.all(g[:, ~adjacency_matrix[:, j].bool()] == 0), f"feature {j} reads a non-parent"
```

**3. Jacobian matches autograd** (if you implement `compute_input_jacobian`), for the full range and
for a chunk:

```python
J_auto = torch.func.vmap(torch.func.jacrev(lambda v: surrogate(v.unsqueeze(0)).squeeze(0)))(X[:4])
assert torch.allclose(surrogate.compute_input_jacobian(X[:4], 0, d), J_auto, atol=1e-6)
assert torch.allclose(surrogate.compute_input_jacobian(X[:4], 1, 3), J_auto[:, 1:3], atol=1e-6)
```

**4. Both paths agree.** This is the end-to-end version of check 3:

```python
kw = dict(baseline=baseline, adjacency_matrix=adjacency_matrix, separate_effects=True)
d1, i1 = cig.attribute(inputs, surrogate, has_closed_form_jacobian=True, **kw)
d2, i2 = cig.attribute(inputs, surrogate, has_closed_form_jacobian=False, **kw)
assert torch.allclose(d1, d2, atol=1e-5) and torch.allclose(i1, i2, atol=1e-5)
```

**5. Completeness and fit.** Check that `return_convergence_delta=True` gives a delta that is small
relative to `f(input) - f(baseline)`, and that the reconstruction error is reasonable (see
{ref}`surrogate-quality`).

## Contributing a surrogate to `causalig`

To ship a new surrogate with the library:

1. **Implement it** in `causalig/models/surrogate.py`, or in a new module under `causalig/models/`. Use
   only `torch` and `numpy`, since those are the core dependencies. Give it a Google-style docstring with
   an `Args:` section, which these docs render automatically.
2. **Export it** from `causalig/models/__init__.py` by adding it to the imports and to `__all__`.
3. **Test it** in `tests/`. `tests/test_masked_global_linear.py` is a good template. At a minimum, cover
   checks 2–4 above and a completeness check on a small graph. Include a graph with parentless nodes,
   since roots exercise a separate code path.
4. **Document it** by adding an `autoclass` entry to `docs/api/models.md` and, if it needs any setup, a
   section to {doc}`surrogates`.
5. **Run** `pytest`, `ruff check .`, `black --check .` and the strict docs build (see {doc}`../development`).
