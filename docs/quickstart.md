# Quickstart

This page runs CIG and CEG end to end on a toy problem small enough to train on a CPU in a few seconds.
The setup mirrors the motivating biology example. A *regulator* (feature 0) drives two *target* features
(1 and 2), and a fourth feature is unrelated. The model only ever reads the targets.

```text
        0  (regulator)            3  (unrelated)
       / \
      v   v
     1     2  (targets)           model reads 1, 2 and 3
```

Plain Integrated Gradients credits the targets, because those are the inputs the model uses. CIG moves
that credit to the regulator that caused them.

## 1. Data and a causal graph

The graph is a `(d, d)` boolean tensor where `adjacency_matrix[i, j] = True` means *feature `i` causes feature `j`*.

```python
import torch
from torch import nn

torch.manual_seed(0)

d = 4
adjacency_matrix = torch.zeros(d, d, dtype=torch.bool)
adjacency_matrix[0, 1] = True  # 0 -> 1
adjacency_matrix[0, 2] = True  # 0 -> 2
# feature 3 has no parents and no children

n = 5000
X = torch.zeros(n, d)
X[:, 0] = torch.randn(n)
X[:, 1] = 2.0 * X[:, 0] + 0.3 * torch.randn(n)
X[:, 2] = torch.tanh(2.0 * X[:, 0]) + 0.3 * torch.randn(n)
X[:, 3] = torch.randn(n)
```

## 2. The model to explain

Any differentiable callable works. Here it is a small MLP fit to a target that depends only on the
downstream features 1, 2 and 3:

```python
y = X[:, 1] + X[:, 2] + X[:, 3]

model = nn.Sequential(nn.Linear(d, 32), nn.Tanh(), nn.Linear(32, 1))
optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
for _ in range(1000):
    optimizer.zero_grad()
    loss = ((model(X).squeeze(-1) - y) ** 2).mean()
    loss.backward()
    optimizer.step()
model.eval()
```

## 3. Train a surrogate on the graph

The surrogate learns to predict every feature from its causal parents. {class}`~causalig.models.MaskedGlobalLinear`
only parameterizes the edges in `adjacency_matrix`, and its default basis expansion lets each edge be nonlinear
(the `0 -> 2` edge here is a `tanh`).

```python
from causalig.models import MaskedGlobalLinear, train_surrogate_model

surrogate = MaskedGlobalLinear(adjacency_matrix=adjacency_matrix, X_train=X)
surrogate = train_surrogate_model(
    surrogate, X, epochs=100, lr=1e-2, device="cpu", num_workers=0
)
```

:::{note}
`train_surrogate_model` defaults to `device="cuda"` and `num_workers=4`. Pass `device="cpu"` on
machines without a GPU.

Check the fit before attributing. Here the reconstruction MSE of features 1 and 2 should approach the
noise variance, `0.3**2 = 0.09`. A badly underfit surrogate quietly pulls CIG back towards plain
Integrated Gradients; see {ref}`surrogate-quality`.
:::

## 4. Attribute with CIG

```python
from causalig.attr import CausalIntegratedGradients

inputs = X[:200]
baseline = torch.zeros(d)

cig = CausalIntegratedGradients(model)
direct, indirect, delta = cig.attribute(
    input=inputs,
    surrogate=surrogate,
    baseline=baseline,
    n_steps=50,
    separate_effects=True,           # return (direct, indirect) instead of their sum
    adjacency_matrix=adjacency_matrix,
    has_closed_form_jacobian=True,   # MaskedGlobalLinear has an exact analytic Jacobian
    return_convergence_delta=True,   # also return the completeness error per sample
)
direct.shape, indirect.shape, delta.shape   # (200, 4), (200, 4), (200,)
```

Mean absolute attribution per feature over the 200 explained samples (your numbers will differ slightly):

| Feature | Role | Integrated Gradients | CIG direct | CIG indirect | CIG total |
|---|---|---:|---:|---:|---:|
| 0 | regulator, never read by the model | 0.03 | 0.03 | **2.20** | **2.18** |
| 1 | target of 0 | **1.50** | 0.23 | 0 | 0.23 |
| 2 | target of 0 | **0.74** | 0.21 | 0 | 0.21 |
| 3 | unrelated | 0.85 | 0.85 | 0 | 0.85 |

The Integrated Gradients column was computed with Captum on the same model and baseline, for comparison.

How to read the table:

- **Feature 0** gets almost all of the credit Integrated Gradients gave to 1 and 2, as an *indirect*
  effect, because it moved them.
- **Features 1 and 2** keep a small *direct* effect, which is the part of their value their regulator
  does not explain (the `0.3 * noise` term).
- **Feature 3** has no causal relatives, so CIG and Integrated Gradients agree on it exactly.

`delta` measures how far `direct + indirect` is from `f(input) - f(baseline)`. Here its median is about
0.25% of the output change. It shrinks as `n_steps` grows; see {ref}`completeness-delta`.

## 5. Attribute with CEG

{class}`~causalig.attr.CausalExpectedGradientsMultiStep` replaces the single baseline with a set of
reference points drawn from the data and averages over them. It takes the same arguments, with
`baselines` instead of `baseline`:

```python
from causalig.attr import CausalExpectedGradientsMultiStep

baselines = X[torch.randperm(n)[:20]]   # 20 reference rows from the data

ceg = CausalExpectedGradientsMultiStep(model)
direct, indirect = ceg.attribute(
    input=inputs,
    surrogate=surrogate,
    baselines=baselines,
    n_steps=20,                      # steps *per baseline*: 20 x 20 = 400 gradient evaluations
    separate_effects=True,
    adjacency_matrix=adjacency_matrix,
    has_closed_form_jacobian=True,
)
```

On this example CEG gives the same picture as CIG: the regulator carries the credit and the targets
keep only their own noise.

## Next steps

- {doc}`user_guide/concepts`: what "direct" and "indirect" mean precisely, and when to turn on
  `include_second_order_indirect`.
- {doc}`user_guide/arguments`: every argument, the return shapes, and how to keep memory under
  control for large `d`.
- {doc}`user_guide/custom_surrogate`: plug in your own surrogate model.
