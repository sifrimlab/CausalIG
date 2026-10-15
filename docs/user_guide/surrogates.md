# Surrogate models

The surrogate is the third ingredient of CIG/CEG. It is a model of the causal *mechanisms*: given the
graph's structure, it predicts each feature from its parents. CIG/CEG use its predictions to split every
feature into an inherited part and its own innovation, and its Jacobian to route credit along edges
(see {doc}`concepts`).

This page covers what ships with the library:

| Component | Role |
|---|---|
| {class}`~causalig.models.MaskedGlobalLinear` | The standard surrogate: sparse, per-edge and optionally nonlinear, with an exact closed-form Jacobian. |
| {func}`~causalig.models.train_surrogate_model` | Trains a surrogate by reconstructing every non-root feature. |
| {class}`~causalig.models.DifferentiableSCM` + {func}`~causalig.models.create_autoencoder_from_scm` | Use a known structural causal model as an oracle surrogate (simulations). |

To plug in a model of your own, see {doc}`custom_surrogate`.

## `MaskedGlobalLinear`

For every non-root feature $j$, `MaskedGlobalLinear` predicts

$$
s_j(x) = b_j + \sum_{i \in \mathrm{pa}(j)} \big\langle w_{ij},\ \varphi(x_i) \big\rangle ,
$$

where $\varphi$ is a per-feature basis expansion, or the identity when `use_basis_expansion=False`. Only
the edges present in `adjacency_matrix` carry weights, so the parameter count grows with the number of
edges $E$, not with $d^2$. Because the model is linear in $\varphi$, its input Jacobian has a closed
form. Pass `has_closed_form_jacobian=True` to CIG/CEG to use it.

```python
from causalig.models import MaskedGlobalLinear

surrogate = MaskedGlobalLinear(
    adjacency_matrix=adjacency_matrix,          # (d, d), adjacency_matrix[i, j] = 1 means i -> j
    X_train=X_train,             # used to standardize inputs to the basis (recommended)
    use_basis_expansion=True,    # nonlinear per-edge functions; False = plain linear SEM
)
```

| Argument | Default | Meaning |
|---|---|---|
| `adjacency_matrix` | *required* | `(d, d)` array or tensor, with parents as rows and children as columns. At least one edge is required. |
| `predict_only_children` | `True` | Root features (no parents) are output as `0`. With `False` they are copied from the input instead. CIG/CEG ignore root outputs either way. |
| `use_basis_expansion` | `True` | Expand each parent through the basis $\varphi$ below. `False` gives a linear structural equation model, $\varphi(x_i) = x_i$. |
| `X_train` | `None` | Training data, used only to set the basis' per-feature mean and standard deviation. Without it, the basis acts on raw values with mean 0 and scale 1, and the default knot positions assume standardized data. |
| `n_knots` | `5` | Number of truncated-power terms $\mathrm{ReLU}(z - c_k)^p$ per feature. |
| `knot_power` | `2` | The power $p$ of those terms. |
| `polynomial_degree` | `3` | Include $z, z^2, \ldots, z^{D}$. |
| `fourier_features` | `0` | Number of $\sin/\cos$ frequency pairs. |
| `learnable_knots` | `True` | Train the knot positions $c_k$, initialized evenly on $[-2, 2]$. |

The basis is applied to the standardized value $z = (x_i - \mu_i)/\sigma_i$, clipped to $[-8, 8]$. It
consists of:

- polynomial terms;
- truncated-power splines with knots $c_k$;
- optional Fourier terms.

That gives $K$ = `polynomial_degree + n_knots + 2 * fourier_features` functions per edge. Its exact
derivative (zero in the clipped region) is {meth}`~causalig.models.surrogate.BasisExpansion.derivative`,
which {meth}`~causalig.models.MaskedGlobalLinear.compute_input_jacobian` uses.

:::{note}
`MaskedGlobalLinear` is **additive over parents**. It can model a nonlinear effect of each parent, but
not interactions such as $x_j = x_a x_b$. The repository's
`run_synthetic_interaction_indirect_effect_multiseed.py` experiment quantifies what this costs (see
{doc}`../experiments`). If your mechanisms are strongly interactive, use a
{doc}`custom surrogate <custom_surrogate>`.
:::

## Training

```python
from causalig.models import train_surrogate_model

surrogate = train_surrogate_model(
    surrogate, X_train,
    epochs=100, lr=1e-2, batch_size=256,
    device="cpu", num_workers=0,
)
```

### `train_surrogate_model`

The trainer minimizes the mean squared error between `surrogate(x)` and `x` on the non-root features
(`surrogate.predictable_children`). That is valid without any input masking because `MaskedGlobalLinear`
only reads parents by construction.

| Argument | Default | Meaning |
|---|---|---|
| `surrogate` | *required* | The model to train. It must expose `predictable_children`, the indices of features with at least one parent. |
| `X` | *required* | Training data, `(n_samples, d)`. A tensor or array, cast to `float32`. |
| `epochs` | `100` | Passes over the training split. |
| `batch_size` | `256` | Mini-batch size. |
| `lr` | `1e-3` | Peak learning rate (AdamW, cosine-annealed per step down to `lr / 100`). |
| `weight_decay` | `1e-5` | AdamW weight decay, not applied to parameters named `bias` or `knots`. |
| `train_frac`, `val_frac` | `0.8`, `0.1` | Random train/validation/test split. The rest is the test split. |
| `seed` | `42` | Seed for the split. |
| `device` | `"cuda"` | Training device. **Pass `"cpu"` on machines without a GPU.** |
| `grad_clip_norm` | `1.0` | Gradient-norm clipping. |
| `num_workers` | `4` | `DataLoader` workers. Use `0` in notebooks and on macOS/Windows. |
| `logger` | `None` | A `logging.Logger` for per-epoch losses. Defaults to the module logger, which is silent unless you configure logging. |
| `save_dir` | `None` | If set, saves the best `surrogate.weight` to `best_masked_weight.npy`. This needs a `.weight` attribute. |

The trainer keeps the weights with the best validation loss, logs the test loss, and returns the
surrogate in `eval()` mode, on `device`. Batches with non-finite losses or gradients are skipped.

(surrogate-quality)=
## Checking surrogate quality

The surrogate decides how credit is split between direct and indirect effects. Anything it fails to
predict stays in the residual $h = x - s(x)$ and is attributed as a *direct* effect. An underfit
surrogate therefore quietly pushes CIG back towards plain Integrated Gradients. Always look at its
error before reading the attributions:

```python
with torch.no_grad():
    residual = surrogate(X_val) - X_val
    children = surrogate.predictable_children
    mse = (residual[:, children] ** 2).mean(dim=0)   # one value per non-root feature
    r2 = 1 - mse / X_val[:, children].var(dim=0)
```

On a toy problem, the per-feature MSE should approach the noise variance of each mechanism. On real data,
compare $R^2$ across features. A feature with $R^2$ near zero is one the surrogate cannot predict from
its parents: its prediction is roughly constant, so its derivative with respect to the parents is
roughly zero. CIG then attributes that feature like plain IG would, as a direct effect only, and passes
none of its credit up to its parents. If you expected such a feature to be driven by its parents, the
graph or the surrogate is missing that relationship.

On small datasets, the defaults (`lr=1e-3`, `batch_size=256`) give few optimizer steps. Raise `lr`
or `epochs` if the validation loss is still dropping when training ends. The {doc}`../quickstart` uses
`lr=1e-2, epochs=100` on 5,000 samples.

## Using a known SCM as the surrogate

In simulations where the true structural causal model is known, you can hand it to CIG/CEG directly
as an *oracle* surrogate. {class}`~causalig.models.DifferentiableSCM` holds:

- the adjacency matrix;
- one assignment function per node, which maps a `(batch, n_parents)` tensor to `(batch,)`, with `None`
  for roots;
- a topological order.

{func}`~causalig.models.create_autoencoder_from_scm` wraps it as a surrogate:

```python
from causalig.models import DifferentiableSCM, create_autoencoder_from_scm

assignments = [None, lambda p: 2.0 * p[:, 0], lambda p: torch.tanh(2.0 * p[:, 0]), None]
scm = DifferentiableSCM(adjacency_matrix.numpy(), assignments, topological_order=[0, 3, 1, 2])

surrogate = create_autoencoder_from_scm(scm, structural=True)
# no compute_input_jacobian: keep has_closed_form_jacobian=False (the default)
```

:::{warning}
Always pass **`structural=True`** when using an SCM as a CIG/CEG surrogate. In structural mode each
node's mechanism is applied directly to the parent values it is given, which is what a surrogate must
do.

With the default `structural=False`, the SCM instead re-simulates every non-root parent from its own,
masked-out ancestors. Indirect effects then vanish on any graph deeper than one edge. All of the paper's
scripts use `structural=True`.
:::

The same `DifferentiableSCM` also provides `forward_do`, which {class}`~causalig.attr.CausalSHAP`
needs for its interventional sampling.
