# Attribution methods

All methods live in {mod}`causalig.attr` and follow [Captum](https://captum.ai/)'s pattern:

1. construct the explainer around the model, `explainer = Method(model)`;
2. call `explainer.attribute(input, ...)`.

| Class | Reference point(s) | Gradient evaluations per input | Use it when |
|---|---|---|---|
| {class}`~causalig.attr.CausalIntegratedGradients` (CIG) | one `baseline` | `n_steps` | You have a meaningful reference state (all-zero, control condition, training mean). |
| {class}`~causalig.attr.CausalExpectedGradientsMultiStep` (CEG) | `baselines`, `(B, d)` | `B × n_steps` | You want attributions that do not hinge on one arbitrary baseline. **Recommended CEG variant.** |
| {class}`~causalig.attr.CausalExpectedGradients` | `baselines`, `(B, d)` | `B` | You follow the original Expected Gradients estimator, with one random path point per baseline. |
| {class}`~causalig.attr.CausalSHAP` | SCM + background | permutation sampling | You need the causal Shapley baseline from the paper (small `d`). |
| {class}`~causalig.attr.ConditionalSHAP` | Gaussian fit to background | permutation sampling | You need the conditional SHAP baseline from the paper (small `d`). |

Each "gradient evaluation" is one model backward pass plus one surrogate Jacobian evaluation. The
surrogate side is usually the more expensive of the two; see {ref}`performance`.

The three gradient methods share almost all of their arguments; {doc}`arguments` documents them in one
place.

## Causal Integrated Gradients

CIG integrates along the straight line from one `baseline` to each input. The baseline is a `(d,)`
tensor shared by every row of `input`.

```python
from causalig.attr import CausalIntegratedGradients

cig = CausalIntegratedGradients(model)
direct, indirect = cig.attribute(
    input=inputs,                 # (batch, d) or (d,)
    surrogate=surrogate,
    baseline=baseline,            # (d,)
    n_steps=50,
    separate_effects=True,
    adjacency_matrix=adjacency_matrix,
    has_closed_form_jacobian=True,
)
```

**Choosing a baseline.** The attributions explain $f(x) - f(x')$, so pick an $x'$ for which that
difference is the question you care about. Common choices:

- zeros, for standardized or count data where zero means "absent";
- the training mean;
- a reference sample, such as an unperturbed or control condition.

If no single reference is natural, use CEG.

Internally, `CausalIntegratedGradients` simply calls `CausalExpectedGradientsMultiStep` with
`baselines=baseline.unsqueeze(0)`, so the two always agree.

## Causal Expected Gradients (multi-step)

`CausalExpectedGradientsMultiStep` integrates every baseline's path on its own `n_steps` grid, then
averages the attributions over baselines. Completeness then holds against the *average* baseline
output, $f(x) - \frac{1}{B}\sum_b f(x'_b)$.

```python
from causalig.attr import CausalExpectedGradientsMultiStep

baselines = X_train[torch.randperm(len(X_train))[:20]]   # (B, d) reference rows

ceg = CausalExpectedGradientsMultiStep(model)
direct, indirect = ceg.attribute(
    input=inputs,
    surrogate=surrogate,
    baselines=baselines,
    n_steps=20,                   # per baseline
    separate_effects=True,
    adjacency_matrix=adjacency_matrix,
    has_closed_form_jacobian=True,
)
```

**Budget.** The cost is `B × n_steps` gradient evaluations per input. To compare against CIG at equal
compute, divide CIG's step count by the number of baselines. The paper's experiments do exactly that
(`n_steps = total_steps // n_baselines`). For a fixed budget, more baselines with fewer steps each
reduces baseline-choice variance, and fewer baselines with more steps reduces discretization error.

## Causal Expected Gradients (single step)

`CausalExpectedGradients` is the direct transcription of Expected Gradients. For each baseline $x'_b$ it
draws one $\alpha_b \sim \mathcal{U}(0, 1)$, evaluates the causal gradient at
$x'_b + \alpha_b (x - x'_b)$, and approximates the path derivative of $h$ by the secant
$(h(\gamma(\alpha_b)) - h(x'_b)) / \alpha_b$. It has no `n_steps`. The number of baselines *is* the
sample size, so it typically needs hundreds of baselines.

```python
from causalig.attr import CausalExpectedGradients

ceg1 = CausalExpectedGradients(model)
attributions = ceg1.attribute(
    input=inputs,
    surrogate=surrogate,
    baselines=X_train[:500],      # one path sample per row
    adjacency_matrix=adjacency_matrix,
    has_closed_form_jacobian=True,
    seed=0,                       # makes the alpha draws reproducible
)
```

For the same number of gradient evaluations, the multi-step variant spends them as several path points
per baseline instead of one. The paper found this stabilizes attributions substantially, which is why
`CausalExpectedGradientsMultiStep` is the recommended estimator.

## Shapley-value baselines

`CausalSHAP` and `ConditionalSHAP` are dependency-aware Shapley-value methods, reimplemented here to
produce the paper's comparisons. They wrap
[`shap.PermutationExplainer`](https://shap.readthedocs.io/) and need the `shap` extra
(`pip install causalig[shap]`). Their cost grows quickly with `d`, which is the scaling problem CIG and
CEG avoid.

Both differ from the gradient methods in a few ways:

- the model must be a {class}`torch.nn.Module`, because they look up its device through
  `.parameters()`;
- `attribute()` takes a **2-D** `(batch, d)` tensor and returns a NumPy array;
- they return a single attribution per feature, with no direct/indirect split.

### CausalSHAP

Causal Shapley values ([Heskes et al., 2020](https://arxiv.org/abs/2011.01625)). Each coalition's
background samples are drawn by intervening on the SCM with $\mathrm{do}(X_S = x_S)$. That needs a
structural causal model with a `forward_do(X_input, noise, do_nodes, do_values)` method, such as
{class}`~causalig.models.DifferentiableSCM`.

```python
from causalig.attr import CausalSHAP
from causalig.models import DifferentiableSCM

scm = DifferentiableSCM(adjacency_matrix.numpy(), assignments, topological_order=[0, 3, 1, 2])  # see the API reference
explainer = CausalSHAP(
    model, scm,
    background=X_train,   # subsampled to n_samples rows
    n_samples=50,         # interventional samples per coalition
    npermutations=10,
    noise_sigma=0.01,     # std of the exogenous noise added by the SCM
    seed=0,
)
shap_values = explainer.attribute(inputs)   # numpy, (batch, d)
```

### ConditionalSHAP

Conditional SHAP ([Aas et al., 2021](https://arxiv.org/abs/1903.10464)). Masked features are imputed from
a multivariate Gaussian fitted to `background`, conditioned on the observed features.

```python
from causalig.attr import ConditionalSHAP

explainer = ConditionalSHAP(
    model,
    background=X_train.numpy(),  # used to fit the Gaussian
    n_samples=50,                # conditional samples per coalition
    npermutations=10,
    seed=0,
)
shap_values = explainer.attribute(inputs)   # numpy, (batch, d)
```

For reproducible results pass `seed`, and also call `torch.manual_seed` for `CausalSHAP`, whose SCM
noise comes from torch's global RNG.
