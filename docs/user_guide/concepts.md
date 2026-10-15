# How CIG and CEG work

This page explains what CIG and CEG compute, using the notation of the paper.

## The problem with independent-feature attribution

Let $F: \mathbb{R}^n \to \mathbb{R}$ be a differentiable model, $\boldsymbol{x}$ the input to explain and
$b$ a baseline. [Integrated gradients](https://arxiv.org/abs/1703.01365) (IG) integrates the gradient of $F$
along the straight-line path $\gamma(\alpha) = b + \alpha(\boldsymbol{x} - b)$, $\alpha \in [0, 1]$:

$$
R_i^{\mathrm{IG}} = \int_{0}^{1} \frac{\partial F(\gamma(\alpha))}{\partial \gamma_i(\alpha)} \frac{d\gamma_i(\alpha)}{d\alpha} \, d\alpha = (x_i - b_i) \int_{0}^{1} \frac{\partial F(b + \alpha(\boldsymbol{x} - b))}{\partial x_i} \, d\alpha.
$$

$R_i^{\mathrm{IG}}$ is non-zero only if the model *reads* feature $x_i$. If $x_k$ causes $x_i$ and the model
only uses $x_i$, then $x_k$ gets nothing, even though moving $x_k$ is what moved $x_i$. In most scientific
settings $x_k$ is the answer you actually want.

## Ingredients and notation

- **Causal graph.** A DAG over the $n$ input features, given as an adjacency matrix
  $M \in \mathbb{R}^{n \times n}$, where $M_{ij} = 1$ indicates that feature $i$ is a parent of feature $j$ In code, $M$ is the `adjacency_matrix` argument.
  Features are ordered topologically (the `ordering` argument): if $x_j$ depends on $x_i$, then $j > i$.
  We write $\mathrm{parents}(j)$ and $\mathrm{children}(i)$ for the parents and children of a feature.
  Features with no parents are *roots*.
- **Causal mechanism and surrogate.** Each feature is modeled as an additive structural causal model,

  $$
  x_i = \gamma_i(x_1, \dots, x_{i-1}, \alpha) = c_i(x_1, \dots, x_{i-1}) + h_i(\alpha), \quad i = 1, \dots, n,
  $$

  where $c_i$ is the causal mechanism mapping the parents of $x_i$ to its expected value. The surrogate
  $s: \mathbb{R}^n \to \mathbb{R}^n$ approximates it: its component $s_i$ is trained to predict $x_i$ from
  $x_{\mathrm{parents}(i)}$ only. For roots, $s_i \equiv 0$.
- **Residual.** $h_i^{(p)} \approx x_i^{(p)} - s_i(x_1^{(p)}, \dots, x_{i-1}^{(p)})$ is the part of $x_i$
  that its parents do *not* explain: its own innovation, or the exogenous noise in structural causal model
  terms. For a root, $h_i = x_i$.

## Direct and indirect effects

The path is still the straight line $\gamma(\alpha) = b + \alpha(\boldsymbol{x} - b)$, but each
$\gamma_i$ now also depends on the path variables of its parents. By the chain rule, the full derivative
of $\gamma_j$ along the path splits into a local part and a part inherited from its parents:

$$
\frac{d\gamma_j}{d\alpha} = \frac{\partial \gamma_j}{\partial \alpha} + \sum_{i < j} \frac{\partial \gamma_j}{\partial \gamma_i} \frac{d\gamma_i}{d\alpha}.
$$

Substituting this into the path integral of $F$ and regrouping by feature gives the attribution of
feature $i$, returned as two terms by `separate_effects=True`:

$$
R_i = \int_{0}^{1} \frac{\partial F}{\partial \gamma_i} \frac{\partial \gamma_i}{\partial \alpha} \, d\alpha
+ \sum_{j > i} \int_{0}^{1} \frac{\partial F}{\partial \gamma_j} \frac{\partial \gamma_j}{\partial \gamma_i} \frac{d\gamma_i}{d\alpha} \, d\alpha = R_{i, \mathrm{direct}} + R_{i, \mathrm{indirect}}.
$$

Missing edges simply correspond to zero terms ($\frac{\partial \gamma_j}{\partial \gamma_i} = 0$ if $x_j$
does not depend on $x_i$), so the sum over $j > i$ effectively runs over $\mathrm{children}(i)$. The two
extra gradients come from the surrogate:

$$
\frac{\partial \gamma_j}{\partial \gamma_i} = \frac{\partial s_j}{\partial x_i}, \qquad
\frac{\partial \gamma_i}{\partial \alpha} = \frac{d h_i}{d\alpha}.
$$

- The **direct effect** $R_{i, \mathrm{direct}}$ is the model's sensitivity to $x_i$, weighted by how much
  of $x_i$'s movement is its *own* innovation rather than inherited from its parents.
- The **indirect effect** $R_{i, \mathrm{indirect}}$ is the model's sensitivity to $x_i$'s *children*,
  routed back to $x_i$ through the surrogate's Jacobian.

**Completeness.** Summing both terms over all features recovers the full path integral, so

$$
\sum_{i=1}^{n} \left( R_{i, \mathrm{direct}} + R_{i, \mathrm{indirect}} \right) = \sum_{i=1}^{n} R_i = F(\boldsymbol{x}) - F(b).
$$

This holds for *any* differentiable surrogate. The surrogate decides how credit is split, never how much
there is in total.

Two limiting cases help build intuition:

- A surrogate that explains nothing ($s \equiv 0$) gives $h_i = \gamma_i$ and
  $\frac{\partial \gamma_j}{\partial \gamma_i} = 0$, so every indirect effect is zero and
  $R_{i, \mathrm{direct}} = R_i^{\mathrm{IG}}$: CIG reduces exactly to integrated gradients.
- A feature with no parents and no children (feature 3 in the {doc}`../quickstart`) gets its IG
  attribution, up to discretization.

:::{admonition} Mapping to the code
:class: note
In `causalig/attr/_core/causal_expected_gradients.py`:

- `grads_f` is $\frac{\partial F}{\partial \gamma_i}$;
- `grads_f_ae` is $\sum_{j > i} \frac{\partial F}{\partial \gamma_j} \frac{\partial \gamma_j}{\partial \gamma_i}$;
- `grads_h` is $\frac{\partial \gamma_i}{\partial \alpha}$;
- `grads_x_alpha` is $\frac{d\gamma_i}{d\alpha}$.

The surrogate's Jacobian $\frac{\partial s_j}{\partial x_i}$ comes from
{func}`~causalig.attr._utils.surrogate.compute_surrogate_gradients`.
:::

## Second-order indirect effects

The indirect term above uses the full derivative $\frac{d\gamma_i}{d\alpha}$, including the part of
$x_i$'s movement inherited from its own parents. It therefore attributes only indirect effects that are one
hop long (first-order). Along a path $x_k \to x_i \to x_j \to F$, the credit stops at $x_i$, one hop
upstream of the feature the model reads. It does not reach $x_k$.

`include_second_order_indirect=True` also tracks dependency paths of length two (second-order). The
one-hop term now uses the local $\frac{\partial \gamma_i}{\partial \alpha}$, and a two-hop term takes over
the full derivative:

$$
R_{i, \mathrm{indirect}}^{(2)} = \sum_{j > i} \int_{0}^{1} \frac{\partial F}{\partial \gamma_j} \frac{\partial \gamma_j}{\partial \gamma_i} \frac{\partial \gamma_i}{\partial \alpha} \, d\alpha + \sum_{\substack{j > i \\ i < k < j}} \int_{0}^{1} \frac{\partial F}{\partial \gamma_j} \frac{\partial \gamma_j}{\partial \gamma_k} \frac{\partial \gamma_k}{\partial \gamma_i} \frac{d\gamma_i}{d\alpha} \, d\alpha.
$$

One-hop effects are now weighted by $x_i$'s own innovation, and two-hop effects reach $x_i$ through its
grandchildren. Completeness still holds exactly, since every hop beyond the second is folded into the
deepest tracked layer's full derivative $\frac{d\gamma_i}{d\alpha}$ rather than discarded. Turning this on
costs roughly $2\times$ the first-order-only cost. In the paper it gave the best results on the Dyngen
benchmark.

### Worked example: a three-node chain

Take a chain $x_1 \to x_2 \to x_3$ with mechanisms $x_2 = 2x_1$ and $x_3 = 3x_2$, and a model
$F(\boldsymbol{x}) = x_3$ that only reads $x_3$. Explain $\boldsymbol{x} = (1,\ 2.5,\ 7.7)$ against the
baseline $b = \boldsymbol{0}$. Then $x_2$ carries an innovation of $h_2 = 2.5 - 2 \cdot 1 = 0.5$, and $x_3$
an innovation of $h_3 = 7.7 - 3 \cdot 2.5 = 0.2$. In code, features are zero-indexed, so $x_1$ is column 0.

```python
import torch
from causalig.attr import CausalIntegratedGradients

adjacency_matrix = torch.tensor([[0, 1, 0], [0, 0, 1], [0, 0, 0]], dtype=torch.bool)

def surrogate(x):  # x_2 = 2 x_1,  x_3 = 3 x_2
    return torch.stack([torch.zeros_like(x[:, 0]), 2 * x[:, 0], 3 * x[:, 1]], dim=1)

def model(x):  # F only reads x_3
    return x[:, 2:3]

x = torch.tensor([1.0, 2.5, 7.7])
baseline = torch.zeros(3)
cig = CausalIntegratedGradients(model)
for second_order in (False, True):
    direct, indirect = cig.attribute(
        x, surrogate, baseline=baseline,
        separate_effects=True, adjacency_matrix=adjacency_matrix,
        include_second_order_indirect=second_order,
    )
```

| | $x_1$ | $x_2$ | $x_3$ |
|---|---:|---:|---:|
| $R_i^{\mathrm{IG}}$ | 0 | 0 | 7.7 |
| $R_{i, \mathrm{direct}}$ (first-order) | 0 | 0 | 0.2 |
| $R_{i, \mathrm{indirect}}$ (first-order) | 0 | **7.5** | 0 |
| $R_{i, \mathrm{direct}}$ (second-order) | 0 | 0 | 0.2 |
| $R_{i, \mathrm{indirect}}^{(2)}$ (second-order) | **6.0** | 1.5 | 0 |

The attributions of each method sum to $F(\boldsymbol{x}) - F(b) = 7.7$.

- **First-order** moves the credit one hop up, to $x_2$:
  $\frac{\partial F}{\partial \gamma_3} \frac{\partial \gamma_3}{\partial \gamma_2} \frac{d\gamma_2}{d\alpha} = 1 \cdot 3 \cdot 2.5 = 7.5$.
- **Second-order** sends the part of that credit which $x_1$ caused on to $x_1$:
  $\frac{\partial F}{\partial \gamma_3} \frac{\partial \gamma_3}{\partial \gamma_2} \frac{\partial \gamma_2}{\partial \gamma_1} \frac{d\gamma_1}{d\alpha} = 1 \cdot 3 \cdot 2 \cdot 1 = 6$.
  $x_2$ keeps only what its own innovation contributed,
  $\frac{\partial F}{\partial \gamma_3} \frac{\partial \gamma_3}{\partial \gamma_2} \frac{\partial \gamma_2}{\partial \alpha} = 1 \cdot 3 \cdot 0.5 = 1.5$.

## From CIG to CEG

CIG uses a single baseline $b$, so the attributions depend on that choice. Zeros, the training mean and a
control condition can all give different answers.
[Expected gradients](https://arxiv.org/abs/1906.10670) instead averages over baselines drawn from the data
distribution $\mathcal{D}$. CEG does the same for the causal path:

$$
R_i^{\mathrm{CEG}} = \mathbb{E}_{b \sim \mathcal{D}} \left[ R_i(\boldsymbol{x}; b) \right], \qquad
\sum_{i=1}^{n} R_i^{\mathrm{CEG}} = F(\boldsymbol{x}) - \mathbb{E}_{b \sim \mathcal{D}} \left[ F(b) \right].
$$

The library implements two estimators of that expectation. {doc}`methods` compares them:

- {class}`~causalig.attr.CausalExpectedGradientsMultiStep` integrates each baseline's path with a $K$-step
  Riemann sum (`n_steps`) and averages over baselines. {class}`~causalig.attr.CausalIntegratedGradients`
  is exactly this class with a single baseline.
- {class}`~causalig.attr.CausalExpectedGradients` follows the original expected gradients estimator. It
  samples the baseline $b \sim \mathcal{D}$ and the path parameter $\alpha \sim \mathcal{U}(0,1)$ jointly,
  drawing one path point per baseline.

## How the surrogate is queried

CIG/CEG need two things from the surrogate at every point on the path: its prediction $s_j$ for each
non-root feature $j$, and the Jacobian $\frac{\partial s_j}{\partial x_i}$. They get them in one of two ways:

- **Autograd path** (default, works with any differentiable callable). For each non-root child $j$, the
  input is copied and every feature that is *not* a parent of $j$ is set to `0`. The surrogate is called
  on that masked copy, and column $j$ of its output is read and differentiated. Masking guarantees that
  $s_j$ can only use $x_{\mathrm{parents}(j)}$, even if the surrogate is a generic network over all $n$
  inputs.
- **Closed-form path** (`has_closed_form_jacobian=True`). The surrogate is called once on the unmasked
  input and asked for its Jacobian directly through `compute_input_jacobian`. This is much cheaper for
  large $n$, but the surrogate must enforce the parent structure itself.

{doc}`custom_surrogate` specifies both contracts precisely.

When `adjacency_matrix` is `None`, the parents of $j$ are taken to be *every feature before $j$ in
`ordering`*, i.e. every $i < j$. That is the fully connected DAG consistent with the ordering, and usually
not what you want. Pass `adjacency_matrix` whenever you have it.

(completeness-delta)=
## Discretization and the completeness delta

All methods compute the path integral numerically, so the attributions sum to $F(\boldsymbol{x}) - F(b)$
only approximately. For CIG and `CausalExpectedGradientsMultiStep` the error shrinks as `n_steps` grows.
It is somewhat larger than plain IG's, because the path derivative of the residual $h_i$ is estimated
numerically rather than known in closed form. `CausalExpectedGradients` instead uses one random path
point per baseline, so its error shrinks with more baselines rather than more steps.

`return_convergence_delta=True` returns the per-sample error over the $B$ baselines
$b^{(1)}, \dots, b^{(B)}$,

$$
\delta = \sum_{i=1}^{n} R_i - \left( F(\boldsymbol{x}) - \frac{1}{B} \sum_{l=1}^{B} F(b^{(l)}) \right),
$$

which should be small relative to $|F(\boldsymbol{x}) - F(b)|$. For CIG and the multi-step estimator it
shrinks roughly in proportion to `1 / n_steps`. Surrogates with sharper nonlinearities, such as the default basis expansion of
`MaskedGlobalLinear`, need more steps than linear ones for the same accuracy. The paper's
`run_synthetic_nsteps_ablation.py` script measures this convergence (see {doc}`../experiments`).

## Assumptions and caveats

- **The graph is prior knowledge, and it is trusted.** A missing edge stops credit from flowing up that
  edge. A spurious edge lets a learned surrogate route credit along it. The paper's edge-robustness
  experiment measures both effects.
- **The surrogate's quality sets the split.** Anything the surrogate cannot predict stays in $h_i$ and is
  attributed as a direct effect. An underfit surrogate therefore looks like plain IG. Check its
  reconstruction error before trusting the indirect effects (see {ref}`surrogate-quality`).
- **The built-in surrogate is additive over parents.** `MaskedGlobalLinear` models
  $s_j = \sum_{i: M_{ij}=1} \langle w_{ij}, \phi(x_i) \rangle + \mu_j$, where $\phi(x_i) \in \mathbb{R}^{Q}$
  is a per-feature basis expansion, so it cannot represent interactions such as $x_j = x_a \cdot x_b$. Use
  a {doc}`custom surrogate <custom_surrogate>` if your mechanisms are strongly non-additive.
