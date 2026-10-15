# Arguments reference

This page documents the arguments of `attribute()` for the three gradient-based methods. They share
almost all of them. For each class's exact signature, see the {doc}`API reference <../api/attr>`.

## At a glance

✓ = accepted, — = not accepted.

| Argument | CIG | CEG multi-step | CEG single-step | Default | Summary |
|---|:-:|:-:|:-:|---|---|
| [`input`](#arg-input) | ✓ | ✓ | ✓ | *required* | Samples to explain, `(batch, d)` or `(d,)`. |
| [`surrogate`](#arg-surrogate) | ✓ | ✓ | ✓ | *required* | Model predicting each feature from its parents. |
| [`ordering`](#arg-ordering) | ✓ | ✓ | ✓ | `None` | Topological order of the `d` features; derived from `adjacency_matrix` if omitted. |
| [`baseline`](#arg-baseline) | ✓ | — | — | *required* | One reference point, `(d,)`. |
| [`baselines`](#arg-baseline) | — | ✓ | ✓ | *required* | Reference points, `(B, d)` or `(d,)`. |
| [`n_steps`](#arg-n-steps) | ✓ | ✓ | — | `50` | Integration steps per path. |
| [`target`](#arg-target) | ✓ | ✓ | ✓ | `None` | Output index to explain. |
| [`separate_effects`](#arg-separate-effects) | ✓ | ✓ | ✓ | `False` | Return `(direct, indirect)` instead of their sum. |
| [`adjacency_matrix`](#arg-adjacency-matrix) | ✓ | ✓ | ✓ | `None` | Causal graph, `(d, d)` torch tensor. |
| [`has_closed_form_jacobian`](#arg-has-closed-form-jacobian) | ✓ | ✓ | ✓ | `False` | Use the surrogate's analytic Jacobian. |
| [`include_second_order_indirect`](#arg-include-second-order-indirect) | ✓ | ✓ | ✓ | `False` | Also propagate credit two hops upstream. |
| [`return_convergence_delta`](#arg-return-convergence-delta) | ✓ | ✓ | ✓ | `False` | Also return the completeness error. |
| [`dim_chunk_size`](#chunking-arguments) | ✓ | ✓ | ✓ | `None` | Surrogate outputs per Jacobian pass. |
| [`step_chunk_size`](#chunking-arguments) | ✓ | ✓ | — | `None` | Path steps per pass (must be > 1). |
| [`baselines_chunk_size`](#chunking-arguments) | — | ✓ | ✓ | `None` | Baselines per pass. |
| [`seed`](#arg-seed) | — | — | ✓ | `None` | Seed for the random path points. |

## Inputs

(arg-input)=
### `input`

The samples to explain: a float tensor of shape `(batch, d)`, or `(d,)` for a single sample. A 1-D input
gives 1-D outputs. `input` must live on the same device as the model, the surrogate and the baselines.

Computations accumulate in `float32`, so use `float32` inputs and models. Other dtypes are not
rejected, but the returned attributions are `float32`.

(arg-surrogate)=
### `surrogate`

A model of the causal mechanisms: a callable mapping `(N, d)` to `(N, d)` whose output column `j`
predicts feature `j` from its parents. Use a trained
{class}`~causalig.models.MaskedGlobalLinear`, or any object that satisfies the
{doc}`surrogate contract <custom_surrogate>`.

Put the surrogate in `eval()` mode. It is called on batches built from many path points at once, so
dropout or batch-norm in training mode would couple samples.

(arg-ordering)=
### `ordering`

A topological ordering of the feature indices (every parent before its children), as a length-`d`
list. You only need it when you do not pass `adjacency_matrix`: every feature that comes earlier in the
ordering is then treated as a potential parent. When `adjacency_matrix` is given, leave `ordering` out
and it is derived from the graph, which also raises an error if the graph contains a cycle.

(arg-baseline)=
### `baseline` / `baselines`

The reference point(s) the path starts from. The attributions explain `f(input) - f(baseline)`, or
`f(input) - mean_b f(baselines[b])` for the CEG estimators.

- **CIG** takes `baseline`, a single `(d,)` tensor shared by every input.
- **CEG** (both estimators) takes `baselines`, shape `(B, d)`, typically rows sampled from the training
  data. Every input is paired with every baseline and the results are averaged.

{doc}`methods` gives guidance on choosing baselines.

(arg-adjacency-matrix)=
### `adjacency_matrix`

The causal graph as a `(d, d)` **torch tensor** of booleans or 0/1 values, where `adjacency_matrix[i, j]` is true if
*feature `i` causes feature `j`*. It must be a `torch.Tensor`: a NumPy array raises an `AttributeError`.
It is moved to the input's device automatically.

It decides which inputs the surrogate may use for each feature, and therefore which edges credit can
flow along. Without it, the dense DAG implied by `ordering` is used instead (see
[`ordering`](#arg-ordering)). **Pass it whenever you have it**, and use the same graph the surrogate was
trained on.

(arg-target)=
### `target`

For models with several outputs, `target=k` explains output `k` only. With `target=None`:

- a model returning `(batch,)` or `(batch, 1)` is explained as is;
- a model returning `(batch, C)` with `C > 1` has *every* output explained, one backward pass per
  output, and the attributions gain a trailing `C` dimension (see [Return values](#return-values)).

## Behavior switches

(arg-n-steps)=
### `n_steps`

The number of points on each integration path (CIG and CEG multi-step). The path derivative is
approximated by finite differences between consecutive points, so more steps means a smaller
[completeness delta](#arg-return-convergence-delta) and proportionally more compute.

- 50 is a reasonable start for CIG.
- For CEG, `n_steps` counts steps *per baseline*: the total is `B × n_steps`.
- Nonlinear surrogates need more steps than linear ones.

(arg-separate-effects)=
### `separate_effects`

If `True`, return the two components `(direct, indirect)` defined in
{doc}`concepts`. If `False`, return their sum. The sum is what you compare against other attribution
methods. The split is what tells you whether a feature matters in itself or through what it causes.

(arg-has-closed-form-jacobian)=
### `has_closed_form_jacobian`

If `True`, the surrogate's Jacobian is taken from its
`compute_input_jacobian(x, dim_start, dim_end)` method instead of autograd. Set it for
{class}`~causalig.models.MaskedGlobalLinear` and for any custom surrogate that implements that method.
It is much faster and lighter for large `d`, because it avoids evaluating the surrogate once per child
on masked copies of the input.

Both paths give the same attributions, up to float error, when the surrogate is consistent with the
graph. {ref}`validate-surrogate` shows how to check this.

(arg-include-second-order-indirect)=
### `include_second_order_indirect`

If `True`, the indirect effect also propagates credit two hops upstream: from the feature the model
reads, through its parent, to its grandparent. It roughly doubles the surrogate cost. Completeness is
unaffected. {doc}`concepts` has the math and a worked example.

(arg-return-convergence-delta)=
### `return_convergence_delta`

If `True`, also return the per-sample completeness error

```text
delta = sum_i(direct_i + indirect_i) - (f(input) - mean_b f(baseline_b))
```

It is appended as the last element of the returned tuple (see below). Compare `|delta|` to
`|f(input) - f(baseline)|`. If the ratio is not small, increase `n_steps` (or the number of baselines,
for single-step CEG).

(arg-seed)=
### `seed`

Seeds the per-baseline $\alpha$ draws of single-step CEG, the only method with random path points. CIG
and CEG multi-step use an even grid and are deterministic. Baseline selection is up to you, so seed
that yourself if you sample baselines randomly.

## Chunking arguments

These arguments trade speed for memory and never change the result. One "pass" evaluates the model and
surrogate on `N = steps × baselines × batch` path points at once:

- `step_chunk_size` caps the steps per pass;
- `baselines_chunk_size` caps the baselines per pass;
- the `batch` part is controlled by how many inputs you pass in.

`dim_chunk_size` caps how many surrogate outputs (children) have their Jacobian rows computed at once.
Each pass builds a Jacobian block of `N × dim_chunk_size × d` floats. On the autograd path the surrogate
also runs on `N × dim_chunk_size` masked copies of the input and keeps their autograd graph, which
dominates memory.

| Argument | Default (`None`) | Lower it when |
|---|---|---|
| `dim_chunk_size` | all `d` outputs at once | `d` is large. Memory grows as `d²` without it, so this is usually the first knob to turn. |
| `step_chunk_size` | all `n_steps` at once | `n_steps` is large. Must be greater than 1. |
| `baselines_chunk_size` | all baselines at once | You average over many baselines. |

(performance)=
### Rules of thumb for large `d`

- Prefer a surrogate with a closed-form Jacobian and set `has_closed_form_jacobian=True`. It is the
  biggest saving by far.
- If you still run out of memory, lower `dim_chunk_size` (and then `step_chunk_size`) until it fits.
  Results do not change; only speed does.
- Explain inputs in batches: call `attribute()` on slices of your data instead of all of it at once.
- Multi-output models with `target=None` cost one backward pass per output. Pass `target` if you only
  need one.

For a sense of scale, peak GPU memory of CIG on 100 inputs, 100 features and `n_steps=50`
(`N = 5000` path points):

| Jacobian | `dim_chunk_size` | Peak memory |
|---|---|---|
| autograd (default) | `None` | 9.8 GB |
| autograd (default) | `25` | 2.5 GB |
| closed form | `None` | 0.53 GB |
| closed form | `25` | 0.26 GB |

## Return values

The return type depends on `separate_effects` and `return_convergence_delta`:

| `separate_effects` | `return_convergence_delta` | Returns |
|---|---|---|
| `False` | `False` | `attributions` |
| `True` | `False` | `(direct, indirect)` |
| `False` | `True` | `(attributions, delta)` |
| `True` | `True` | `(direct, indirect, delta)` |

The shapes depend on the input and the model output:

| Input | Model output / `target` | Attributions | `delta` |
|---|---|---|---|
| `(batch, d)` | scalar output, or `target` set | `(batch, d)` | `(batch,)` |
| `(batch, d)` | `(batch, C)`, `C > 1`, `target=None` | `(batch, d, C)` | `(batch, C)` |
| `(d,)` | scalar output, or `target` set | `(d,)` | `()` |

All returned tensors are on the input's device and detached from the autograd graph.
