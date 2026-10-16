[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![MIT License][license-shield]][license-url]
[![CI][ci-shield]][ci-url]
[![Documentation][docs-shield]][docs-url]

<!-- MARKDOWN LINKS & IMAGES -->
[ci-shield]: https://img.shields.io/github/actions/workflow/status/sifrimlab/CausalIG/ci.yml?branch=main&style=for-the-badge&label=CI
[ci-url]: https://github.com/sifrimlab/CausalIG/actions/workflows/ci.yml
[docs-shield]: https://img.shields.io/badge/docs-done-green?style=for-the-badge
[docs-url]: https://sifrimlab.org/CausalIG/
[contributors-shield]: https://img.shields.io/github/contributors/sifrimlab/CausalIG.svg?style=for-the-badge
[contributors-url]: https://github.com/sifrimlab/CausalIG/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/sifrimlab/CausalIG.svg?style=for-the-badge
[forks-url]: https://github.com/sifrimlab/CausalIG/network/members
[stars-shield]: https://img.shields.io/github/stars/sifrimlab/CausalIG.svg?style=for-the-badge
[stars-url]: https://github.com/sifrimlab/CausalIG/stargazers
[issues-shield]: https://img.shields.io/github/issues/sifrimlab/CausalIG.svg?style=for-the-badge
[issues-url]: https://github.com/sifrimlab/CausalIG/issues
[license-shield]: https://img.shields.io/github/license/sifrimlab/CausalIG.svg?style=for-the-badge
[license-url]: https://github.com/sifrimlab/CausalIG/blob/main/LICENSE


﻿
<!-- PROJECT LOGO -->
<br />
<p align="center">
  <img src="docs/logo.png" alt="logo" align="center" width="40%">
  <h3 align="center">Causal Integrated Gradients: Scalable Feature Attribution Along Causal Paths</h3>
  <p align="center">
    <br />
    <a href="https://github.com/sifrimlab/CausalIG/issues">Report Bug</a>
    ·
    <a href="https://github.com/sifrimlab/CausalIG/pulls">Add Feature</a>
  </p>
</p>

Official implementation of **Causal Integrated Gradients (CIG)** and **Causal Expected Gradients (CEG)**: principled extensions of Integrated Gradients that incorporate causal relationships between input features, while preserving its axiomatic guarantees and scalability. By reparametrizing the integration path to follow the causal structure of the inputs, CIG/CEG decompose each feature's attribution into a **direct** effect and **indirect** effects mediated through its causal descendants, scaling to high-dimensional settings where existing dependency-aware methods (conditional SHAP, causal Shapley values) become computationally prohibitive.

> **Paper**: J. Michiels\*, A. Ismail\*, M. De Vos, A. Sifrim. *Causal Integrated Gradients: Scalable Feature Attribution Along Causal Paths.* arXiv preprint, 2026. [arXiv:XXXX.XXXXX](https://arxiv.org/abs/XXXX.XXXXX) (\* equal contribution)

---

## Overview

Standard attribution methods (Integrated Gradients, SHAP) treat input features as independent. When features are causally related, ignoring these dependencies causes such methods to misattribute credit toward downstream, correlated features rather than their true upstream causal drivers e.g. assigning high importance to a target gene co-expressed with its regulating transcription factor, rather than to the transcription factor itself. This matters whenever the goal is to recover the actual causal drivers rather than any predictive correlate. CIG and CEG address this by decomposing each feature's attribution into a **direct** effect (its own path to the output) and an **indirect** effect (mediated through the features it causally influences), correctly pushing attribution toward upstream drivers.

To use them, you need three ingredients:

1. **A trained differentiable model** to explain: Used as a black box, any callable `(batch, d) -> (batch, C)` works.
2. **A causal graph over the input features**: Just the *structure* (which features causally influence which), supplied as prior knowledge (domain expertise, a curated database like [OmniPath](https://omnipathdb.org/), a tool like [SCENIC](https://scenic.aertslab.org/), etc.). You provide the edges, not the edge weights or functional form.
3. **A surrogate model**, trained on your data, that learns the actual functional relationships implied by that graph, i.e. it fills in what the causal mechanisms behind those edges look like. We provide `MaskedGlobalLinear` (shipped in `causalig.models`) as a standard model, or you can bring your own: anything implementing the single-method `ModelInterface` works.

Given these three ingredients, CIG/CEG integrate gradients along a path that follows the causal graph, using the surrogate's gradients at each step to split every feature's attribution into its direct and indirect components.

**Methods implemented:**
- `CausalIntegratedGradients` (CIG): integrates gradients along a causally-conditioned path from baseline to input
- `CausalExpectedGradients` / `CausalExpectedGradientsMultiStep` (CEG): expectation over a distribution of baselines drawn from training data

Both support optional second-order (two-hop) indirect effects via `include_second_order_indirect=True`.

The repo also includes `CausalSHAP` (causal Shapley values, Heskes et al., 2020) and `ConditionalSHAP` (conditional SHAP, Aas et al., 2021), dependency-aware Shapley-value baselines used for the experimental comparisons in the paper (see [Attribution Methods](#attribution-methods)); they require the optional `shap` extra (`pip install causalig[shap]`).

---

## Installation

With conda:

```bash
conda env create -f environment.yml
conda activate causalig
pip install -e ".[shap,experiments]"   # adds shap, needed for the paper's experiments
```

Or, if you already have PyTorch ≥ 2.0 installed in your own environment:

```bash
pip install -e .
```

Optional extras:

```bash
pip install -e ".[shap]"          # SHAP-based methods (CausalSHAP, ConditionalSHAP)
pip install -e ".[experiments]"   # reproduce the paper's synthetic experiments (captum, sempler, matplotlib, scipy)
pip install -e ".[dev]"           # tests + linting (pytest, ruff, black)
```

`causalig` itself only depends on PyTorch and NumPy, the `experiments` extra is only needed to run the code under `experiments/` and `scripts/`, which reproduce the paper's synthetic benchmark and are not required to use the attribution methods on your own data.

---

## Quick Start

### Step 1: Train a surrogate on your causal graph

The surrogate learns to predict each feature from its causal parents: the third ingredient from the [Overview](#overview). `adjacency_matrix` here is the representation of your prior knowledge of the graph structure (`adjacency_matrix[i, j] = 1` means feature `i` causes feature `j`). The surrogate learns the mechanism behind each edge from `X_train`.

```python
from causalig.models import MaskedGlobalLinear, train_surrogate_model

# X_train: tensor [n_samples, d]
surrogate = MaskedGlobalLinear(adjacency_matrix=adjacency_matrix, X_train=X_train)
surrogate = train_surrogate_model(surrogate, X_train, epochs=50, device="cpu")
```

See [Surrogate Model](#surrogate-model) below for the full set of options.

### Step 2: Attribute with CIG or CEG

`model` is the trained model you want to explain. The order in which features are processed is derived from `adjacency_matrix`. `has_closed_form_jacobian=True` tells CIG/CEG to use the surrogate's exact analytic Jacobian instead of autograd.

```python
import torch
from causalig.attr import CausalIntegratedGradients

cig = CausalIntegratedGradients(model)

direct, indirect = cig.attribute(
    input=inputs,
    surrogate=surrogate,
    baseline=baseline,     # single (d,) reference point, e.g. zeros or the training mean
    n_steps=50,
    separate_effects=True,
    adjacency_matrix=adjacency_matrix,  # bool tensor (d, d)
    has_closed_form_jacobian=True,
)
```

`CausalExpectedGradientsMultiStep` (CEG) takes the same arguments, except that `baseline` becomes `baselines`, a set of reference points it averages over:

```python
from causalig.attr import CausalExpectedGradientsMultiStep

ceg = CausalExpectedGradientsMultiStep(model)

direct, indirect = ceg.attribute(
    input=inputs,
    surrogate=surrogate,
    baselines=baselines,   # (n_baselines, d), e.g. rows sampled from the training data
    n_steps=50,            # steps *per baseline*: cost is n_baselines x that of CIG
    separate_effects=True,
    adjacency_matrix=adjacency_matrix,
    has_closed_form_jacobian=True,
)
```

Always pass `adjacency_matrix` when you have it. Without it, you must pass a topological `ordering` instead, and CIG/CEG then assume every feature may depend on every feature before it.

---

## Attribution Methods

### `CausalIntegratedGradients` (CIG)

Integrates gradients along a causally-conditioned path. At each integration step, the surrogate model reconstructs features causally downstream of the current feature, decomposing the total effect into:

- **Direct effect**: path through the residual `h = x − surrogate(x)`
- **Indirect effect**: path through the surrogate output

```python
from causalig.attr import CausalIntegratedGradients

cig = CausalIntegratedGradients(model)
direct, indirect = cig.attribute(
    input, surrogate, baseline=baseline,
    n_steps=50,
    separate_effects=True,
    adjacency_matrix=adjacency_matrix,  # bool tensor (d, d), [i,j] = i causes j
    step_chunk_size=100,              # memory-saving: chunk integration steps
    dim_chunk_size=None,              # memory-saving: chunk feature dimensions
)
```

By default only one-hop (first-order) indirect effects are attributed separately. Longer paths are folded into the one-hop term, so completeness still holds. Pass `include_second_order_indirect=True` to also track two-hop indirect effects, but beware that this roughly doubles the attribution cost. 

### `CausalExpectedGradients` / `CausalExpectedGradientsMultiStep`

Replaces the fixed baseline of CIG with an expectation over baseline samples drawn from training data.

- `CausalExpectedGradients` follows the original Expected Gradients formulation directly: baseline and path position are sampled jointly, drawing a single path point per baseline.
- `CausalExpectedGradientsMultiStep` instead draws multiple path points per baseline, as Integrated Gradients does within a single baseline. For a fixed total number of gradient evaluations, this trades off number of baselines for path resolution per baseline. It does not compute more or different gradients than `CausalExpectedGradients`, just distributed differently. We find this empirically stabilizes attributions substantially.

### Baselines used in the paper's experiments: `CausalSHAP`, `ConditionalSHAP`

Dependency-aware Shapley-value methods reimplemented here mainly to produce the experimental comparisons in the paper, also usable standalone:

- `CausalSHAP` : causal Shapley values [Heskes et al., 2020]; uses the SCM to generate counterfactual interventional background samples for computing Shapley values.
- `ConditionalSHAP` : conditional SHAP [Aas et al., 2021]; imputes masked features using a Gaussian conditional distribution, marginalising out feature correlations.

### Completeness delta

`CausalIntegratedGradients`, `CausalExpectedGradients`, and `CausalExpectedGradientsMultiStep` all accept `return_convergence_delta=True` to additionally return a completeness-axiom delta: `sum(attributions) - (f(input) - mean_b f(baseline_b))`. A value near zero confirms the attributions satisfy completeness; it shrinks as `n_steps` increases (mirrors Captum's `compute_convergence_delta`).

```python
direct, indirect, delta = cig.attribute(
    input, surrogate, baseline=baseline,
    n_steps=50,
    separate_effects=True,
    adjacency_matrix=adjacency_matrix,
    return_convergence_delta=True,
)
```

---

## Surrogate Model

The surrogate predicts each feature from its causal parents. `MaskedGlobalLinear` only has weights on the edges of `adjacency_matrix`, so output `j` reads nothing but the parents of `j`, and it is trained by reconstruction of `X_train`.

Training utilities are exported from `causalig.models`:

```python
from causalig.models import MaskedGlobalLinear, train_surrogate_model

# X_train: tensor [n_samples, d]
surrogate = MaskedGlobalLinear(
    adjacency_matrix=adjacency_matrix, # 0/1 tensor [d, d]
    use_basis_expansion=True,  # nonlinear per-edge basis; set False for a linear fit
    X_train=X_train,           # optional, its mean/std standardize the basis inputs (recommended)
)
surrogate = train_surrogate_model(surrogate, X_train, epochs=50, device="cpu")
```

With `use_basis_expansion=True`, each parent feature is expanded into polynomial and truncated-power (learnable knots) terms, plus optional Fourier terms (`fourier_features`), before the per-edge linear map.

`MaskedGlobalLinear` reconstructs each feature from its direct parents over the causal graph, parameterizing only the edges present in `adjacency_matrix` (sparse in the number of edges, not dense in `d²`). It exposes an exact closed-form input Jacobian: pass `has_closed_form_jacobian=True` to CIG/CEG to use it instead of autograd.

---

## Experiments

`causalig` itself works with any differentiable model and any user-supplied causal graph, the Quick Start above is all you need to use it on your own data. `experiments/` and `scripts/` additionally contain the code used to run the experiments in our paper for the synthetic CATE benchmark. Requires the `experiments` extra (`pip install -e ".[experiments]"`). `experiments/` is not installed as a package: run the scripts and the example below from a clone of this repository (the example from the repository root).

### Running an experiment

```python
import numpy as np

from experiments.synthetic import CausalAttributionExperiment
from experiments.config.experiment import SyntheticExperimentConfig

num_features = 100
config = SyntheticExperimentConfig(
    num_features=num_features,
    num_causal=20,
    num_samples=100,                       # number of samples explained (as in the paper)
    noise_multiplier=0.1,
    structural=True,                       # true SCM as surrogate applies each mechanism to the given parent values (as all paper scripts do)
    ordering=np.arange(num_features),      # required: topological feature ordering
    attribution_n_baselines=10,            # required for CEG: number of baseline samples
    attribution_dim_chunk_size=25,         # caps GPU memory (~2 GB here); raise it if you have more
    output_dir="results/example",          # results/ is git-ignored
)

experiment = CausalAttributionExperiment(config)
results = experiment.run(names=['CIG', 'CEG', 'IG'])
```

### Scripts

Each experiment is a `run_*.py` script (edit the hardcoded parameters at the top, then `python scripts/run_*.py`) paired with a companion script that reports on its results (`plot_*.py` or `print_*.py`, with a proper `--results-dir` CLI):

| What it tests | Run | Report |
|---|---|---|
| Attribution quality vs. CATE ground truth, across feature-count scale | `run_synthetic_scaling.py` | `plot_synthetic_scaling.py` (`--node-set all_nodes` to evaluate on all nodes instead of root causes) |
| Robustness to removed, reversed or spurious causal graph edges | `run_synthetic_scaling_edge_robustness.py` (after `run_synthetic_scaling.py`; writes to the same directory) | `plot_synthetic_scaling.py --edge-drop-overlay --overlay-method {remove,reverse,add_dag} --methods CIG IG` |
| How well attributions satisfy the completeness axiom | `run_synthetic_completeness_comparison.py` | `print_synthetic_completeness_table.py` |
| Completeness-delta convergence vs. number of integration steps | `run_synthetic_nsteps_ablation.py` | `plot_nsteps_ablation.py` |
| How completeness delta is distributed across individual explained samples, not just averaged | *(reuses the results of `run_synthetic_completeness_comparison.py`)* | `plot_completeness_delta_per_sample.py --compare-method IG` (774 features by default, as in the paper; `--num-features N` for another size, or pass any result file directly) |
| CIG with a trained surrogate vs. the true SCM as surrogate | `run_synthetic_surrogate_comparison.py` | `plot_synthetic_surrogate_comparison.py` |
| Whether the additive `MaskedGlobalLinear` surrogate mishandles genuine multi-parent interactions | `run_synthetic_interaction_indirect_effect_multiseed.py` | `print_synthetic_interaction_diagnostic.py` |

All `run_*.py` scripts write to `results/<experiment>/` in the repository root (git-ignored), and the `plot_*`/`print_*` scripts read from there by default, so they can be run from any directory.

---

## Citation

If you use this code, please cite:

```bibtex
@article{michiels2026causalig,
  title   = {Causal Integrated Gradients: Scalable Feature Attribution Along Causal Paths},
  author  = {Michiels, Joran and Ismail, Anis and De Vos, Maarten and Sifrim, Alejandro},
  journal = {arXiv preprint arXiv:XXXX.XXXXX},
  year    = {2026},
}
```

---

## Dependencies

Core (`causalig`):
- PyTorch ≥ 2.0
- NumPy

Optional:
- shap (`[shap]` extra — `CausalSHAP`, `ConditionalSHAP`)
- Captum ≥ 0.7, sempler, matplotlib, scipy (`[experiments]` extra — reproducing the paper's synthetic benchmark)
