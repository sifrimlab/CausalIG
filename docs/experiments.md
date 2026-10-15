# Reproducing the paper

`causalig` works with any differentiable model and any user-supplied causal graph; the
{doc}`quickstart` is all you need for your own data. The repository additionally contains the code for
the paper's **synthetic CATE benchmark**:

- the `experiments/` package holds the data simulation, models, metrics and plots;
- `scripts/` holds one runner per experiment, plus a script that reports on its results.

The Dyngen and IFN-β experiments are not included. Their setup is described in the paper's appendix.

## Setup

```bash
git clone https://github.com/sifrimlab/CausalIG.git
cd CausalIG
pip install -e ".[shap,experiments]"
```

`experiments/` is **not** installed as a package. Run the scripts, and the example below, from the
repository root.

## Running an experiment from Python

`CausalAttributionExperiment` runs a full experiment:

1. it simulates a random DAG and SCM;
2. it trains a Lasso model on the data;
3. it computes attributions with the requested methods;
4. it scores them against the conditional average treatment effect (CATE) of each feature.

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

### Main configuration fields

`SyntheticExperimentConfig` is a dataclass defined in `experiments/config/experiment.py`; see that
file for the full list of fields. The ones you are most likely to change:

| Field | Default | Meaning |
|---|---|---|
| `num_features`, `num_causal` | `100`, `20` | Graph size, and how many features truly drive the outcome. |
| `num_samples`, `num_samples_train` | `1000`, `5000` | Samples explained (the first `num_samples` rows), and samples simulated from the SCM to train the model and surrogate. |
| `noise_multiplier` | `0.1` | Scale of the SCM noise. |
| `max_depth` | `None` | Limits the DAG depth. The paper's scripts use `2`. |
| `ordering` | `None` | Topological order of the features. **Required.** |
| `sim_autoencoder` | `True` | `True` uses the true SCM as an oracle surrogate. `False` trains a `MaskedGlobalLinear` on the data. |
| `structural` | `False` | Structural mode for the oracle surrogate. **Set `True`**, as all paper scripts do (see {doc}`user_guide/surrogates`). |
| `surrogate_epochs`, `surrogate_lr`, `surrogate_batch_size` | `10`, `1e-3`, `512` | Training settings for the learned surrogate. |
| `attribution_n_steps` | `200` | Total gradient steps. CEG divides them across its baselines. |
| `attribution_n_baselines` | `None` | Number of CEG baselines. **Required for CEG.** |
| `attribution_step_chunk_size`, `attribution_dim_chunk_size` | `100`, `None` | Memory knobs (see {ref}`performance`). |
| `graph_perturbation_method`, `edge_drop_frac` | `"remove"`, `0.0` | Corrupt the graph given to the methods (`"remove"`, `"reverse"` or `"add_dag"`) to test robustness. |
| `device` | CUDA if available | Compute device. |
| `output_dir`, `log_dir` | `"."`, `"logs"` | Where results and logs are written. |

## Scripts

Each experiment is a `run_*.py` script with hardcoded parameters at the top. Edit them, then run
`python scripts/run_*.py`. Each one has a companion `plot_*.py` or `print_*.py` script with a
`--results-dir` option.

| What it tests | Run | Report |
|---|---|---|
| Attribution quality vs. CATE ground truth, across feature-count scale | `run_synthetic_scaling.py` | `plot_synthetic_scaling.py` (`--node-set all_nodes` to evaluate on all nodes instead of root causes) |
| Robustness to removed, reversed or spurious causal graph edges | `run_synthetic_scaling_edge_robustness.py` (after `run_synthetic_scaling.py`; writes to the same directory) | `plot_synthetic_scaling.py --edge-drop-overlay --overlay-method {remove,reverse,add_dag} --methods CIG IG` |
| How well attributions satisfy the completeness axiom | `run_synthetic_completeness_comparison.py` | `print_synthetic_completeness_table.py` |
| Completeness-delta convergence vs. number of integration steps | `run_synthetic_nsteps_ablation.py` | `plot_nsteps_ablation.py` |
| How completeness delta is distributed across individual explained samples, not just averaged | *(reuses the results of `run_synthetic_completeness_comparison.py`)* | `plot_completeness_delta_per_sample.py --compare-method IG` (774 features by default, as in the paper; `--num-features N` for another size, or pass any result file directly) |
| CIG with a trained surrogate vs. the true SCM as surrogate | `run_synthetic_surrogate_comparison.py` | `plot_synthetic_surrogate_comparison.py` |
| Whether the additive `MaskedGlobalLinear` surrogate mishandles genuine multi-parent interactions | `run_synthetic_interaction_indirect_effect_multiseed.py` | `print_synthetic_interaction_diagnostic.py` |

All `run_*.py` scripts write to `results/<experiment>/` in the repository root, which is git-ignored. The
`plot_*` and `print_*` scripts read from there by default, so they can be run from any directory.
