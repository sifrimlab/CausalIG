"""
Robustness of CIG to graph misspecification, across problem scale.

Companion to `run_synthetic_scaling.py`: reruns the same num_features sweep,
but for CIG only, with the graph handed to it corrupted by one of three
DAG-safe methods (see `experiments.utils.graph_perturbation`):

- "remove": drop existing edges (the original ablation).
- "add_dag": add spurious edges, cycle-safe by construction.
- "reverse": flip the direction of existing edges, cycle-safe by construction.

Writes into the same output_dir as `run_synthetic_scaling.py` so
`plot_synthetic_scaling.py` can overlay these CIG curves on top of the main
scaling figure. Data generation and CATE ground truth are unaffected — only
the graph handed to CIG (and, for add_dag/reverse, the surrogate trained on
it — see below) is corrupted.

IMPORTANT — why "remove" and "add_dag"/"reverse" use different surrogates:
with `sim_autoencoder=True` (the oracle surrogate, i.e. the true SCM standing
in for the surrogate), CIG's structural forward pass reads each node's true
parent columns directly off the SCM's own fixed adjacency, ignoring whatever
extra columns a corrupted graph happens to unmask. So "remove" (which zeroes
out true-parent columns the SCM actually needs) still degrades CIG correctly,
but "add_dag"/"reverse" (which only add or redirect edges, never removing a
true parent) would show *no effect at all* under the oracle surrogate — verified
directly: feeding CIG the true graph vs. an almost-complete graph produced
bit-identical attributions. To make add_dag/reverse meaningful, this script
uses a *learned* `MaskedGlobalLinear` surrogate (`sim_autoencoder=False`)
instead for those two methods, fit fresh per fraction on that fraction's own
corrupted graph (mirroring `run_synthetic_surrogate_comparison.py`'s "learned"
variant) — so a spurious or reversed edge can actually mislead what the
surrogate believes the structure is. "remove" keeps the oracle surrogate,
matching the numbers already reported for it.

Note this makes add_dag/reverse considerably more expensive than remove: the
learned surrogate is retrained from scratch per (num_features, fraction,
seed), whereas the oracle surrogate is instantaneous.

For each (num_features, method) pair, `run_edge_drop_sweep` shares data
generation, model training, and CATE ground truth (the expensive steps,
independent of the perturbation) across all `EDGE_DROP_FRACS`. Different
*methods* still each pay for their own data/model/CATE (they're independent
`run_edge_drop_sweep` calls) — sharing across methods too would need a
further generalization, not done here.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
import numpy as np
import torch

from experiments.config import SyntheticExperimentConfig
from experiments.synthetic import CausalAttributionExperiment

# Same feature sweep as run_synthetic_scaling.py.
num_features_list = np.logspace(np.log10(100), np.log10(1000), 10, dtype=int)
EDGE_DROP_FRACS = [0.25, 0.5, 0.75]

# "remove" uses the oracle (true-SCM) surrogate, matching the original ablation.
# "add_dag"/"reverse" need a learned surrogate to have any effect at all (see
# module docstring) — hyperparameters mirror the "learned" variant in
# run_synthetic_surrogate_comparison.py.
SURROGATE_CONFIG_BY_METHOD = {
    "remove": {"sim_autoencoder": True},
    "add_dag": {
        "sim_autoencoder": False,
        "surrogate_epochs": 50,
        "surrogate_batch_size": 512,
    },
    "reverse": {
        "sim_autoencoder": False,
        "surrogate_epochs": 50,
        "surrogate_batch_size": 512,
    },
}

for perturbation_method, surrogate_overrides in SURROGATE_CONFIG_BY_METHOD.items():
    for num_features in num_features_list:
        print(f"\n{'=' * 60}")
        print(
            f"Running CIG {perturbation_method!r} sweep with "
            f"num_features={num_features}"
        )
        print(f"{'=' * 60}\n")

        config = SyntheticExperimentConfig(
            num_features=num_features,
            num_causal=num_features // 5,  # 20%
            num_samples_train=20000,
            num_samples=100,
            noise_multiplier=0.1,
            ancestors_in_causal=True,
            only_eldest_ancestors=True,
            structural=True,
            output_dir=str(RESULTS_DIR / "synthetic_general_sim_autoencoder"),
            log_dir=str(RESULTS_DIR / "synthetic_general_sim_autoencoder"),
            ordering=np.arange(0, num_features),
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            lasso_lambda=1e-1,
            max_depth=2,
            seed=42,
            attribution_step_chunk_size=100,
            attribution_dim_chunk_size=100,
            attribution_n_steps=100,
            attribution_n_baselines=10,
            **surrogate_overrides,
        )

        experiment = CausalAttributionExperiment(config)
        seeds = [0, 1, 2, 3, 4]
        results = experiment.run_edge_drop_sweep(
            edge_drop_fracs=EDGE_DROP_FRACS,
            seeds=seeds,
            names=["CIG"],
            perturbation_method=perturbation_method,
        )
