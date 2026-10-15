"""
CIG completeness-delta convergence vs. `n_steps`, across problem scale and
surrogate type.

Companion to `run_synthetic_completeness_comparison.py`: reruns the same
num_features points, but at several `attribution_n_steps` values, to check
whether CIG's relative completeness delta (see
`scripts/print_synthetic_completeness_table.py`) actually shrinks as the
Riemann-integration step count grows, and whether that convergence rate
depends on problem scale or on the surrogate used.

Sweeps two surrogate variants:
  - "oracle": the true SCM as surrogate (`sim_autoencoder=True`) — the
    smooth, kink-free case.
  - "learned": a `MaskedGlobalLinear` surrogate fit to the simulated data
    (`sim_autoencoder=False`) — mirrors how CIG is actually used on real
    data, where the true SCM isn't available.

Uses `CausalAttributionExperiment.run_nsteps_sweep` once per surrogate
variant, which shares data generation, model training, and CATE
ground-truth computation across all `n_steps` values for a given
`num_features` and seed — none of those three steps depend on `n_steps`,
only the CIG attribution integration does. CATE in particular has an
O(num_features) Monte Carlo loop (one SCM `forward_do` per feature — see
`experiments.data.synthetic.compute_cate`) and dominates runtime at large
`num_features`; sharing it avoids recomputing it once per `n_steps` value
for no reason. Data/model/CATE are still regenerated once per surrogate
variant (each gets its own `run_nsteps_sweep` call) — the two variants are
otherwise independent experiments and neither shares that cost with the
other.

`save_plot=False` here (unlike run_synthetic_completeness_comparison.py) —
per-seed CATE comparison plots aren't needed for this diagnostic and were
the dominant cost in the original (non-shared) sweep.

Each variant's `n_steps` results land in their own output_dir, distinguished
by an `_nsteps<N>` filename suffix within that directory; use
`scripts/plot_nsteps_ablation.py --learned-results-dir ...` to overlay both
on one plot.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
import numpy as np
import torch

from experiments.config import SyntheticExperimentConfig
from experiments.synthetic import CausalAttributionExperiment

# Start with just 100 features (cheap); add more points (e.g. 1000) once
# you've confirmed the convergence trend and have time for the CATE cost
# at larger num_features.
num_features_list = [100]
N_STEPS_LIST = [50, 100, 200, 400, 800, 1600, 3200]

SURROGATE_VARIANTS = {
    "oracle": {
        "output_dir": str(RESULTS_DIR / "synthetic_nsteps_ablation"),
        "config_overrides": {"sim_autoencoder": True},
    },
    "learned": {
        "output_dir": str(RESULTS_DIR / "synthetic_nsteps_ablation_learned_surrogate"),
        "config_overrides": {
            "sim_autoencoder": False,
            "surrogate_epochs": 50,
            "surrogate_batch_size": 512,
        },
    },
}

for variant_label, variant in SURROGATE_VARIANTS.items():
    for num_features in num_features_list:
        output_dir = variant["output_dir"]

        print(f"\n{'=' * 60}")
        print(
            f"variant={variant_label}, num_features={num_features}, "
            f"n_steps in {N_STEPS_LIST}"
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
            output_dir=output_dir,
            log_dir=output_dir,
            ordering=np.arange(0, num_features),
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            lasso_lambda=1e-1,
            max_depth=2,
            seed=42,
            attribution_step_chunk_size=50,
            attribution_dim_chunk_size=100,
            attribution_n_baselines=10,
            save_plot=False,
            **variant["config_overrides"],
        )

        experiment = CausalAttributionExperiment(config)
        seeds = [0, 1, 2, 3, 4]
        experiment.run_nsteps_sweep(
            n_steps_list=N_STEPS_LIST, seeds=seeds, names=["CIG", "IG"]
        )
