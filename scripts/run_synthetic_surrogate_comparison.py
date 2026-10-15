"""
CIG with a surrogate trained on simulated data vs. CIG with the true SCM.

Compares four variants against the same CATE ground truth, sharing data
generation, model training, and CATE computation across them for each seed:
  - "CIG_oracle": CIG using the true SCM as surrogate (`sim_autoencoder=True`)
  - "IG_oracle": plain Integrated Gradients (doesn't depend on the surrogate;
    computed alongside the oracle variant since it's already cheap there)
  - "CIG_learned": CIG using a `MaskedGlobalLinear` surrogate (with its
    default per-parent basis expansion, i.e. a generalized additive model)
    fit to the simulated data (`sim_autoencoder=False`), i.e. only the graph
    structure is given, the mechanism is learned — mirroring how CIG would
    be used on real data.
  - "CIG_learned_linear": same as "CIG_learned" but with
    `surrogate_use_basis_expansion=False`, i.e. a plain linear (no per-parent
    nonlinearity) surrogate — isolates how much of the "learned" variant's
    quality comes from the basis expansion vs. just the graph structure.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
import numpy as np
import torch

from experiments.config import SyntheticExperimentConfig
from experiments.synthetic import CausalAttributionExperiment

# Generate 10 different feature counts between 100 and 1000
num_features_list = np.logspace(np.log10(100), np.log10(1000), 10, dtype=int)[0:10]

for num_features in num_features_list:
    print(f"\n{'='*60}")
    print(f"Running experiment with num_features={num_features}")
    print(f"{'='*60}\n")

    config = SyntheticExperimentConfig(
        num_features=num_features,
        num_causal=num_features // 5,  # 20%
        num_samples_train=20000,
        num_samples=100,
        noise_multiplier=0.1,
        ancestors_in_causal=True,
        only_eldest_ancestors=True,
        structural=True,
        output_dir=str(RESULTS_DIR / "synthetic_surrogate_comparison"),
        log_dir=str(RESULTS_DIR / "synthetic_surrogate_comparison"),
        ordering=np.arange(0, num_features),
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        lasso_lambda=1e-1,
        max_depth=2,
        attribution_step_chunk_size=100,
        attribution_dim_chunk_size=100,
        attribution_n_steps=100,
        surrogate_epochs=50,
        surrogate_batch_size=512,
    )

    experiment = CausalAttributionExperiment(config)
    seeds = [0, 1, 2, 3, 4]

    results = experiment.run_surrogate_comparison(
        surrogate_configs={
            "oracle": {
                "sim_autoencoder": True,
                "attribution_names": ("CIG", "IG"),
            },
            "learned": {
                "sim_autoencoder": False,
                "attribution_names": ("CIG",),
            },
            "learned_linear": {
                "sim_autoencoder": False,
                "surrogate_use_basis_expansion": False,
                "attribution_names": ("CIG",),
            },
        },
        seeds=seeds,
    )

    for node_set, methods in results["metrics_by_node_set"].items():
        print(f"\n--- {node_set} ---")
        for method, metrics in methods.items():
            spearman = metrics["spearman"]
            print(
                f"  {method}: spearman={spearman['mean']:.3f} (+/- {spearman['std']:.3f})"
            )
