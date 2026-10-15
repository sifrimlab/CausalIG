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

    shap_n_permutations, shap_n_samples = (10, 50) if num_features < 464 else (4, 10)

    # Configuration
    config = SyntheticExperimentConfig(
        num_features=num_features,
        num_causal=num_features // 5,  # 20%
        num_samples_train=20000,
        num_samples=100,
        noise_multiplier=0.1,
        ancestors_in_causal=True,
        only_eldest_ancestors=True,
        sim_autoencoder=True,
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
        # Shapley sampling budget is reduced from 464 features on, where the
        # full budget (10 permutations, 50 background samples) became prohibitive
        causal_shap_n_permutations=shap_n_permutations,
        conditional_shap_n_permutations=shap_n_permutations,
        causal_shap_n_samples=shap_n_samples,
        conditional_shap_n_samples=shap_n_samples,
    )

    # Run experiment
    experiment = CausalAttributionExperiment(config)
    seeds = [0, 1, 2, 3, 4]
    results = experiment.run_multiple_seeds(
        names=["CIG", "IG", "CausalSHAP", "CondSHAP", "CEG"], seeds=seeds
    )
