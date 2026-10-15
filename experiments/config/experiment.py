"""
Configuration classes for experiments.
"""

import logging
from dataclasses import dataclass, field

import torch

from experiments.utils.graph_perturbation import perturbation_filename_suffix


@dataclass
class SyntheticExperimentConfig:
    """
    Configuration for synthetic attribution experiments.

    This configuration class encapsulates all parameters needed to run
    a synthetic data attribution experiment.
    """

    # Data generation parameters
    seed: int = None
    num_features: int = 100
    num_causal: int = 20
    num_samples: int = 1000
    num_samples_train: int = 5000
    noise_multiplier: float = 0.1
    # If None, uses random DAG without depth constraint
    max_depth: int | None = None

    # Graph-misspecification robustness ablation: `graph_perturbation_method`
    # (one of experiments.utils.graph_perturbation.PerturbationMethod's
    # values — "remove", "add_dag", "reverse") selects how the graph handed
    # to the attribution methods is corrupted; `edge_drop_frac` is the
    # fraction of that method's eligible edges to perturb (kept under its
    # original name since "remove" was the only method when it was added).
    # The data is still generated from (and CATE ground truth still computed
    # against) the true, uncorrupted DAG.
    #
    # IMPORTANT: with sim_autoencoder=True (the true SCM standing in as the
    # surrogate), CIG reads each node's true-parent columns straight off the
    # SCM's own fixed adjacency and ignores anything else unmasked — so
    # "add_dag" (which only adds edges, never removing a true parent) has
    # *no effect at all* under that surrogate, however large the corruption
    # (verified: bit-identical attributions). "reverse" degrades identically
    # to "remove" for the same reason (the added reverse-direction edge is
    # likewise ignored). Only "remove" is meaningful with sim_autoencoder=True;
    # "add_dag"/"reverse" need a learned surrogate (sim_autoencoder=False)
    # to actually be misled by a spurious or reversed edge. See
    # scripts/run_synthetic_scaling_edge_robustness.py and
    # tests/test_causal_experiment.py's oracle/learned-surrogate tests.
    graph_perturbation_method: str = "remove"
    edge_drop_frac: float = 0.0

    # Causal feature processing
    ancestors_in_causal: bool = False
    only_eldest_ancestors: bool = False

    # Autoencoder configuration
    sim_autoencoder: bool = True  # Use SCM as autoencoder
    structural: bool = False  # Whether to use structural forward pass in autoencoder

    # Surrogate training (if sim_autoencoder=False): a MaskedGlobalLinear
    # fit on the simulated data
    surrogate_epochs: int = 10
    surrogate_batch_size: int = 512
    surrogate_lr: float = 1e-3
    ordering: list[int] = None
    surrogate_use_basis_expansion: bool = True
    surrogate_weight_decay: float = 1e-5

    # Model training
    lasso_lambda: float = 1e-3
    lasso_lr: float = 1e-2
    lasso_steps: int = 5000

    # Attribution parameters
    attribution_n_baselines: int | None = None
    attribution_n_steps: int = 200
    attribution_baseline: torch.Tensor | None = None  # If None, uses zeros
    attribution_step_chunk_size: int | None = 100
    attribution_dim_chunk_size: int | None = None
    conditional_shap_n_samples: int = 50
    conditional_shap_n_permutations: int = 10
    causal_shap_n_samples: int = 50
    causal_shap_n_permutations: int = 10
    causal_shap_noise_sigma: float = 1e-6
    cate_n_noise_samples: int = 10

    # Device
    device: str | torch.device = field(
        default_factory=lambda: torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
    )

    # Output paths
    output_dir: str = "."
    save_plot: bool = True
    save_results: bool = True

    # Logging
    log_dir: str = "logs"
    log_level: int = logging.INFO
    log_to_console: bool = True

    def __post_init__(self):
        """Validate and set defaults after initialization."""
        # Ensure num_causal is reasonable
        if self.num_causal > self.num_features:
            raise ValueError(
                f"num_causal ({self.num_causal}) cannot exceed num_features ({self.num_features})"
            )

        # Convert device string to torch.device if needed
        if isinstance(self.device, str):
            self.device = torch.device(self.device)

        # Set default baseline if not provided
        if self.attribution_baseline is None:
            self.attribution_baseline = torch.zeros(
                self.num_features, dtype=torch.float32, device=self.device
            )
        else:
            # Ensure baseline is on correct device
            self.attribution_baseline = self.attribution_baseline.to(self.device)

    def _edge_drop_suffix(self) -> str:
        """Filename fragment encoding the perturbation method/fraction, omitted when unused."""
        return perturbation_filename_suffix(
            self.graph_perturbation_method, self.edge_drop_frac
        )

    def get_results_filename(self) -> str:
        """Generate results filename based on configuration."""

        return (
            f"synthetic_causal_metrics_{self.num_features}_noise{self.noise_multiplier}_"
            f"ancestors{self.ancestors_in_causal}_eldest{self.only_eldest_ancestors}_"
            f"sim_autoencoder{self.sim_autoencoder}_structural{self.structural}"
            f"{self._edge_drop_suffix()}.json"
        )
