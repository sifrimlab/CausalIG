"""
Multi-seed pooled indirect-effect-discrepancy test for interaction nodes.

A single-seed in-graph indirect-effect test showed a modest aggregate
correlation (mean|r| ~ 0.11 at num_features=1000) that, on inspection of
individual nodes, turned out to be a real, heterogeneous mixture: some nodes
show a clear, significant discrepancy between the oracle's indirect effect
and the learned (additive) surrogate's indirect effect, correlated with the
co-parent exactly as a `g = TF*z` interaction predicts, while others show
none. The oracle here uses the SAME convention as `compute_cate`/CIG's oracle
throughout the codebase (the closure applied directly to standardized
`features_tensor` -- NOT the raw-recovery correction used by the noise-floor
diagnostics, which answers a different question).

Rather than push `num_features` up to get more eligible nodes (the oracle's
per-call cost scales with `num_features`, since it re-runs a Python loop
over every SCM node each call -- this becomes impractically slow past a few
hundred features), this pools eligible nodes across multiple seeds at a
fixed, modest `num_features`. Eligible nodes are `interaction`-type AND
literally in `causal_indices` (required for single-hop CIG's indirect
effect to be nonzero at all).

Reuses `CausalAttributionExperiment`, `MaskedGlobalLinear`, the existing
`CausalIntegratedGradients` class, and the `restricted_adjacency` helper --
no new architecture, no new DGP.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"

import json

import numpy as np
import torch

from causalig.attr import CausalIntegratedGradients
from experiments.config import SyntheticExperimentConfig
from experiments.diagnostics.stats_utils import pearson_with_stats, select_nodes_by_type
from experiments.synthetic import CausalAttributionExperiment, set_determinism

NUM_FEATURES = 300
SEEDS = [0, 1, 2, 3, 4]
N_EVAL = 100
N_STEPS = 30


def restricted_adjacency(dag_tensor: torch.Tensor, child: int) -> torch.Tensor:
    """Zero every column except `child`'s, so CIG's sum over children collapses to one edge."""
    masked = torch.zeros_like(dag_tensor)
    masked[:, child] = dag_tensor[:, child]
    return masked


def indirect_effect_discrepancy_for_node(
    node,
    meta,
    X_eval,
    model,
    oracle_autoencoder,
    learned_surrogate,
    dag_tensor,
    ordering,
    baseline,
):
    """
    Correlate against the PRODUCT `x_i * x_k`, not the bare co-parent: the
    indirect effect is a path integral from a zero baseline, so
    `dx_i/dalpha` itself equals `x_i` (not 1) -- the discrepancy picks up an
    extra factor of `x_i` beyond the co-parent dependence already in the
    Jacobian. Confirmed empirically: correlating against the bare co-parent
    gave r=-0.16 (p=0.12); against the product, r=0.83 (p=7e-27), on the
    same data.

    Also reports the relative error `RMSE(discrepancy) / RMSE(oracle)` from
    the SAME pooled (i, k) discrepancy tensor used for the correlation, so
    the two statistics always describe the same set of nodes (same `n`) --
    computing them from separate runs previously produced mismatched `n`'s
    across tables, which is confusing to a reader.
    """
    cig = CausalIntegratedGradients(model)
    adj = restricted_adjacency(dag_tensor, node)

    _, oracle_indirect = cig.attribute(
        X_eval,
        surrogate=oracle_autoencoder,
        ordering=ordering,
        baseline=baseline,
        n_steps=N_STEPS,
        adjacency_matrix=adj,
        has_closed_form_jacobian=False,
        separate_effects=True,
    )
    _, learned_indirect = cig.attribute(
        X_eval,
        surrogate=learned_surrogate,
        ordering=ordering,
        baseline=baseline,
        n_steps=N_STEPS,
        adjacency_matrix=adj,
        has_closed_form_jacobian=True,
        separate_effects=True,
    )
    discrepancy = oracle_indirect - learned_indirect

    parents = meta["parents"]
    i_idx, k_idx = parents[0], parents[1]
    disc_i, disc_k = discrepancy[:, i_idx], discrepancy[:, k_idx]
    oracle_i, oracle_k = oracle_indirect[:, i_idx], oracle_indirect[:, k_idx]
    x_i, x_k = X_eval[:, i_idx], X_eval[:, k_idx]
    product = x_i * x_k
    pooled_disc = torch.cat([disc_i, disc_k])
    pooled_oracle = torch.cat([oracle_i, oracle_k])
    pooled_coparent = torch.cat([product, product])

    corr = pearson_with_stats(pooled_disc, pooled_coparent)
    rmse_oracle = (pooled_oracle**2).mean().sqrt().item()
    rmse_disc = (pooled_disc**2).mean().sqrt().item()
    relative_error = rmse_disc / rmse_oracle if rmse_oracle > 1e-8 else float("nan")
    return {**corr, "relative_error": relative_error}


def run_seed(seed: int) -> list[dict]:
    print(f"\n{'=' * 60}")
    print(f"seed={seed}")
    print(f"{'=' * 60}")
    # This script calls the experiment's steps directly instead of `run()`,
    # so it has to seed torch (Lasso, surrogate training) itself.
    set_determinism(seed)

    config = SyntheticExperimentConfig(
        num_features=NUM_FEATURES,
        num_causal=NUM_FEATURES // 5,
        num_samples_train=20000,
        num_samples=N_EVAL,
        structural=True,
        max_depth=2,
        seed=seed,
        ordering=np.arange(NUM_FEATURES),
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        lasso_lambda=1e-1,
        output_dir=str(RESULTS_DIR / "synthetic_interaction_indirect_effect_multiseed"),
        log_dir=str(RESULTS_DIR / "synthetic_interaction_indirect_effect_multiseed"),
        save_plot=False,
        save_results=False,
        surrogate_epochs=50,
        surrogate_batch_size=512,
    )
    experiment = CausalAttributionExperiment(config)
    data = experiment.generate_data()
    model = experiment.train_model(data)

    config.sim_autoencoder = True
    oracle_autoencoder = experiment._prepare_surrogate(data)
    config.sim_autoencoder = False
    learned_surrogate = experiment._prepare_surrogate(data)

    groups = select_nodes_by_type(data["assignment_metadata"])
    causal_indices_set = {int(c) for c in data["causal_indices"]}
    interaction_causal = [
        n for n in groups.get("interaction", []) if n in causal_indices_set
    ]
    # Linear control: restricted to >=2 parents so the (i, k) pair is
    # well-defined the same way as for interaction (which always has an
    # active 2-parent pair regardless of total in-degree) -- apples-to-apples.
    linear_causal = [
        n
        for n in groups.get("linear", [])
        if n in causal_indices_set
        and len(data["assignment_metadata"][n]["parents"]) >= 2
    ]
    print(
        f"  {len(interaction_causal)} eligible interaction-causal nodes, "
        f"{len(linear_causal)} eligible linear-causal (control) nodes"
    )

    X_eval = data["features_tensor"][:N_EVAL]
    baseline = config.attribution_baseline
    dag_tensor = data["dag_tensor"]

    results = []
    for type_name, nodes in (
        ("interaction", interaction_causal),
        ("linear", linear_causal),
    ):
        for node in nodes:
            meta = data["assignment_metadata"][node]
            r = indirect_effect_discrepancy_for_node(
                node,
                meta,
                X_eval,
                model,
                oracle_autoencoder,
                learned_surrogate,
                dag_tensor,
                config.ordering,
                baseline,
            )
            w = meta["weights"]
            print(
                f"    [{type_name}] node {node}: w1w2={w[0] * w[1]:.3f}  r={r['r']:.3f}  "
                f"p={r['p']:.4f}  rel_err={r['relative_error']:.3f}"
            )
            results.append(
                {
                    "seed": seed,
                    "type": type_name,
                    "node": int(node),
                    "w1w2": float(w[0] * w[1]),
                    **r,
                }
            )

    return results


if __name__ == "__main__":
    all_results = []
    for seed in SEEDS:
        all_results.extend(run_seed(seed))

    out_dir = RESULTS_DIR / "synthetic_interaction_indirect_effect_multiseed"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "multiseed_results.json", "w") as f:
        json.dump({"per_node": all_results}, f, indent=2)
    print(f"\nSaved to {out_dir / 'multiseed_results.json'}")
    print(
        "Run scripts/print_synthetic_interaction_diagnostic.py to print the "
        "pooled per-type summary."
    )
