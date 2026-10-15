"""
Small stats/selection helpers for the additive-surrogate interaction
diagnostic (`scripts/run_synthetic_interaction_indirect_effect_multiseed.py`).
"""

from collections import defaultdict

import numpy as np
import torch
from scipy.stats import pearsonr


def _to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def pearson_with_stats(x, y) -> dict:
    """Pearson correlation + p-value between two 1-D arrays (torch or numpy)."""
    x = _to_numpy(x).reshape(-1)
    y = _to_numpy(y).reshape(-1)
    finite = np.isfinite(x) & np.isfinite(y)
    x, y = x[finite], y[finite]
    if len(x) < 3 or np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return {"r": 0.0, "p": 1.0, "n": int(len(x))}
    r, p = pearsonr(x, y)
    return {"r": float(r), "p": float(p), "n": int(len(x))}


def select_nodes_by_type(
    assignment_metadata: list[dict], min_count_per_type: int = 0
) -> dict:
    """
    Group node indices by their true assignment type ("linear", "tanh",
    "interaction"; root nodes are dropped since they have no assignment to
    diagnose).

    Returns: {"linear": [...], "tanh": [...], "interaction": [...]}
    """
    groups = defaultdict(list)
    for node, meta in enumerate(assignment_metadata):
        if meta["type"] == "root":
            continue
        groups[meta["type"]].append(node)

    for type_name, nodes in groups.items():
        if len(nodes) < min_count_per_type:
            raise ValueError(
                f"Only {len(nodes)} '{type_name}' nodes available, "
                f"need at least {min_count_per_type}. Increase num_features."
            )
    return dict(groups)


def summarize_correlations(
    per_node_stats: list[dict], key: str, alpha: float = 0.05
) -> dict:
    """
    Summarize a per-node correlation ("r", "p") across nodes of one type.

    The *signed* mean is reported for reference, but is not the right
    discriminator between node types here: the true cross-coefficient behind
    each node's correlation (e.g. w_i * w_k for an interaction node) has a
    random sign, so signed correlations across different nodes can partially
    cancel even when a systematic effect is present in every single node.
    `mean_abs_r` and `frac_significant` are the statistics that actually
    detect "is there a per-node effect", regardless of its sign.
    """
    r_values = np.array([s[key]["r"] for s in per_node_stats if key in s])
    p_values = np.array([s[key]["p"] for s in per_node_stats if key in s])
    if len(r_values) == 0:
        return {
            "mean_r": None,
            "std_r": None,
            "mean_abs_r": None,
            "frac_significant": None,
            "n_nodes": 0,
        }
    return {
        "mean_r": float(r_values.mean()),
        "std_r": float(r_values.std(ddof=1)) if len(r_values) > 1 else 0.0,
        "mean_abs_r": float(np.abs(r_values).mean()),
        "frac_significant": float((p_values < alpha).mean()),
        "n_nodes": int(len(r_values)),
    }
