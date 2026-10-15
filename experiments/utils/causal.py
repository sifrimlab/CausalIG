"""
Causal feature processing utilities.
"""

import numpy as np
import torch

from causalig.utils.graph import find_ancestors, find_eldest_ancestors


def process_causal_features(
    causal_indices: set[int] | np.ndarray | list,
    dag: np.ndarray,
    include_ancestors: bool = False,
    only_eldest: bool = False,
) -> set[int]:
    """
    Process causal feature indices based on DAG structure.

    Args:
        causal_indices: Initial set of causal feature indices
        dag: DAG adjacency matrix (p x p) where dag[i, j] != 0 means i -> j
        include_ancestors: If True, include all ancestors of causal features
        only_eldest: If True, keep only eldest ancestors (no parents in causal set)

    Returns:
        Processed set of causal feature indices
    """
    # Convert to set for easier manipulation
    if isinstance(causal_indices, (np.ndarray, list)):
        causal_set = set(causal_indices)
    else:
        causal_set = set(causal_indices)

    if isinstance(dag, torch.Tensor):
        dag = dag.cpu().numpy()

    # Include ancestors if requested
    if include_ancestors:
        ancestor_indices = find_ancestors(causal_set, dag)
        causal_set = causal_set.union(ancestor_indices)

    # Filter to only eldest ancestors if requested
    if only_eldest:
        causal_set = find_eldest_ancestors(causal_set, dag)

    return causal_set
