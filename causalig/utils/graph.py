"""
Graph utilities for working with directed acyclic graphs (DAGs).
"""

from collections import deque

import numpy as np
import torch


def find_ancestors(causal_nodes, adjacency_matrix):
    """
    Compute all ancestor nodes (direct AND indirect parents)
    of a set of nodes in a directed acyclic graph.

    Parameters
    ----------
    causal_nodes : iterable of ints
        The nodes whose ancestors we want.
    adjacency_matrix : 2D array-like (p x p)
        Adjacency matrix where adjacency_matrix[i, j] != 0 means i -> j.

    Returns
    -------
    ancestors : set of ints
        All ancestors of the causal nodes.
    """
    adjacency_matrix = np.asarray(adjacency_matrix)
    p = adjacency_matrix.shape[0]

    # precompute parent lists for efficiency
    parents_of = {j: np.where(adjacency_matrix[:, j] != 0)[0] for j in range(p)}

    ancestors = set()

    def dfs(node):
        for parent in parents_of[node]:
            if parent not in ancestors:
                ancestors.add(parent)
                dfs(parent)

    for c in causal_nodes:
        dfs(c)

    return ancestors


def find_eldest_ancestors(causal_nodes, adjacency_matrix):
    """
    Find the eldest ancestors among the causal nodes.
    An eldest ancestor is a causal node that has no parents
    within the set of causal nodes.

    Parameters
    ----------
    causal_nodes : iterable of ints
        The nodes considered as causal.
    adjacency_matrix : 2D array-like (p x p)
        Adjacency matrix where adjacency_matrix[i, j] != 0 means i -> j.

    Returns
    -------
    eldest_ancestors : set of ints
        The eldest ancestors among the causal nodes.
    """
    adjacency_matrix = np.asarray(adjacency_matrix)
    causal_set = set(causal_nodes)

    eldest_ancestors = set()

    for node in causal_nodes:
        parents = np.where(adjacency_matrix[:, node] != 0)[0]
        if not any(parent in causal_set for parent in parents):
            eldest_ancestors.add(node)

    return eldest_ancestors


def get_connected_groups(adjacency_matrix):
    """
    BFS on a directed graph (adjacency matrix, rows=sources, cols=targets).
    Returns a list of groups, where each group is a list of node indices
    reachable from a root (in-degree 0, out-degree > 0).
    Isolated or cyclic nodes are appended as separate singleton groups.
    """
    n = adjacency_matrix.shape[0]
    col_sums = adjacency_matrix.sum(axis=1)
    row_sums = adjacency_matrix.sum(axis=0)
    start_indices = np.where((row_sums == 0) & (col_sums > 0))[0]

    all_groups = []
    visited = np.zeros(n, dtype=bool)

    for root_idx in start_indices:
        if visited[root_idx]:
            continue
        queue = deque([root_idx])
        group = []
        while queue:
            idx = queue.popleft()
            if visited[idx]:
                continue
            visited[idx] = True
            group.append(idx)
            for nbr in np.where(adjacency_matrix[idx, :] != 0)[0]:
                if not visited[nbr]:
                    queue.append(nbr)
        all_groups.append(group)

    for i in range(n):
        if not visited[i]:
            queue = deque([i])
            group = []
            while queue:
                idx = queue.popleft()
                if visited[idx]:
                    continue
                visited[idx] = True
                group.append(idx)
                for nbr in np.where(adjacency_matrix[idx, :] != 0)[0]:
                    if not visited[nbr]:
                        queue.append(nbr)
            all_groups.append(group)

    return all_groups


def propagate_upstream(
    attributions: torch.Tensor,
    adjacency_matrix: torch.Tensor,
    n_hops: int = 1,
    push_fraction: float = 1.0,
) -> torch.Tensor:
    """
    Redistribute feature attributions upstream along a causal DAG.

    Model- and method-agnostic post-processing step: takes attributions
    already produced by any attribution method (IG, SHAP, ...) and pushes
    them toward upstream ancestors using only the graph topology, with no
    knowledge of the model or data. Each round, every feature sends
    `push_fraction` of its currently held mass to its parents (split evenly
    across them) and keeps the rest; a feature with no parents (a root) has
    nowhere to send to, so it keeps its mass regardless of `push_fraction`.
    Total attribution mass is conserved exactly across rounds.

    Parameters
    ----------
    attributions : (d,) or (batch, d) tensor
        Per-feature attributions from any attribution method.
    adjacency_matrix : (d, d) tensor/array
        Adjacency matrix where `adjacency_matrix[i, j] != 0` means `i -> j`.
    n_hops : int
        Number of propagation rounds. `n_hops=0` returns `attributions`
        unchanged.
    push_fraction : float
        Fraction, in `[0, 1]`, of a node's mass sent to its parents each
        round. The rest is retained at that node.

    Returns
    -------
    Tensor of the same shape as `attributions`.
    """
    if not 0.0 <= push_fraction <= 1.0:
        raise ValueError(f"push_fraction must be in [0, 1], got {push_fraction}")

    was_1d = attributions.dim() == 1
    r = attributions.unsqueeze(0) if was_1d else attributions

    A = adjacency_matrix
    if not isinstance(A, torch.Tensor):
        A = torch.as_tensor(A)
    A = A.to(dtype=r.dtype, device=r.device)

    parent_count = A.sum(dim=0)  # number of parents per node, shape (d,)
    has_parents = parent_count > 0
    safe_parent_count = parent_count.clamp_min(1.0)

    # send_weights[i, j] = share of j's pushed mass that flows to parent i.
    send_weights = push_fraction * A / safe_parent_count
    retain_frac = torch.where(
        has_parents, 1.0 - push_fraction, torch.ones_like(parent_count)
    )

    for _ in range(n_hops):
        message = r @ send_weights.T
        r = r * retain_frac + message

    return r.squeeze(0) if was_1d else r
