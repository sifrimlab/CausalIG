"""
Graph perturbation utilities for the graph-misspecification robustness ablation.

Standalone and independent of any experiment's data structures: operates on a
plain binary adjacency matrix (``numpy.ndarray``, ``A[i, j] != 0`` meaning edge
``i -> j``) plus an optional partition of the graph into disjoint "communities"
(e.g. dyngen's simulated regulatory modules, each with a root gene). Shared
between the synthetic benchmark (``experiments/synthetic.py``) and the dyngen
experiment.

Four perturbation methods:

- ``REMOVE``: drop a percentage of existing edges. Cannot create a cycle.
- ``ADD_DAG``: add a percentage of currently-missing edges, each accepted only
  if it doesn't create a cycle given everything already committed this call.
  DAG-safe by construction.
- ``REVERSE``: flip the direction of a percentage of existing edges, each
  accepted only if it doesn't create a cycle given everything already
  committed this call. DAG-safe by construction.
- ``ADD_CYCLIC``: add a percentage of missing edges with no DAG constraint.
  Kept for parity with prior (non-DAG-safe) usage; out of scope for the CIG
  robustness ablation, which requires the corrupted graph to stay a DAG.

The percentage denominator ("base count") is always the community's
*existing* edge count, for every method — including ADD_DAG/ADD_CYCLIC. So
"50% removed" and "50% added" both mean a number of edges equal to half the
original edge count, and are directly comparable in scale. (An earlier
version of this module used the DAG-safe *candidate pool* — i.e. all missing
edges that don't individually create a cycle — as ADD_DAG's denominator.
For a sparse graph that pool is close to every possible pair, so "50%" of it
could balloon a graph with a handful of edges into one with thousands,
silently changing the ablation from "add a few spurious edges" into "replace
the graph with a near-complete one" — and blew up memory for surrogates
whose cost scales with edge count, like `MaskedGlobalLinear`. Don't
reintroduce that denominator.)

``ADD_DAG`` and ``REVERSE`` may not always reach the requested percentage —
see ``metadata["per_root"][root]["achieved_percentage"]``.
"""

from enum import Enum

import numpy as np


class PerturbationMethod(Enum):
    REMOVE = "remove"
    ADD_DAG = "add_dag"
    ADD_CYCLIC = "add_cyclic"
    REVERSE = "reverse"


# Filename-safe tokens for each method, used by experiment configs/sweep
# scripts to build result/plot filenames that stay unique per (method,
# fraction) pair. REMOVE keeps the "edgedrop" token used before REVERSE/
# ADD_DAG existed, so already-generated result filenames stay valid.
METHOD_FILENAME_TOKENS = {
    PerturbationMethod.REMOVE.value: "edgedrop",
    PerturbationMethod.ADD_DAG.value: "adddag",
    PerturbationMethod.REVERSE.value: "reverse",
    PerturbationMethod.ADD_CYCLIC.value: "addcyclic",
}


def perturbation_filename_suffix(method: str, fraction: float) -> str:
    """
    Build a filename suffix like `_edgedrop0.5` (empty string if `fraction`
    is falsy), shared across experiment scripts so result/plot filenames
    never collide across (method, fraction) pairs.
    """
    if not fraction:
        return ""
    token = METHOD_FILENAME_TOKENS.get(method, method)
    return f"_{token}{fraction}"


# Methods that must never introduce a cycle, as opposed to ADD_CYCLIC which
# deliberately ignores the DAG constraint.
_DAG_SAFE_METHODS = {
    PerturbationMethod.REMOVE,
    PerturbationMethod.ADD_DAG,
    PerturbationMethod.REVERSE,
}


def is_dag(adjacency: np.ndarray) -> bool:
    """
    Check whether a binary adjacency matrix is acyclic, via Kahn's algorithm.

    Args:
        adjacency: (p, p) binary adjacency matrix, ``A[i, j] != 0`` meaning ``i -> j``.

    Returns:
        True iff `adjacency` has no directed cycle.
    """
    adjacency = np.asarray(adjacency)
    n = adjacency.shape[0]
    remaining_in_degree = (adjacency != 0).sum(axis=0).astype(int)
    queue = list(np.flatnonzero(remaining_in_degree == 0))
    visited = 0

    while queue:
        node = queue.pop()
        visited += 1
        for neighbor in np.flatnonzero(adjacency[node]):
            remaining_in_degree[neighbor] -= 1
            if remaining_in_degree[neighbor] == 0:
                queue.append(neighbor)

    return visited == n


def compute_community_descendants(
    adjacency: np.ndarray,
    nodes: list[int],
) -> dict[int, set[int]]:
    """
    Compute descendants for nodes inside an isolated DAG community.

    ``descendants[u]`` contains all nodes reachable from ``u``, restricted to
    edges that stay within ``nodes`` (edges leaving the community are ignored).
    This is a point-in-time snapshot; it is not kept up to date as `adjacency`
    is mutated (see ``_reachable_from`` for the live, per-call version used
    internally to keep multi-edge perturbations cycle-safe).
    """
    node_set = set(nodes)
    descendants = {}

    for source in nodes:
        visited = set()
        stack = [source]

        while stack:
            current = stack.pop()

            for neighbor in np.flatnonzero(adjacency[current]):
                if neighbor not in node_set:
                    continue
                if neighbor not in visited:
                    visited.add(neighbor)
                    stack.append(neighbor)

        visited.discard(source)
        descendants[source] = visited

    return descendants


def _reachable_from(adjacency: np.ndarray, node_set: set, source: int) -> set:
    """Live (uncached) reachable set from `source`, restricted to `node_set`."""
    visited = set()
    stack = [source]
    while stack:
        current = stack.pop()
        for neighbor in np.flatnonzero(adjacency[current]):
            if neighbor not in node_set:
                continue
            if neighbor not in visited:
                visited.add(neighbor)
                stack.append(neighbor)
    return visited


def _community_edges(
    adjacency: np.ndarray, nodes: list[int]
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    """Existing and missing ordered node pairs (u != v) within a community."""
    node_arr = np.asarray(nodes)
    sub = adjacency[np.ix_(node_arr, node_arr)]
    off_diag = ~np.eye(len(nodes), dtype=bool)

    existing_local = np.argwhere((sub != 0) & off_diag)
    missing_local = np.argwhere((sub == 0) & off_diag)

    existing_edges = [(int(node_arr[i]), int(node_arr[j])) for i, j in existing_local]
    missing_edges = [(int(node_arr[i]), int(node_arr[j])) for i, j in missing_local]
    return existing_edges, missing_edges


def _perturb_community(
    adjacency: np.ndarray,
    nodes: list[int],
    method: PerturbationMethod,
    percentage: float,
    rng: np.random.Generator,
) -> dict:
    """
    Apply one perturbation method to one community, in place on `adjacency`.

    The percentage denominator ("base count") is always the community's
    existing edge count, for every method — see the module docstring for why
    ADD_DAG/ADD_CYCLIC must not use their (much larger) candidate-pool size
    instead.
    """
    node_set = set(nodes)
    existing_edges, missing_edges = _community_edges(adjacency, nodes)
    base_count = len(existing_edges)
    target = round(base_count * percentage / 100)
    num_skipped = 0

    if method == PerturbationMethod.REMOVE:
        candidates = existing_edges
        rng.shuffle(candidates)
        num_actual = min(target, len(candidates))
        for u, v in candidates[:num_actual]:
            adjacency[u, v] = 0
        num_candidates_available = len(candidates)

    elif method == PerturbationMethod.ADD_CYCLIC:
        candidates = missing_edges
        rng.shuffle(candidates)
        num_actual = min(target, len(candidates))
        for u, v in candidates[:num_actual]:
            adjacency[u, v] = 1
        num_candidates_available = len(candidates)

    elif method == PerturbationMethod.ADD_DAG:
        # Pre-filter to edges that are individually DAG-safe against the
        # pre-perturbation graph (mirrors `compute_community_descendants`).
        # Committing is still done incrementally with a *live* re-check (not
        # just this static filter), since adding one valid_edges candidate
        # can make a later one unsafe — e.g. an edge-free community where
        # both (u, v) and (v, u) are individually valid, but adding both at
        # once creates a 2-cycle.
        descendants = compute_community_descendants(adjacency, nodes)
        candidates = [(u, v) for (u, v) in missing_edges if u not in descendants[v]]
        rng.shuffle(candidates)
        num_actual = 0
        for u, v in candidates:
            if num_actual >= target:
                break
            if u in _reachable_from(adjacency, node_set, v):
                num_skipped += 1
                continue
            adjacency[u, v] = 1
            num_actual += 1
        num_candidates_available = len(candidates)

    elif method == PerturbationMethod.REVERSE:
        candidates = list(existing_edges)
        rng.shuffle(candidates)
        num_actual = 0
        for u, v in candidates:
            if num_actual >= target:
                break
            adjacency[u, v] = 0  # tentatively remove
            # Reversing u->v to v->u creates a cycle iff u can still reach v
            # via some other path once the direct edge is gone (e.g. a
            # "shortcut" edge u->v alongside a longer u->w->v path: reversing
            # the shortcut to v->u closes a u->w->v->u loop). Checked live
            # against the graph as mutated so far this call, since an earlier
            # reversal in this same loop can change reachability for later
            # candidates.
            if v in _reachable_from(adjacency, node_set, u):
                adjacency[u, v] = 1  # not safe, put it back unreversed
                num_skipped += 1
                continue
            adjacency[v, u] = 1  # commit the reversal
            num_actual += 1
        num_candidates_available = len(candidates)

    else:
        raise ValueError(f"Unknown perturbation method: {method!r}")

    achieved_percentage = (100.0 * num_actual / base_count) if base_count else 0.0

    return {
        "num_base_edges": base_count,
        "num_target_changes": target,
        "num_actual_changes": num_actual,
        "achieved_percentage": achieved_percentage,
        "num_candidates_available": num_candidates_available,
        "num_skipped_infeasible": num_skipped,
    }


def perturb_graph(
    adjacency: np.ndarray,
    communities: dict[int, list[int]] | None = None,
    roots_to_perturb: list[int] | None = None,
    method: PerturbationMethod = PerturbationMethod.REMOVE,
    percentage: float = 0,
    seed: int | None = None,
) -> tuple[np.ndarray, dict]:
    """
    Perturb one or more DAG communities within `adjacency`.

    Args:
        adjacency: (p, p) binary directed adjacency matrix, `A[i, j] != 0`
            meaning `i -> j`. Must be acyclic.
        communities: Mapping community root -> list of member node indices.
            If `None`, the whole graph is treated as a single community
            (root `0`) — the natural default for experiments (like the
            synthetic benchmark) that don't partition their DAG into
            communities.
        roots_to_perturb: Which community roots to modify. If `None`, every
            community in `communities` is perturbed.
        method: One of `PerturbationMethod.{REMOVE, ADD_DAG, ADD_CYCLIC, REVERSE}`.
        percentage: Percentage (0-100) of each community's eligible edges to
            perturb. "Eligible" is method-specific — see the module docstring.
        seed: Random seed for reproducible perturbation.

    Returns:
        `(perturbed_adjacency, metadata)`. `metadata["per_root"][root]`
        reports, per perturbed community: base edge count, target vs. actual
        changes, achieved percentage, and (for ADD_DAG/REVERSE) how many
        candidates were skipped as cycle-unsafe. `metadata` also aggregates
        these across all perturbed roots.
    """
    if not 0 <= percentage <= 100:
        raise ValueError(f"percentage must be in [0, 100], got {percentage}")

    adjacency = np.asarray(adjacency)
    if not is_dag(adjacency):
        raise ValueError("adjacency must be a DAG (acyclic) before perturbation.")

    if communities is None:
        communities = {0: list(range(adjacency.shape[0]))}
    if roots_to_perturb is None:
        roots_to_perturb = list(communities.keys())

    rng = np.random.default_rng(seed)
    perturbed = adjacency.copy()

    per_root = {}
    for root in roots_to_perturb:
        if root not in communities:
            raise ValueError(f"Community root {root} not found")
        per_root[root] = _perturb_community(
            perturbed, communities[root], method, percentage, rng
        )

    if method in _DAG_SAFE_METHODS:
        assert is_dag(perturbed), (
            f"perturb_graph produced a cyclic graph for method={method!r} — "
            "this should never happen; please report this as a bug."
        )

    num_base_total = sum(m["num_base_edges"] for m in per_root.values())
    num_actual_total = sum(m["num_actual_changes"] for m in per_root.values())

    metadata = {
        "method": method.value,
        "requested_percentage": percentage,
        "seed": seed,
        "roots_perturbed": list(roots_to_perturb),
        "per_root": per_root,
        "num_base_edges_total": num_base_total,
        "num_actual_changes_total": num_actual_total,
        "achieved_percentage_total": (
            100.0 * num_actual_total / num_base_total if num_base_total else 0.0
        ),
    }

    return perturbed, metadata
