"""Unit tests for CausalIG graph and causal utilities.

Covers ``causalig.utils.graph`` (``find_ancestors``, ``find_eldest_ancestors``,
``get_connected_groups``) and ``experiments.utils.causal``
(``process_causal_features``, ``causal_indices_to_mask``).

Convention used throughout: adjacency matrices are ``p x p`` with
``A[i, j] != 0`` meaning there is a directed edge ``i -> j`` (rows are
sources, columns are targets). Ancestors of a node ``n`` are read from
column ``n``.
"""

import numpy as np
import pytest
import torch

from causalig.utils.graph import (
    find_ancestors,
    find_eldest_ancestors,
    get_connected_groups,
    propagate_upstream,
)
from experiments.utils.causal import (
    process_causal_features,
)

# ---------------------------------------------------------------------------
# Fixtures / helpers: small, explicit DAGs with known structure.
# ---------------------------------------------------------------------------


def chain_dag():
    """Chain 0 -> 1 -> 2 -> 3."""
    A = np.zeros((4, 4), dtype=int)
    A[0, 1] = 1
    A[1, 2] = 1
    A[2, 3] = 1
    return A


def diamond_dag():
    """Diamond / branching DAG.

    Edges: 0 -> 1, 0 -> 2, 1 -> 3, 2 -> 3.
    Node 4 is isolated (no edges).
    """
    A = np.zeros((5, 5), dtype=int)
    A[0, 1] = 1
    A[0, 2] = 1
    A[1, 3] = 1
    A[2, 3] = 1
    return A


# ---------------------------------------------------------------------------
# find_ancestors
# ---------------------------------------------------------------------------


def test_find_ancestors_chain_leaf():
    A = chain_dag()
    # Ancestors of node 3 in the chain are 0, 1, 2.
    assert find_ancestors([3], A) == {0, 1, 2}


def test_find_ancestors_chain_middle():
    A = chain_dag()
    assert find_ancestors([2], A) == {0, 1}
    assert find_ancestors([1], A) == {0}


def test_find_ancestors_root_has_none():
    A = chain_dag()
    # Node 0 is a root: no ancestors. Result excludes the node itself.
    assert find_ancestors([0], A) == set()


def test_find_ancestors_diamond():
    A = diamond_dag()
    # Node 3 is reachable from 0 via both 1 and 2.
    assert find_ancestors([3], A) == {0, 1, 2}
    assert find_ancestors([1], A) == {0}
    assert find_ancestors([2], A) == {0}


def test_find_ancestors_multiple_causal_nodes_union():
    A = diamond_dag()
    # Union of ancestors of nodes 1 and 2 is just {0}.
    assert find_ancestors([1, 2], A) == {0}
    # Ancestors of {3} plus {1}: still {0, 1, 2}.
    assert find_ancestors([3, 1], A) == {0, 1, 2}


def test_find_ancestors_isolated_node():
    A = diamond_dag()
    # Node 4 is isolated; it has no ancestors.
    assert find_ancestors([4], A) == set()


def test_find_ancestors_accepts_torch_via_asarray():
    # The function calls np.asarray(A); a torch tensor on CPU converts fine.
    A = torch.from_numpy(chain_dag())
    assert find_ancestors([3], A) == {0, 1, 2}


# ---------------------------------------------------------------------------
# find_eldest_ancestors
# ---------------------------------------------------------------------------


def test_find_eldest_ancestors_chain_full_set():
    A = chain_dag()
    # Whole chain as causal set: only node 0 has no parent within the set.
    assert find_eldest_ancestors([0, 1, 2, 3], A) == {0}


def test_find_eldest_ancestors_gap_in_set():
    A = chain_dag()
    # Causal set {2, 3}: node 2's parent (1) is NOT in the set, so 2 is eldest.
    # Node 3's parent (2) IS in the set, so 3 is not eldest.
    assert find_eldest_ancestors([2, 3], A) == {2}


def test_find_eldest_ancestors_diamond():
    A = diamond_dag()
    # Full causal set {0,1,2,3}: only 0 has no parent inside the set.
    assert find_eldest_ancestors([0, 1, 2, 3], A) == {0}
    # Causal set {1, 2, 3}: 1 and 2 have parent 0 (not in set) -> eldest.
    # 3 has parents 1,2 which ARE in set -> not eldest.
    assert find_eldest_ancestors([1, 2, 3], A) == {1, 2}


def test_find_eldest_ancestors_all_roots_when_disconnected():
    A = diamond_dag()
    # Causal set {1, 4}: 1's parent 0 not in set, 4 has no parents -> both eldest.
    assert find_eldest_ancestors([1, 4], A) == {1, 4}


def test_find_eldest_ancestors_single_node():
    A = chain_dag()
    # A single causal node always qualifies as eldest (no parent in the set).
    assert find_eldest_ancestors([3], A) == {3}


# ---------------------------------------------------------------------------
# get_connected_groups
# ---------------------------------------------------------------------------


def test_get_connected_groups_chain():
    A = chain_dag()
    groups = get_connected_groups(A)
    # Single root (node 0, in-deg 0, out-deg > 0) reaches the whole chain.
    assert len(groups) == 1
    assert set(groups[0]) == {0, 1, 2, 3}
    # BFS from root gives a topological-ish order starting at the root.
    assert groups[0][0] == 0


# ---------------------------------------------------------------------------
# propagate_upstream
# ---------------------------------------------------------------------------


def test_propagate_upstream_zero_hops_is_identity():
    A = torch.from_numpy(chain_dag()).float()
    attr = torch.tensor([1.0, 2.0, 3.0, 4.0])
    result = propagate_upstream(attr, A, n_hops=0, push_fraction=1.0)
    assert torch.allclose(result, attr)


def test_propagate_upstream_zero_push_fraction_is_identity():
    A = torch.from_numpy(chain_dag()).float()
    attr = torch.tensor([1.0, 2.0, 3.0, 4.0])
    result = propagate_upstream(attr, A, n_hops=3, push_fraction=0.0)
    assert torch.allclose(result, attr)


def test_propagate_upstream_chain_single_hop_full_push():
    # Chain 0 -> 1 -> 2 -> 3, full push moves each node's mass entirely onto
    # its immediate parent; the root (0) has nowhere to send and keeps its own.
    A = torch.from_numpy(chain_dag()).float()
    attr = torch.tensor([1.0, 2.0, 3.0, 4.0])
    result = propagate_upstream(attr, A, n_hops=1, push_fraction=1.0)
    assert torch.allclose(result, torch.tensor([1.0 + 2.0, 3.0, 4.0, 0.0]))


def test_propagate_upstream_chain_two_hops_reaches_root():
    A = torch.from_numpy(chain_dag()).float()
    attr = torch.tensor([1.0, 2.0, 3.0, 4.0])
    result = propagate_upstream(attr, A, n_hops=3, push_fraction=1.0)
    # Full push, enough hops: everything ends up on the root (node 0).
    assert torch.allclose(result, torch.tensor([10.0, 0.0, 0.0, 0.0]))


def test_propagate_upstream_diamond_splits_evenly_across_parents():
    # Diamond: 0 -> 1, 0 -> 2, 1 -> 3, 2 -> 3. Node 3 has two parents (1, 2),
    # so a full push from 3 splits its mass evenly between them.
    A = torch.from_numpy(diamond_dag()).float()
    attr = torch.tensor([0.0, 0.0, 0.0, 8.0, 0.0])
    result = propagate_upstream(attr, A, n_hops=1, push_fraction=1.0)
    assert torch.allclose(result, torch.tensor([0.0, 4.0, 4.0, 0.0, 0.0]))


def test_propagate_upstream_isolated_node_untouched():
    A = torch.from_numpy(diamond_dag()).float()
    attr = torch.tensor([1.0, 1.0, 1.0, 1.0, 5.0])
    result = propagate_upstream(attr, A, n_hops=2, push_fraction=1.0)
    # Node 4 is isolated: no parents, no children, so its mass never moves.
    assert result[4] == 5.0


def test_propagate_upstream_conserves_total_mass():
    A = torch.from_numpy(diamond_dag()).float()
    attr = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    for push_fraction in (0.0, 0.3, 1.0):
        for n_hops in (0, 1, 2, 5):
            result = propagate_upstream(
                attr, A, n_hops=n_hops, push_fraction=push_fraction
            )
            assert result.sum().item() == pytest.approx(attr.sum().item())


def test_propagate_upstream_batched_input_preserves_shape():
    A = torch.from_numpy(chain_dag()).float()
    attr = torch.tensor([[1.0, 2.0, 3.0, 4.0], [4.0, 3.0, 2.0, 1.0]])
    result = propagate_upstream(attr, A, n_hops=1, push_fraction=0.5)
    assert result.shape == attr.shape
    assert torch.allclose(result.sum(dim=1), attr.sum(dim=1))


def test_propagate_upstream_partial_push_matches_manual_computation():
    # Chain 0 -> 1 -> 2 -> 3, push_fraction=0.5, one hop.
    A = torch.from_numpy(chain_dag()).float()
    attr = torch.tensor([0.0, 0.0, 0.0, 4.0])
    result = propagate_upstream(attr, A, n_hops=1, push_fraction=0.5)
    # Node 3 sends 0.5*4=2 to its only parent (2), keeps the other 2.
    assert torch.allclose(result, torch.tensor([0.0, 0.0, 2.0, 2.0]))


def test_propagate_upstream_invalid_push_fraction_raises():
    A = torch.from_numpy(chain_dag()).float()
    attr = torch.tensor([1.0, 2.0, 3.0, 4.0])
    with pytest.raises(ValueError):
        propagate_upstream(attr, A, n_hops=1, push_fraction=1.5)
    with pytest.raises(ValueError):
        propagate_upstream(attr, A, n_hops=1, push_fraction=-0.1)


def test_get_connected_groups_diamond_with_isolated():
    A = diamond_dag()
    groups = get_connected_groups(A)
    # Groups are (in-order): the BFS group rooted at 0, then the isolated
    # node 4 appended as its own singleton by the unvisited-node pass.
    reachable = {frozenset(g) for g in groups}
    assert frozenset({0, 1, 2, 3}) in reachable
    assert frozenset({4}) in reachable
    assert len(groups) == 2


def test_get_connected_groups_two_components():
    # Two disjoint chains: 0->1 and 2->3.
    A = np.zeros((4, 4), dtype=int)
    A[0, 1] = 1
    A[2, 3] = 1
    groups = get_connected_groups(A)
    comps = {frozenset(g) for g in groups}
    assert comps == {frozenset({0, 1}), frozenset({2, 3})}


def test_get_connected_groups_all_isolated():
    # No edges at all: every node is isolated. No node has out-degree > 0,
    # so all are appended as singletons by the second pass.
    A = np.zeros((3, 3), dtype=int)
    groups = get_connected_groups(A)
    comps = {frozenset(g) for g in groups}
    assert comps == {frozenset({0}), frozenset({1}), frozenset({2})}


# ---------------------------------------------------------------------------
# process_causal_features
# ---------------------------------------------------------------------------


def test_process_causal_features_no_options():
    A = chain_dag()
    # With neither flag set, the input set is returned unchanged.
    assert process_causal_features({2}, A) == {2}
    assert process_causal_features([2, 3], A) == {2, 3}


def test_process_causal_features_include_ancestors():
    A = chain_dag()
    # {2} plus all its ancestors {0, 1} -> {0, 1, 2}.
    assert process_causal_features({2}, A, include_ancestors=True) == {0, 1, 2}


def test_process_causal_features_include_ancestors_diamond():
    A = diamond_dag()
    assert process_causal_features({3}, A, include_ancestors=True) == {0, 1, 2, 3}


def test_process_causal_features_only_eldest():
    A = chain_dag()
    # only_eldest filters the given set (no ancestors added): {2,3} -> {2}.
    assert process_causal_features({2, 3}, A, only_eldest=True) == {2}


def test_process_causal_features_include_ancestors_and_only_eldest():
    A = chain_dag()
    # {2} -> add ancestors -> {0, 1, 2} -> keep eldest -> {0}.
    result = process_causal_features({2}, A, include_ancestors=True, only_eldest=True)
    assert result == {0}


def test_process_causal_features_empty_set():
    A = chain_dag()
    assert process_causal_features(set(), A) == set()
    assert process_causal_features(set(), A, include_ancestors=True) == set()
    assert process_causal_features(set(), A, only_eldest=True) == set()


def test_process_causal_features_accepts_torch_dag():
    # dag passed as a torch tensor is converted to numpy inside the function.
    A = torch.from_numpy(chain_dag())
    assert process_causal_features({2}, A, include_ancestors=True) == {0, 1, 2}


def test_process_causal_features_accepts_list_and_ndarray():
    A = chain_dag()
    assert process_causal_features([2, 3], A) == {2, 3}
    assert process_causal_features(np.array([2, 3]), A) == {2, 3}
