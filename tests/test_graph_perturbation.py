"""
Tests for `experiments.utils.graph_perturbation`.

Covers the DAG-safety invariant (the whole point of ADD_DAG/REVERSE) across
every method, percentage, and a large sweep of seeds, plus targeted checks
for the metadata each method reports and the multi-edge cycle-safety bug
that a naive static-snapshot implementation of ADD_DAG/REVERSE would hit.
"""

import numpy as np
import pytest

from experiments.utils.graph_perturbation import (
    PerturbationMethod,
    compute_community_descendants,
    is_dag,
    perturb_graph,
)

SEEDS = list(range(50))
PERCENTAGES = [0, 25, 50, 75]
DAG_SAFE_METHODS = [
    PerturbationMethod.REMOVE,
    PerturbationMethod.ADD_DAG,
    PerturbationMethod.REVERSE,
]


def chain_dag(n=6):
    """Chain 0 -> 1 -> ... -> n-1."""
    A = np.zeros((n, n), dtype=int)
    for i in range(n - 1):
        A[i, i + 1] = 1
    return A


def diamond_dag():
    """Diamond / branching DAG. Edges: 0->1, 0->2, 1->3, 2->3. Node 4 isolated."""
    A = np.zeros((5, 5), dtype=int)
    A[0, 1] = 1
    A[0, 2] = 1
    A[1, 3] = 1
    A[2, 3] = 1
    return A


def dense_random_dag(n=20, edge_prob=0.3, seed=0):
    """A denser random DAG: edges only from lower to higher index (in a random
    permutation), so it's acyclic by construction, but dense enough to
    exercise the ADD_DAG/REVERSE cycle-skip branches."""
    rng = np.random.default_rng(seed)
    order = rng.permutation(n)
    A = np.zeros((n, n), dtype=int)
    for i in range(n):
        for j in range(i + 1, n):
            if rng.random() < edge_prob:
                A[order[i], order[j]] = 1
    return A


TEST_GRAPHS = {
    "chain": chain_dag(),
    "diamond": diamond_dag(),
    "dense_random": dense_random_dag(),
}


@pytest.mark.parametrize("graph_name", list(TEST_GRAPHS.keys()))
@pytest.mark.parametrize("method", DAG_SAFE_METHODS)
@pytest.mark.parametrize("percentage", PERCENTAGES)
def test_dag_invariant_holds_across_seeds(graph_name, method, percentage):
    """The core requirement: DAG-safe methods must never produce a cycle,
    across every method x percentage combination and ~50 seeds, with no
    exceptions raised."""
    adjacency = TEST_GRAPHS[graph_name]

    for seed in SEEDS:
        perturbed, metadata = perturb_graph(
            adjacency, method=method, percentage=percentage, seed=seed
        )
        assert is_dag(
            perturbed
        ), f"{graph_name}/{method}/{percentage}%/seed={seed} produced a cyclic graph"
        # Actual changes can't exceed the (rounded) target — rounding can
        # push the *achieved* percentage slightly above the requested one
        # (e.g. round(0.75 * 5) = 4 -> 80% achieved for a 75% request), so we
        # check against the target count, not the raw requested percentage.
        for root_meta in metadata["per_root"].values():
            assert root_meta["num_actual_changes"] <= root_meta["num_target_changes"]
            assert root_meta["num_actual_changes"] <= root_meta["num_base_edges"]


def test_remove_never_skips():
    """Dropping edges carries no cycle risk, so it should always hit the exact target."""
    adjacency = dense_random_dag()
    for seed in SEEDS[:10]:
        for percentage in PERCENTAGES:
            _, metadata = perturb_graph(
                adjacency,
                method=PerturbationMethod.REMOVE,
                percentage=percentage,
                seed=seed,
            )
            root_meta = metadata["per_root"][0]
            assert root_meta["num_skipped_infeasible"] == 0
            assert root_meta["num_actual_changes"] == root_meta["num_target_changes"]


def test_add_dag_multi_edge_batch_stays_acyclic():
    """Regression test for the static-snapshot bug: a small "seed" chain
    (giving ADD_DAG a nonzero, existing-edge-based target) attached to a
    large edge-free zone (giving it many candidate pairs that are
    individually valid but mutually exclusive — both directions of the same
    pair — unless committed incrementally) must never add both (u, v) and
    (v, u) simultaneously."""
    n = 10
    adjacency = np.zeros((n, n), dtype=int)
    adjacency[0, 1] = 1
    adjacency[1, 2] = 1
    adjacency[2, 3] = 1  # 3 existing edges -> target = 3 additions at 100%
    # nodes 4..9 are left edge-free, providing the ambiguous candidate pool.
    for seed in range(50):
        perturbed, metadata = perturb_graph(
            adjacency,
            method=PerturbationMethod.ADD_DAG,
            percentage=100,
            seed=seed,
        )
        assert is_dag(perturbed)
        # No pair should have edges in both directions.
        assert not np.any((perturbed == 1) & (perturbed.T == 1))
        assert metadata["per_root"][0]["num_actual_changes"] > 0


def test_reverse_reports_partial_achievement_when_infeasible():
    """On a dense graph, reversing a large fraction of edges should often be
    infeasible for all of them; achieved_percentage must reflect that rather
    than silently padding with something else."""
    adjacency = dense_random_dag(n=15, edge_prob=0.5, seed=1)
    _, metadata = perturb_graph(
        adjacency, method=PerturbationMethod.REVERSE, percentage=75, seed=2
    )
    root_meta = metadata["per_root"][0]
    assert root_meta["num_actual_changes"] <= root_meta["num_target_changes"]
    if root_meta["num_actual_changes"] < root_meta["num_target_changes"]:
        assert root_meta["achieved_percentage"] < 75
        assert root_meta["num_skipped_infeasible"] > 0


def test_reverse_actually_flips_edges():
    adjacency = chain_dag(4)  # 0->1->2->3, no branching -> reversal always safe
    perturbed, metadata = perturb_graph(
        adjacency, method=PerturbationMethod.REVERSE, percentage=100, seed=0
    )
    assert metadata["per_root"][0]["num_actual_changes"] == 3
    assert perturbed[1, 0] == 1 and perturbed[0, 1] == 0
    assert perturbed[2, 1] == 1 and perturbed[1, 2] == 0
    assert perturbed[3, 2] == 1 and perturbed[2, 3] == 0


def test_percentage_zero_is_a_no_op():
    adjacency = dense_random_dag()
    for method in DAG_SAFE_METHODS:
        perturbed, metadata = perturb_graph(
            adjacency, method=method, percentage=0, seed=0
        )
        assert np.array_equal(perturbed, adjacency)
        assert metadata["num_actual_changes_total"] == 0


def test_add_cyclic_is_not_dag_safe_by_design():
    """ADD_CYCLIC deliberately ignores the DAG constraint; unlike the other
    three methods it is not asserted to stay acyclic, and on a dense enough
    graph it can in fact produce a cycle."""
    adjacency = np.zeros((3, 3), dtype=int)
    adjacency[0, 1] = 1
    perturbed, metadata = perturb_graph(
        adjacency, method=PerturbationMethod.ADD_CYCLIC, percentage=100, seed=0
    )
    assert metadata["per_root"][0]["num_actual_changes"] > 0
    # No assertion that `perturbed` is a DAG here — that's the point of this method.


def test_communities_restrict_perturbation_to_selected_root():
    """Two disjoint communities; perturbing one must leave the other untouched."""
    adjacency = np.zeros((6, 6), dtype=int)
    adjacency[0, 1] = 1
    adjacency[1, 2] = 1
    adjacency[3, 4] = 1
    adjacency[4, 5] = 1
    communities = {0: [0, 1, 2], 3: [3, 4, 5]}

    perturbed, metadata = perturb_graph(
        adjacency,
        communities=communities,
        roots_to_perturb=[0],
        method=PerturbationMethod.REMOVE,
        percentage=100,
        seed=0,
    )

    assert metadata["roots_perturbed"] == [0]
    assert perturbed[0, 1] == 0 and perturbed[1, 2] == 0  # community 0 edges dropped
    assert perturbed[3, 4] == 1 and perturbed[4, 5] == 1  # community 3 untouched


def test_default_communities_treats_whole_graph_as_one():
    adjacency = chain_dag(4)
    _, metadata = perturb_graph(
        adjacency, method=PerturbationMethod.REMOVE, percentage=100, seed=0
    )
    assert metadata["roots_perturbed"] == [0]
    assert metadata["per_root"][0]["num_base_edges"] == 3


def test_rejects_non_dag_input():
    cyclic = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]])
    with pytest.raises(ValueError):
        perturb_graph(cyclic, method=PerturbationMethod.REMOVE, percentage=10)


def test_rejects_invalid_percentage():
    adjacency = chain_dag()
    with pytest.raises(ValueError):
        perturb_graph(adjacency, percentage=101)
    with pytest.raises(ValueError):
        perturb_graph(adjacency, percentage=-1)


def test_rejects_unknown_root():
    adjacency = chain_dag()
    with pytest.raises(ValueError):
        perturb_graph(
            adjacency,
            communities={0: [0, 1, 2, 3, 4, 5]},
            roots_to_perturb=[99],
        )


def test_is_dag_matches_expectations():
    assert is_dag(chain_dag())
    assert is_dag(diamond_dag())
    assert not is_dag(np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]]))
    assert is_dag(np.zeros((4, 4), dtype=int))  # edgeless is trivially acyclic


def test_compute_community_descendants_matches_reachability():
    A = diamond_dag()
    descendants = compute_community_descendants(A, [0, 1, 2, 3, 4])
    assert descendants[0] == {1, 2, 3}
    assert descendants[1] == {3}
    assert descendants[4] == set()
