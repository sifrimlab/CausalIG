import numpy as np
import torch

from causalig.models.scm import DifferentiableSCM
from experiments.data.assignments import generate_random_assignments_torch_and_numpy
from experiments.data.synthetic import compute_cate, generate_synthetic_data


def test_generate_random_assignments():
    # Simple DAG with 3 nodes: 0 -> 1 -> 2
    dag = np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]])

    assignments_torch, assignments_numpy = generate_random_assignments_torch_and_numpy(
        dag
    )

    assert len(assignments_torch) == 3
    assert assignments_torch[0] is None  # Node 0 has no parents
    assert callable(assignments_torch[1])  # Node 1 has parent 0
    assert callable(assignments_torch[2])  # Node 2 has parent 1

    assert len(assignments_numpy) == 3
    assert assignments_numpy[0] is None  # Node 0 has no parents
    assert callable(assignments_numpy[1])  # Node 1 has parent 0
    assert callable(assignments_numpy[2])  # Node 2 has parent 1


def test_generate_synthetic_data():
    num_features = 5
    num_causal = 3
    num_samples = 100

    features, target, dag, causal_indices, scm = generate_synthetic_data(
        num_features, num_causal, num_samples
    )

    assert features.shape == (num_samples, num_features)
    assert target.shape == (num_samples,)
    assert dag.shape == (num_features, num_features)
    # Check that the DAG is ordered (no edges from later to earlier nodes)
    for i in range(num_features):
        for j in range(i):
            assert dag[i, j] == 0

    assert len(causal_indices) == num_causal


def test_differentiable_scm():
    adjacency = np.array([[0, 1, 1], [0, 0, 1], [0, 0, 0]])
    assignments_torch = [
        None,
        lambda Xp: Xp[:, 0] * 2,  # Node 1 = 2 * Node 0
        lambda Xp: Xp[:, 0] + Xp[:, 1],  # Node 2 = Node 0 + Node 1
    ]
    scm = DifferentiableSCM(
        adjacency_matrix=adjacency,
        assignments=assignments_torch,
        topological_order=[0, 1, 2],
    )

    X_input = torch.tensor([[1.0, 0.0, 0.0]])
    output = scm(X_input)
    assert output.shape == X_input.shape

    # Check jacobian
    X_input.requires_grad_(True)
    jacobian = torch.autograd.functional.jacobian(
        lambda x: scm(x, structural=True), X_input
    )
    jacobian = jacobian.diagonal(dim1=0, dim2=2).permute(2, 0, 1)

    # Manually compute expected jacobian
    expected_jacobian = torch.tensor(
        [[[1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [1.0, 1.0, 0.0]]]
    )
    assert torch.allclose(jacobian, expected_jacobian)


def test_compute_cate():
    adjacency = np.array([[0, 1, 1], [0, 0, 1], [0, 0, 0]])
    assignments_torch = [
        None,
        lambda Xp: Xp[:, 0] * 2,  # Node 1 = 2 * Node 0
        lambda Xp: Xp[:, 0] + Xp[:, 1],  # Node 2 = Node 0 + Node 1
    ]
    scm = DifferentiableSCM(
        adjacency_matrix=adjacency,
        assignments=assignments_torch,
        topological_order=[0, 1, 2],
    )

    class DummyModel:
        def __call__(self, x):
            return x[:, 2]  # Predict Node 2

    input = torch.tensor([[1.0, 0.0, 0.0]])
    baseline = torch.tensor([0.0, 0.0, 0.0])
    cate = compute_cate(scm, DummyModel(), input, baseline, n_noise_samples=100)

    # The true CATE for changing Node 0 from 0 to 1 is:
    # Node 1 changes from 0 to 2, and Node 2 changes from 0 to 3, so total effect is 3.
    assert torch.isclose(cate[0, 0], torch.tensor(3.0), atol=0.1)
