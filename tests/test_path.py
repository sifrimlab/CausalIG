import torch

from causalig.attr._utils.path import LinearPathSampler, linear_path_samples


def test_linear_path_samples():
    baseline = torch.tensor([0.0, 0.0])
    input = torch.tensor([1.0, 1.0])
    n_steps = 5

    path_samples = linear_path_samples(baseline, input, n_steps)

    expected_samples = torch.tensor(
        [[0.0, 0.0], [0.25, 0.25], [0.5, 0.5], [0.75, 0.75], [1.0, 1.0]]
    )

    assert torch.allclose(path_samples.squeeze(), expected_samples, atol=1e-6)


def test_linear_path_samples_stochastic():
    baseline = torch.tensor([[0.0, 0.0], [1.0, 1.0]])
    input = torch.tensor([[1.0, 1.0], [2.0, 2.0]])
    n_steps = 3

    path_samples, alphas = LinearPathSampler().sample_pairwise_stochastic(
        baseline, input, n_steps
    )

    assert path_samples.shape == (3, 2, 2, 2)
    assert alphas.shape == (3, 2, 1, 1)
