import numpy as np
import pytest
import torch

from causalig.attr import CausalIntegratedGradients
from causalig.attr._utils.surrogate import MASK_VAL


def test_integrated_two_inputs_simple_two_steps():
    class SimpleModel:
        def __call__(self, x):
            return x[:, 0:1] + x[:, 1:2]

    class SimpleAutoencoder:
        def __call__(self, x):
            return torch.hstack([x[:, 0:1], x[:, 0:1]])

    model = SimpleModel()
    autoencoder = SimpleAutoencoder()

    input = torch.tensor([1.0, 2.0])
    baseline = torch.tensor([0.0, 0.0])

    ordering = [0, 1]

    R1, R2 = CausalIntegratedGradients(model).attribute(
        input, autoencoder, ordering, baseline, n_steps=2
    )

    # assert sum
    assert torch.isclose(
        R1 + R2, model(input.unsqueeze(0)) - model(baseline.unsqueeze(0))
    ).all()


def test_integrated_two_inputs_simple_ten_steps():
    class SimpleModel:
        def __call__(self, x):
            return x[:, 0:1] + x[:, 1:2]

    class SimpleAutoencoder:
        def __call__(self, x):
            return torch.hstack([x[:, 0:1], x[:, 0:1]])

    model = SimpleModel()
    autoencoder = SimpleAutoencoder()

    ordering = [0, 1]

    input = torch.tensor([1.0, 2.0])
    baseline = torch.tensor([0.0, 0.0])

    R1, R2 = CausalIntegratedGradients(model).attribute(
        input, autoencoder, ordering, baseline, n_steps=10
    )

    # assert sum
    assert torch.isclose(
        R1 + R2, model(input.unsqueeze(0)) - model(baseline.unsqueeze(0))
    ).all()

    # assert equal to two steps
    R1_2, R2_2 = CausalIntegratedGradients(model).attribute(
        input, autoencoder, ordering, baseline, n_steps=2
    )
    assert torch.isclose(R1, R1_2, atol=1e-2).all()
    assert torch.isclose(R2, R2_2, atol=1e-2).all()


def test_integrated_lots_of_inputs_linear():
    class SimpleModel:
        def __call__(self, x):
            return x.sum(dim=1, keepdim=True)

    class SimpleAutoencoder:
        def __call__(self, x):
            # Give back the sum of all previous inputs in all next inputs
            assert MASK_VAL == 0.0
            return torch.cumsum(x, dim=1)

    model = SimpleModel()
    autoencoder = SimpleAutoencoder()

    ordering = [0, 1, 2, 3, 4]

    input = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    baseline = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0])

    R = CausalIntegratedGradients(model).attribute(
        input, autoencoder, ordering, baseline, n_steps=10
    )

    # assert sum
    assert torch.isclose(
        R.sum(), model(input.unsqueeze(0)) - model(baseline.unsqueeze(0))
    ).all()


def test_integrated_lots_of_inputs_complex_function():
    class ComplexModel:
        def __call__(self, x):
            return (x**2).sum(dim=1, keepdim=True) + 2 * x.sum(dim=1, keepdim=True)

    class SimpleAutoencoder:
        def __call__(self, x):
            # Give back the sum of all previous inputs in all next inputs
            assert MASK_VAL == 0.0
            return torch.cumsum(x, dim=1)

    model = ComplexModel()
    autoencoder = SimpleAutoencoder()

    ordering = [0, 1, 2, 3, 4]

    input = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    baseline = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0])

    R = CausalIntegratedGradients(model).attribute(
        input, autoencoder, ordering, baseline, n_steps=10
    )

    print(R)
    # assert sum
    assert torch.isclose(
        R.sum(), model(input.unsqueeze(0)) - model(baseline.unsqueeze(0))
    ).all()


def test_ordering():
    class ComplexModel:
        def __call__(self, x):
            return (x**2).sum(dim=1, keepdim=True) + 2 * x.sum(dim=1, keepdim=True)

    special_ordering = [3, 0, 4, 1, 2]  # shuffled ordering

    class ComplexUnorderedModel:
        def __call__(self, x):
            x_ordered = x[:, special_ordering]
            out_ordered = (x_ordered**2).sum(dim=1, keepdim=True) + 2 * x_ordered.sum(
                dim=1, keepdim=True
            )
            return out_ordered

    class SimpleAutoencoder:
        def __call__(self, x):
            # Give back the sum of all previous inputs in all next inputs
            assert MASK_VAL == 0.0
            return torch.cumsum(x, dim=1)

    class SimpleUnorderedAutoencoder:
        def __call__(self, x):
            assert MASK_VAL == 0.0
            # Give back the sum of all previous inputs in all next inputs
            x_ordered = x[:, special_ordering]
            x_out_ordered = torch.cumsum(x_ordered, dim=1)
            return x_out_ordered[:, np.argsort(special_ordering)]  # reorder back

    model = ComplexModel()
    unordered_model = ComplexUnorderedModel()

    autoencoder = SimpleAutoencoder()
    unordered_autoencoder = SimpleUnorderedAutoencoder()

    ordering = [0, 1, 2, 3, 4]

    input = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    unordered_input = input[np.argsort(special_ordering)].clone().detach()

    baseline = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0])

    R = CausalIntegratedGradients(model).attribute(
        input, autoencoder, ordering, baseline, n_steps=10
    )

    R_unordered = CausalIntegratedGradients(unordered_model).attribute(
        unordered_input, unordered_autoencoder, special_ordering, baseline, n_steps=10
    )

    # assert sum
    assert torch.isclose(
        R_unordered.sum(), model(input.unsqueeze(0)) - model(baseline.unsqueeze(0))
    ).all()

    # assert equal
    assert torch.isclose(R, R_unordered[special_ordering]).all()


def test_step_chunk():
    class SimpleModel:
        def __call__(self, x):
            return x[:, 0:1] + x[:, 1:2]

    class SimpleAutoencoder:
        def __call__(self, x):
            return torch.hstack([x[:, 0:1], x[:, 0:1]])

    model = SimpleModel()
    autoencoder = SimpleAutoencoder()

    ordering = [0, 1]

    input = torch.tensor([1.0, 2.0], requires_grad=True)
    baseline = torch.tensor([0.0, 0.0], requires_grad=True)

    R1, R2 = CausalIntegratedGradients(model).attribute(
        input, autoencoder, ordering, baseline, n_steps=10, step_chunk_size=3
    )

    R1_ref, R2_ref = CausalIntegratedGradients(model).attribute(
        input, autoencoder, ordering, baseline, n_steps=10
    )

    # assert sum
    assert torch.isclose(
        R1 + R2, model(input.unsqueeze(0)) - model(baseline.unsqueeze(0))
    ).all()

    # assert equal to no chunking
    assert torch.isclose(R1, R1_ref).all()
    assert torch.isclose(R2, R2_ref).all()


def test_step_chunk_complex():
    class ComplexModel:
        def __call__(self, x):
            return (x**2).sum(dim=1, keepdim=True) + 2 * x.sum(dim=1, keepdim=True)

    class SimpleAutoencoder:
        def __call__(self, x):
            # Give back the sum of all previous inputs in all next inputs
            assert MASK_VAL == 0.0
            return torch.cumsum(x, dim=1)

    model = ComplexModel()
    autoencoder = SimpleAutoencoder()

    ordering = [0, 1, 2, 3, 4]

    input = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], requires_grad=True)
    baseline = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0], requires_grad=True)

    R = CausalIntegratedGradients(model).attribute(
        input, autoencoder, ordering, baseline, n_steps=50, step_chunk_size=7
    )

    R_ref = CausalIntegratedGradients(model).attribute(
        input, autoencoder, ordering, baseline, n_steps=50
    )

    print(R)
    # assert sum
    assert torch.isclose(
        R.sum(), model(input.unsqueeze(0)) - model(baseline.unsqueeze(0))
    ).all()

    # assert equal to no chunking
    assert torch.isclose(R, R_ref).all()


def test_dim_chunk_complex():
    class ComplexModel:
        def __call__(self, x):
            return (x**2).sum(dim=1, keepdim=True) + 2 * x.sum(dim=1, keepdim=True)

    class SimpleAutoencoder:
        def __call__(self, x):
            # Give back the sum of all previous inputs in all next inputs
            assert MASK_VAL == 0.0
            return torch.cumsum(x, dim=1)

    model = ComplexModel()
    autoencoder = SimpleAutoencoder()

    ordering = [0, 1, 2, 3, 4]

    input = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    baseline = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0])

    R = CausalIntegratedGradients(model).attribute(
        input,
        autoencoder,
        ordering,
        baseline,
        n_steps=50,
        step_chunk_size=7,
        dim_chunk_size=3,
    )

    R_ref = CausalIntegratedGradients(model).attribute(
        input, autoencoder, ordering, baseline, n_steps=50
    )

    print(R)
    # assert sum
    assert torch.isclose(
        R.sum(), model(input.unsqueeze(0)) - model(baseline.unsqueeze(0))
    ).all()

    # assert equal to no chunking
    assert torch.isclose(R, R_ref).all()


def test_multiclass_output():
    class ComplexModel:
        def __call__(self, x):
            return torch.hstack(
                [
                    (x**2).sum(dim=1, keepdim=True) + 2 * x.sum(dim=1, keepdim=True),
                    (x**3).sum(dim=1, keepdim=True) + 3 * x.sum(dim=1, keepdim=True),
                ]
            )

    class SimpleAutoencoder:
        def __call__(self, x):
            # Give back the sum of all previous inputs in all next inputs
            assert MASK_VAL == 0.0
            return torch.cumsum(x, dim=1)

    model = ComplexModel()
    autoencoder = SimpleAutoencoder()

    ordering = [0, 1, 2, 3, 4]

    input = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    baseline = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0])

    R = CausalIntegratedGradients(model).attribute(
        input, autoencoder, ordering, baseline, n_steps=1000
    )

    print(R)
    # assert sum
    assert torch.isclose(
        R[:, 0].sum(),
        model(input.unsqueeze(0))[0, 0] - model(baseline.unsqueeze(0))[0, 0],
    ).all()
    assert torch.isclose(
        R[:, 1].sum(),
        model(input.unsqueeze(0))[0, 1] - model(baseline.unsqueeze(0))[0, 1],
        rtol=0.01,
    ).all()


def test_complex_autoencoder():
    class ComplexModel:
        def __call__(self, x):
            return (x**2).sum(dim=1, keepdim=True) + 2 * x.sum(dim=1, keepdim=True)

    class ComplexAutoencoder:
        def __call__(self, x):
            # Give back the sum of all previous inputs in all next inputs plus some nonlinearity
            assert MASK_VAL == 0.0
            cumsum = torch.cumsum(x, dim=1)
            return cumsum + 0.1 * (cumsum**2)

    model = ComplexModel()
    autoencoder = ComplexAutoencoder()

    ordering = [0, 1, 2, 3, 4]

    input = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    baseline = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0])

    R = CausalIntegratedGradients(model).attribute(
        input, autoencoder, ordering, baseline, n_steps=3000
    )

    print(R)
    # assert sum
    assert torch.isclose(
        R.sum(), model(input.unsqueeze(0)) - model(baseline.unsqueeze(0)), rtol=1e-2
    ).all()


def test_complex_autoencoder_double_hops():
    class ComplexModel:
        def __call__(self, x):
            return (x**2).sum(dim=1, keepdim=True) + 2 * x.sum(dim=1, keepdim=True)

    class ComplexAutoencoder:
        def __call__(self, x):
            # Give back the sum of all previous inputs in all next inputs plus some nonlinearity
            assert MASK_VAL == 0.0
            cumsum = torch.cumsum(x, dim=1)
            return cumsum + 0.1 * (cumsum**2)

    model = ComplexModel()
    autoencoder = ComplexAutoencoder()

    ordering = [0, 1, 2, 3, 4]

    input = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    baseline = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0])

    R = CausalIntegratedGradients(model).attribute(
        input,
        autoencoder,
        ordering,
        baseline,
        n_steps=3000,
        include_second_order_indirect=True,
    )

    R_ = CausalIntegratedGradients(model).attribute(
        input,
        autoencoder,
        ordering,
        baseline,
        n_steps=3000,
        include_second_order_indirect=False,
    )

    print(R)
    # assert sum
    assert torch.isclose(
        R.sum(), model(input.unsqueeze(0)) - model(baseline.unsqueeze(0)), rtol=1e-2
    ).all()

    # assert different from only first-order indirect
    assert not torch.isclose(R, R_).all()


def _graph_setup():
    from causalig.models import MaskedGlobalLinear

    torch.manual_seed(0)
    d = 4
    adjacency = torch.zeros(d, d, dtype=torch.bool)
    adjacency[0, 1] = adjacency[0, 2] = adjacency[2, 3] = True
    X = torch.randn(64, d)
    model = torch.nn.Sequential(
        torch.nn.Linear(d, 8), torch.nn.Tanh(), torch.nn.Linear(8, 1)
    )
    surrogate = MaskedGlobalLinear(adjacency_matrix=adjacency, X_train=X)
    return model, surrogate, adjacency, X


def test_ordering_is_derived_from_adjacency_matrix():
    from causalig.attr import CausalExpectedGradients, CausalExpectedGradientsMultiStep

    model, surrogate, adjacency, X = _graph_setup()
    ordering = [0, 1, 2, 3]
    kw = dict(adjacency_matrix=adjacency, separate_effects=True)

    with_ordering = CausalIntegratedGradients(model).attribute(
        X[:5], surrogate, ordering, torch.zeros(4), n_steps=20, **kw
    )
    without = CausalIntegratedGradients(model).attribute(
        X[:5], surrogate, baseline=torch.zeros(4), n_steps=20, **kw
    )
    for a, b in zip(with_ordering, without):
        torch.testing.assert_close(a, b)

    ms = CausalExpectedGradientsMultiStep(model)
    for a, b in zip(
        ms.attribute(X[:5], surrogate, ordering, X[10:13], n_steps=20, **kw),
        ms.attribute(X[:5], surrogate, baselines=X[10:13], n_steps=20, **kw),
    ):
        torch.testing.assert_close(a, b)

    single = CausalExpectedGradients(model)
    for a, b in zip(
        single.attribute(X[:5], surrogate, ordering, X[10:13], seed=0, **kw),
        single.attribute(X[:5], surrogate, baselines=X[10:13], seed=0, **kw),
    ):
        torch.testing.assert_close(a, b)


def test_ordering_required_without_adjacency_matrix():
    model, surrogate, _, X = _graph_setup()
    with pytest.raises(ValueError, match="adjacency_matrix"):
        CausalIntegratedGradients(model).attribute(
            X[:2], surrogate, baseline=torch.zeros(4)
        )


def test_cyclic_adjacency_matrix_is_rejected():
    model, surrogate, adjacency, X = _graph_setup()
    cyclic = adjacency.clone()
    cyclic[3, 0] = True  # 0 -> 2 -> 3 -> 0
    with pytest.raises(ValueError, match="cycle"):
        CausalIntegratedGradients(model).attribute(
            X[:2], surrogate, baseline=torch.zeros(4), adjacency_matrix=cyclic
        )


def test_missing_baseline_is_reported():
    model, surrogate, adjacency, X = _graph_setup()
    with pytest.raises(TypeError, match="baseline"):
        CausalIntegratedGradients(model).attribute(
            X[:2], surrogate, adjacency_matrix=adjacency
        )
