import torch

from causalig.attr import (
    CausalExpectedGradients,
    CausalExpectedGradientsMultiStep,
    CausalIntegratedGradients,
)
from causalig.attr._utils.surrogate import MASK_VAL


def test_two_inputs():
    class SimpleModel:
        def __call__(self, x):
            return x[:, 0:1] + x[:, 1:2]

    class SimpleAutoencoder:
        def __call__(self, x):
            return torch.hstack([x[:, 0:1], x[:, 0:1]])

    model = SimpleModel()
    autoencoder = SimpleAutoencoder()

    input = torch.tensor([1.0, 2.0])
    baselines = torch.zeros(100, 2)

    ordering = [0, 1]

    R1, R2 = CausalExpectedGradients(model).attribute(
        input, autoencoder, ordering, baselines
    )

    R1_, R2_ = CausalIntegratedGradients(model).attribute(
        input, autoencoder, ordering, baselines[0], n_steps=100
    )

    # assert sum
    assert torch.allclose(R1 + R2, torch.tensor([3.0]), atol=1e-2)

    assert torch.allclose(R1, R1_, atol=1e-2)
    assert torch.allclose(R2, R2_, atol=1e-2)


def test_lots_of_inputs_linear():
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
    baselines = torch.zeros(100, 5)

    R = CausalExpectedGradients(model).attribute(
        input, autoencoder, ordering, baselines
    )

    R_ = CausalIntegratedGradients(model).attribute(
        input, autoencoder, ordering, baselines[0], n_steps=100
    )

    # assert sum
    assert torch.isclose(
        R.sum(), model(input.unsqueeze(0)) - model(baselines[0].unsqueeze(0))
    ).all()

    assert torch.allclose(R, R_, atol=1e-2)


def test_lots_of_inputs_complex_function():
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
    baselines = torch.zeros(1000, 5)

    R = CausalExpectedGradients(model).attribute(
        input, autoencoder, ordering, baselines
    )

    R_ = CausalIntegratedGradients(model).attribute(
        input, autoencoder, ordering, baselines[0], n_steps=1000
    )

    print(R)
    # assert sum
    assert torch.isclose(
        R.sum(), model(input.unsqueeze(0)) - model(baselines[0].unsqueeze(0)), rtol=1e-2
    ).all()

    assert torch.allclose(R, R_, rtol=1e-2)


def test_baselines_chunk():
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
    baselines = torch.zeros(1000, 5)

    R = CausalExpectedGradients(model).attribute(
        input, autoencoder, ordering, baselines, baselines_chunk_size=67
    )

    R_ref = CausalExpectedGradients(model).attribute(
        input, autoencoder, ordering, baselines
    )

    print(R)
    # assert sum
    assert torch.isclose(
        R.sum(), model(input.unsqueeze(0)) - model(baselines[0].unsqueeze(0)), rtol=2e-2
    ).all()

    # assert equal to no chunking
    assert torch.isclose(R, R_ref, rtol=4e-2).all()


def test_different_baselines():
    class SimpleModel:
        def __call__(self, x):
            return x[:, 0:1] + x[:, 1:2]

    class SimpleAutoencoder:
        def __call__(self, x):
            return torch.hstack([x[:, 0:1], x[:, 0:1]])

    model = SimpleModel()
    autoencoder = SimpleAutoencoder()

    input = torch.tensor([1.0, 2.0])
    baselines = torch.randn(100, 2)
    ordering = [0, 1]

    R1, R2 = CausalExpectedGradients(model).attribute(
        input, autoencoder, ordering, baselines
    )

    R1_, R2_ = torch.zeros_like(R1), torch.zeros_like(R2)
    for b in baselines:

        R1_t, R2_t = CausalIntegratedGradients(model).attribute(
            input, autoencoder, ordering, b, n_steps=10
        )
        R1_ += R1_t
        R2_ += R2_t

    R1_ /= baselines.size(0)
    R2_ /= baselines.size(0)

    # assert sum
    assert torch.allclose(
        R1 + R2,
        model(input.unsqueeze(0)) - model(baselines).mean(dim=0, keepdim=True),
        atol=1e-2,
    )

    assert torch.allclose(R1, R1_, atol=1e-2)
    assert torch.allclose(R2, R2_, atol=1e-2)


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
    baselines = torch.randn(1000, 5)

    seed = 42

    R = CausalExpectedGradients(model).attribute(
        input, autoencoder, ordering, baselines, dim_chunk_size=2, seed=seed
    )

    R_ref = CausalExpectedGradients(model).attribute(
        input, autoencoder, ordering, baselines, seed=seed
    )

    print(R)
    # assert equal to no chunking
    assert torch.isclose(R, R_ref, rtol=1e-4).all()


def test_multiple_outputs():
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
    baselines = torch.randn(1000, 5)

    R = CausalExpectedGradients(model).attribute(
        input, autoencoder, ordering, baselines
    )

    # assert sum
    assert torch.isclose(
        R[:, 0].sum(),
        model(input.unsqueeze(0))[0, 0]
        - model(baselines).mean(dim=0, keepdim=True)[0, 0],
        rtol=2e-2,
    ).all()
    assert torch.isclose(
        R[:, 1].sum(),
        model(input.unsqueeze(0))[0, 1]
        - model(baselines).mean(dim=0, keepdim=True)[0, 1],
        rtol=2e-2,
    ).all()


def test_multistep():
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
    baselines = torch.randn(1000, 5)

    R = CausalExpectedGradientsMultiStep(model).attribute(
        input, autoencoder, ordering, baselines, n_steps=50
    )

    R_ref = CausalExpectedGradients(model).attribute(
        input, autoencoder, ordering, baselines
    )

    print(R)
    # assert sum
    assert torch.isclose(
        R.sum(), model(input.unsqueeze(0)) - model(baselines).mean(dim=0, keepdim=True)
    ).all()

    # assert equal to no chunking
    assert torch.isclose(R, R_ref, rtol=0.03).all()


def test_multiple_outputs_multistep():
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
    baselines = torch.randn(1000, 5)

    R = CausalExpectedGradientsMultiStep(model).attribute(
        input, autoencoder, ordering, baselines, n_steps=50
    )

    # assert sum
    assert torch.isclose(
        R[:, 0].sum(),
        model(input.unsqueeze(0))[0, 0]
        - model(baselines).mean(dim=0, keepdim=True)[0, 0],
        rtol=2e-2,
    ).all()
    assert torch.isclose(
        R[:, 1].sum(),
        model(input.unsqueeze(0))[0, 1]
        - model(baselines).mean(dim=0, keepdim=True)[0, 1],
        rtol=2e-2,
    ).all()


def test_multistep_baseline_chunk():
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
    baselines = torch.randn(1000, 5)

    R = CausalExpectedGradientsMultiStep(model).attribute(
        input, autoencoder, ordering, baselines, n_steps=50, baselines_chunk_size=67
    )

    R_ref = CausalExpectedGradientsMultiStep(model).attribute(
        input, autoencoder, ordering, baselines, n_steps=50
    )

    print(R)
    # assert sum
    assert torch.isclose(
        R.sum(),
        model(input.unsqueeze(0)) - model(baselines).mean(dim=0, keepdim=True),
        rtol=2e-2,
    ).all()

    # assert equal to no chunking
    assert torch.isclose(R, R_ref, rtol=4e-2).all()


def test_multistep_dim_chunk():
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
    baselines = torch.randn(1000, 5)

    R = CausalExpectedGradientsMultiStep(model).attribute(
        input, autoencoder, ordering, baselines, n_steps=50, dim_chunk_size=2
    )

    R_ref = CausalExpectedGradientsMultiStep(model).attribute(
        input, autoencoder, ordering, baselines, n_steps=50
    )

    print(R)
    # assert sum
    assert torch.isclose(
        R.sum(),
        model(input.unsqueeze(0)) - model(baselines).mean(dim=0, keepdim=True),
        rtol=1e-2,
    ).all()

    # assert equal to no chunking
    assert torch.isclose(R, R_ref, rtol=1e-4).all()


def test_multistep_step_chunk():
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
    baselines = torch.randn(1000, 5)

    R = CausalExpectedGradientsMultiStep(model).attribute(
        input, autoencoder, ordering, baselines, n_steps=50, step_chunk_size=7
    )

    R_ref = CausalExpectedGradientsMultiStep(model).attribute(
        input, autoencoder, ordering, baselines, n_steps=50
    )

    print(R)
    # assert sum
    assert torch.isclose(
        R.sum(),
        model(input.unsqueeze(0)) - model(baselines).mean(dim=0, keepdim=True),
        rtol=1e-2,
    ).all()

    # assert equal to no chunking
    assert torch.isclose(R, R_ref, rtol=1e-4).all()


def test_return_convergence_delta():
    class ComplexModel:
        def __call__(self, x):
            return (x**2).sum(dim=1, keepdim=True) + 2 * x.sum(dim=1, keepdim=True)

    class SimpleAutoencoder:
        def __call__(self, x):
            assert MASK_VAL == 0.0
            return torch.cumsum(x, dim=1)

    model = ComplexModel()
    autoencoder = SimpleAutoencoder()

    ordering = [0, 1, 2, 3, 4]

    input = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    baselines = torch.randn(1000, 5)

    expected_diff = model(input.unsqueeze(0)) - model(baselines).mean(
        dim=0, keepdim=True
    )

    R, delta = CausalExpectedGradients(model).attribute(
        input, autoencoder, ordering, baselines, return_convergence_delta=True
    )
    assert torch.isclose(R.sum() - expected_diff, delta, rtol=1e-2).all()

    R_ms, delta_ms = CausalExpectedGradientsMultiStep(model).attribute(
        input,
        autoencoder,
        ordering,
        baselines,
        n_steps=50,
        return_convergence_delta=True,
    )
    assert torch.isclose(R_ms.sum() - expected_diff, delta_ms).all()

    direct, indirect, delta_sep = CausalExpectedGradients(model).attribute(
        input,
        autoencoder,
        ordering,
        baselines,
        separate_effects=True,
        return_convergence_delta=True,
    )
    assert torch.isclose((direct + indirect).sum() - expected_diff, delta_sep).all()

    R_cig, delta_cig = CausalIntegratedGradients(model).attribute(
        input,
        autoencoder,
        ordering,
        baselines[0],
        n_steps=100,
        return_convergence_delta=True,
    )
    expected_diff_single_baseline = model(input.unsqueeze(0)) - model(
        baselines[0].unsqueeze(0)
    )
    assert torch.isclose(
        R_cig.sum() - expected_diff_single_baseline, delta_cig, atol=1e-2
    ).all()


def test_multistep_step_all_chunking():
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
    baselines = torch.randn(1000, 5)

    R = CausalExpectedGradientsMultiStep(model).attribute(
        input,
        autoencoder,
        ordering,
        baselines,
        n_steps=50,
        step_chunk_size=7,
        dim_chunk_size=2,
        baselines_chunk_size=67,
    )

    R_ref = CausalExpectedGradientsMultiStep(model).attribute(
        input, autoencoder, ordering, baselines, n_steps=50
    )

    print(R)
    # assert sum
    assert torch.isclose(
        R.sum(),
        model(input.unsqueeze(0)) - model(baselines).mean(dim=0, keepdim=True),
        rtol=1e-2,
    ).all()

    # assert equal to no chunking
    assert torch.isclose(R, R_ref, rtol=1e-4).all()
