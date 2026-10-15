"""
Synthetic data generation utilities.
"""

import numpy as np
import torch
from sempler import ANM
from sempler.generators import dag_avg_deg

from causalig.models.scm import DifferentiableSCM
from experiments.data.assignments import generate_random_assignments_torch_and_numpy

DEFAULT_EXPECTED_DEGREE = 5
ANM_NOISE_STD = 0.01
CATE_NOISE_STD = 0.01


def dag_max_depth(num_features, k, max_depth, random_state=None):
    rng = np.random.default_rng(random_state)

    # assign nodes to layers
    layers = rng.integers(0, max_depth, size=num_features)

    # enforce at least one node in each layer
    for i in range(max_depth):
        layers[i] = i

    ordering = np.argsort(layers)
    layers = layers[ordering]

    dag = np.zeros((num_features, num_features))

    # approximate probability to match expected degree
    p = k / (num_features - 1)

    for i in range(num_features):
        for j in range(i + 1, num_features):
            if layers[j] > layers[i] and rng.random() < p:
                dag[i, j] = 1

    return dag, np.arange(num_features)


def generate_synthetic_data(
    num_features,
    num_causal,
    num_samples,
    noise_multiplier=0.1,
    seed=None,
    max_depth=None,
    return_assignment_metadata=False,
):
    """
    Generate synthetic data using a DAG with specified number of features and causal features.

    Args:
        num_features (int): Total number of features.
        num_causal (int): Number of causal features that influence the target variable.
        num_samples (int): Number of samples to generate.
        noise_multiplier (float): Noise level for target variable (default: 0.1)
        return_assignment_metadata (bool): if True, also return the per-node
            true-assignment metadata produced by
            `generate_random_assignments_torch_and_numpy` (type/parents/weights
            per node). Default False, preserving the original 5-tuple return
            for existing callers.

    Returns:
        features (np.ndarray): Generated feature matrix.
        target (np.ndarray): Generated target variable.
        dag (np.ndarray): The generated DAG adjacency matrix.
        causal_indices (np.ndarray): Indices of the causal features.
        scm (DifferentiableSCM): The differentiable SCM model.
        assignment_metadata (list, only if return_assignment_metadata=True):
            per-node dicts describing the true assignment — see
            `generate_random_assignments_torch_and_numpy`.
        feature_mean, feature_std (np.ndarray, only if
            return_assignment_metadata=True): per-feature mean/std used to
            standardize `features`, i.e. `raw = features * feature_std +
            feature_mean`. Needed to evaluate `scm.assignments[node]` (or
            `assignment_metadata`'s weights) in the coordinate system they
            were actually defined in — see the standardization comment
            below for why applying them directly to standardized values
            is not equivalent.
    """
    if seed is not None:
        np.random.seed(seed)

    # Generate a DAG.
    if max_depth is not None:
        dag, ordering = dag_max_depth(
            num_features,
            k=DEFAULT_EXPECTED_DEGREE,
            max_depth=max_depth,
            random_state=seed,
        )
    else:
        dag, ordering = dag_avg_deg(
            num_features,
            k=DEFAULT_EXPECTED_DEGREE,
            return_ordering=True,
            random_state=seed,
        )

    # Reorder DAG so ordering is parents before children
    dag = dag[np.ix_(ordering, ordering)]

    # Generate random assignments for the DAG, needed by torch and sempler.
    assignment_gen_result = generate_random_assignments_torch_and_numpy(
        dag, return_metadata=return_assignment_metadata
    )
    if return_assignment_metadata:
        assignments_torch, assignments_numpy, assignment_metadata = (
            assignment_gen_result
        )
    else:
        assignments_torch, assignments_numpy = assignment_gen_result

    # Create a differentiable SCM model
    scm = DifferentiableSCM(
        adjacency_matrix=dag,
        assignments=assignments_torch,
        topological_order=list(range(num_features)),
    )

    # Set noise distributions for ANM sampling.
    noise_distributions = [
        lambda n: ANM_NOISE_STD * np.random.randn(n) for _ in range(num_features)
    ]

    # Initialize ANM model
    anm = ANM(
        A=dag, assignments=assignments_numpy, noise_distributions=noise_distributions
    )

    # Generate features.
    features = anm.sample(num_samples, random_state=seed)

    # Standardize features.
    feature_mean = features.mean(axis=0)
    feature_std = features.std(axis=0)
    features = (features - feature_mean) / feature_std

    # Generate target variable based on random causal features.
    causal_indices = np.random.choice(num_features, num_causal, replace=False)
    target = features[:, causal_indices].sum(
        axis=1
    ) + noise_multiplier * np.random.randn(num_samples)

    if return_assignment_metadata:
        return (
            features,
            target,
            dag,
            causal_indices,
            scm,
            assignment_metadata,
            feature_mean,
            feature_std,
        )
    return features, target, dag, causal_indices, scm


def compute_cate(scm, model, inputs, baseline, n_noise_samples=10, seed=None):
    """
    Compute Conditional Average Treatment Effect (CATE) for a batch of inputs and baseline.

    Args:
        scm (DifferentiableSCM): The SCM model to use for counterfactual generation.
        model: The predictive model for which to compute CATE.
        inputs: The input batch for which to compute CATE (shape: (batch, d))
        baseline: The baseline instance for comparison (shape: (d,))
        n_noise_samples: Number of noise samples to average over for CATE estimation (default: 10)
        seed: Optional random seed for deterministic noise sampling

    Returns:
        cate: The estimated CATE values (shape: (batch, d,))
    """
    device = inputs.device
    batch, d = inputs.shape
    n_noise_samples = max(1, int(n_noise_samples))

    generator = None
    if seed is not None:
        generator = torch.Generator(device=device).manual_seed(seed)

    cate = torch.zeros(batch, d, device=device)

    # Reuse the expanded input tensor for all features.
    inputs_expanded = inputs.unsqueeze(0).expand(n_noise_samples, -1, -1)
    inputs_flat = inputs_expanded.reshape(n_noise_samples * batch, d)

    with torch.inference_mode():
        for i in range(d):
            noise = (
                torch.randn(
                    n_noise_samples,
                    batch,
                    d,
                    device=device,
                    generator=generator,
                )
                * CATE_NOISE_STD
            )
            noise_flat = noise.reshape(n_noise_samples * batch, d)

            do_values_intervened = (
                inputs[:, i]
                .unsqueeze(0)
                .expand(n_noise_samples, -1)
                .reshape(n_noise_samples * batch, 1)
            )

            baseline_value = baseline[i].to(device=device, dtype=inputs.dtype)
            do_values_baseline = torch.full(
                (n_noise_samples * batch, 1),
                baseline_value,
                device=device,
                dtype=inputs.dtype,
            )

            x_intervened = scm.forward_do(
                inputs_flat,
                noise_flat,
                do_nodes=[i],
                do_values=do_values_intervened,
            )

            x_baseline = scm.forward_do(
                inputs_flat,
                noise_flat,
                do_nodes=[i],
                do_values=do_values_baseline,
            )

            y_intervened = model(x_intervened).reshape(n_noise_samples, batch, -1)
            y_baseline = model(x_baseline).reshape(n_noise_samples, batch, -1)
            mean_effect = (y_intervened - y_baseline).mean(dim=0)

            cate[:, i] = mean_effect.squeeze(-1)

    return cate
