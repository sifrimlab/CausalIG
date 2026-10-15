"""
Evaluation metrics for attribution methods.
"""

import numpy as np
import torch


def evaluate_model_performance(
    model: torch.nn.Module,
    features: torch.Tensor,
    target: torch.Tensor,
    causal_indices: np.ndarray,
    threshold: float = 1e-6,
) -> dict[str, float]:
    """
    Evaluate model performance and feature selection.

    Args:
        model: Trained model
        features: Input features tensor of shape (batch, num_features)
        target: Target tensor of shape (batch,)
        causal_indices: Array of causal feature indices
        threshold: Threshold for considering weights as non-zero

    Returns:
        Dictionary with performance metrics:
        - 'mse': Mean squared error
        - 'num_nonzero_weights': Number of features with non-zero weights
        - 'num_causal_learned': Number of causal features with non-zero weights
        - 'total_causal': Total number of causal features
    """
    with torch.no_grad():
        predictions = model(features)
        mse = ((predictions - target) ** 2).mean().item()

        # Get learned weights
        if hasattr(model, "w"):
            learned_weights = model.w.cpu().numpy()
            non_zero_indices = set(np.where(np.abs(learned_weights) > threshold)[0])
            num_causal_learned = len(non_zero_indices.intersection(set(causal_indices)))
        else:
            # Model doesn't have explicit weights
            non_zero_indices = set()
            num_causal_learned = 0

    return {
        "mse": mse,
        "num_nonzero_weights": len(non_zero_indices),
        "num_causal_learned": num_causal_learned,
        "total_causal": len(causal_indices),
        "non_zero_indices": non_zero_indices,
        "learned_weights": learned_weights if hasattr(model, "w") else None,
    }


def _normalize(x):
    # Gaussian
    return x / (x.abs().sum(dim=1, keepdim=True) + 1e-8)


def mse(a, b):
    return ((a - b) ** 2).mean().item()


def pearson(a, b, row_wise=True):
    if row_wise:
        a = a - a.mean(dim=1, keepdim=True)
        b = b - b.mean(dim=1, keepdim=True)

        numerator = (a * b).sum(dim=1)
        denom = torch.sqrt((a**2).sum(dim=1) * (b**2).sum(dim=1)) + 1e-8

        corr = numerator / denom

        return corr.mean().item()

    a = a.flatten()
    b = b.flatten()
    return torch.corrcoef(torch.stack([a, b]))[0, 1].item()


def spearman(a, b, row_wise=True):
    def rank(x):
        return x.argsort().argsort().float()

    def rank_tensor(x):
        return x.argsort(dim=1).argsort(dim=1).float()

    if row_wise:
        a = rank_tensor(a)
        b = rank_tensor(b)
        return pearson(a, b, row_wise=True)

    return pearson(rank(a.flatten()), rank(b.flatten()), row_wise=False)


def compare_to_cate(attr, cate, nodes_to_compare=None):
    """Compare attributions against ground-truth CATE values.

    Args:
        attr: Attribution tensor of shape (batch, d).
        cate: Ground-truth CATE tensor of shape (batch, d).
        nodes_to_compare: Optional sequence of feature indices to restrict the
            comparison to (defaults to all features).

    Returns:
        Dict with keys ``"mse"``, ``"pearson"`` and ``"spearman"`` measuring the
        agreement between (normalized) attributions and CATE.
    """
    attr_n = _normalize(attr)
    cate_n = _normalize(cate)

    if nodes_to_compare is not None:
        attr_n = attr_n[:, nodes_to_compare]
        cate_n = cate_n[:, nodes_to_compare]
        attr = attr[:, nodes_to_compare]
        cate = cate[:, nodes_to_compare]

    return {
        "mse": mse(attr_n, cate_n),
        "pearson": pearson(attr, cate),
        "spearman": spearman(attr, cate),
    }
