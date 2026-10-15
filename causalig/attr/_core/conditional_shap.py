"""Conditional SHAP attribution and maskers."""

from collections import OrderedDict
from collections.abc import Callable

import numpy as np
import torch

from causalig.attr._core.attribution import Attribution

try:
    import shap
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "ConditionalSHAP requires the optional 'shap' dependency. "
        "Install it with `pip install causalig[shap]` (or `pip install shap`)."
    ) from e


class ConditionalMasker:
    """
    SHAP masker that fills masked features using a conditional sampler.

    For each SHAP coalition (boolean mask), the masker calls
    `conditional_sampler` to draw samples of the unobserved features from
    their conditional distribution given the observed features. This respects
    marginal feature correlations without requiring a full causal model.

    Args:
        conditional_sampler: Callable with signature
            ``(x, mask, n_samples) -> array (n_samples, n_masked)``.
            Given instance `x` and boolean mask, returns conditional samples
            for the masked (``False``) features.
        n_samples: Number of conditional samples per coalition. Default 50.
    """

    def __init__(self, conditional_sampler: Callable, n_samples: int = 50):
        self.conditional_sampler = conditional_sampler
        self.n_samples = n_samples

    def __call__(self, mask, x):
        """
        Generate conditionally-imputed samples for a SHAP coalition.

        Called by SHAP's ``PermutationExplainer`` for each coalition subset.

        Args:
            mask: Boolean array of length `d`; ``True`` = feature is in the
                coalition (observed), ``False`` = feature is masked out.
            x: Numpy array of length `d`; the instance being explained.

        Returns:
            Array of shape `(n_samples, d)` where unobserved features are
            filled with conditional samples and observed features are fixed
            to their values in `x`.
        """
        mask = np.asarray(mask, dtype=bool)
        x = np.asarray(x, dtype=np.float32)
        cond_samples = self.conditional_sampler(x, mask, self.n_samples)
        samples = np.broadcast_to(x, (self.n_samples, x.shape[0])).copy()
        if cond_samples.shape[1] > 0:
            samples[:, ~mask] = cond_samples
        return samples


class GaussianConditionalSampler:
    """
    Conditional sampler based on a multivariate Gaussian approximation.

    Fits a joint Gaussian to training data and samples from the conditional
    distribution ``p(X_missing | X_observed = x_obs)`` using the standard
    Gaussian conditioning formula. Inverse covariance computations are cached
    per observed-feature coalition to avoid redundant work across SHAP calls.

    Args:
        X_train: Training data array `(n, d)` used to estimate the mean and
            covariance of the joint distribution.
        ridge: Ridge regularisation added to covariance matrices before
            inversion for numerical stability. Default ``1e-8``.
        max_cache_size: Maximum number of coalition patterns to keep in the
            LRU cache. Oldest entries are evicted when full. Default 512.
    """

    def __init__(
        self, X_train: np.ndarray, ridge: float = 1e-8, max_cache_size: int = 512
    ):
        X_train = np.asarray(X_train, dtype=np.float64)
        if X_train.ndim != 2:
            raise ValueError("X_train must be a 2D array")

        self.mu = X_train.mean(axis=0)
        self.cov = np.cov(X_train.T)
        self._p = self.mu.shape[0]
        self._ridge = ridge
        self._max_cache_size = max_cache_size
        self._cache = OrderedDict()

    def __call__(self, x, mask, n_samples=50):
        """
        Sample from the Gaussian conditional distribution.

        Args:
            x: Instance array of length `d`.
            mask: Boolean array of length `d`; ``True`` = observed.
            n_samples: Number of samples to draw.

        Returns:
            Float32 array of shape `(n_samples, n_missing)` containing
            samples of the unobserved features.
        """
        x = np.asarray(x, dtype=np.float64)
        mask = np.asarray(mask, dtype=bool)
        if mask.shape[0] != self._p:
            raise ValueError("mask length must match number of features")

        obs = np.flatnonzero(mask)
        miss = np.flatnonzero(~mask)

        if miss.size == 0:
            return np.zeros((n_samples, 0), dtype=np.float32)

        if obs.size == 0:
            cov_mm = self.cov[np.ix_(miss, miss)] + self._ridge * np.eye(miss.size)
            return np.random.multivariate_normal(
                self.mu[miss], cov_mm, n_samples
            ).astype(np.float32)

        cache_key = tuple(obs.tolist())
        cached = self._cache.get(cache_key)

        if cached is None:
            cov_oo = self.cov[np.ix_(obs, obs)] + self._ridge * np.eye(obs.size)
            cov_mo = self.cov[np.ix_(miss, obs)]
            cov_mm = self.cov[np.ix_(miss, miss)]

            cov_oo_inv = np.linalg.inv(cov_oo)
            cond_cov = cov_mm - cov_mo @ cov_oo_inv @ cov_mo.T
            cond_cov = 0.5 * (cond_cov + cond_cov.T) + self._ridge * np.eye(
                cond_cov.shape[0]
            )

            cached = {
                "obs": obs,
                "miss": miss,
                "mu_o": self.mu[obs],
                "mu_m": self.mu[miss],
                "cov_oo_inv": cov_oo_inv,
                "cov_mo": cov_mo,
                "cond_cov": cond_cov,
            }
            if len(self._cache) >= self._max_cache_size:
                self._cache.popitem(last=False)
            self._cache[cache_key] = cached

        x_o = x[cached["obs"]]
        cond_mean = cached["mu_m"] + cached["cov_mo"] @ (
            cached["cov_oo_inv"] @ (x_o - cached["mu_o"])
        )
        return np.random.multivariate_normal(
            cond_mean, cached["cond_cov"], n_samples
        ).astype(np.float32)


def gaussian_conditional_sampler(
    x, mask, n_samples=50, X_train: np.ndarray | None = None
):
    """
    Convenience wrapper: construct a `GaussianConditionalSampler` and sample once.

    Prefer constructing `GaussianConditionalSampler` directly when the same
    instance will be called repeatedly (avoids re-fitting the covariance on
    every call).

    Args:
        x: Instance array of length `d`.
        mask: Boolean array of length `d`; ``True`` = observed.
        n_samples: Number of conditional samples to draw.
        X_train: Training data `(n, d)` used to estimate the distribution.

    Returns:
        Float32 array of shape `(n_samples, n_missing)`.
    """
    if X_train is None:
        raise ValueError("X_train must be provided")
    return GaussianConditionalSampler(X_train)(x, mask, n_samples)


class ConditionalSHAP(Attribution):
    """
    SHAP attributions using conditional imputation of masked features.

    Wraps SHAP's ``PermutationExplainer`` with a `ConditionalMasker` so
    that each coalition's unobserved features are filled from a conditional
    distribution rather than marginalised independently. This removes the
    assumption of feature independence made by standard (marginal) SHAP
    while not requiring a full causal model.

    Args:
        forward_func: Trained PyTorch model (must have ``.parameters()``).
        background: Training data `(n, d)` used to fit the conditional
            distribution.
        type: Conditional sampler type. Currently only ``'gaussian'`` is
            supported, which fits a multivariate Gaussian to `background`.
        n_samples: Conditional samples per coalition. Default 50.
        npermutations: Permutations for SHAP estimation. Default 10.
        seed: Seed for the permutation sampling and conditional sampling.
            Pass one for reproducible attributions: shap reseeds NumPy's global
            RNG when the explainer is created, from fresh entropy if ``None``.
    """

    def __init__(
        self,
        forward_func,
        background,
        type="gaussian",
        n_samples: int = 50,
        npermutations: int = 10,
        seed: int | None = None,
    ):
        super().__init__(forward_func)
        device = next(forward_func.parameters()).device
        self.npermutations = npermutations
        self._forward_func = forward_func.eval()

        def model_wrapper(x):
            with torch.inference_mode():
                x_tensor = torch.as_tensor(x, dtype=torch.float32, device=device)
                return self._forward_func(x_tensor).cpu().numpy()

        if type == "gaussian":
            gaussian_sampler = GaussianConditionalSampler(background)
            self.masker = ConditionalMasker(
                lambda x, mask, ns: gaussian_sampler(x, mask, ns),
                n_samples=n_samples,
            )
        else:
            raise ValueError(f"Unknown conditional sampler type: {type}")

        self.explainer = shap.PermutationExplainer(
            model_wrapper, masker=self.masker, seed=seed
        )

    def attribute(self, input, npermutations: int | None = None):
        """
        Compute ConditionalSHAP attributions.

        Args:
            input: Input tensor `(batch, d)`; pass a single sample as
                `(1, d)`. Converted to numpy internally for SHAP
                compatibility.
            npermutations: Override the default number of permutations set
                at construction. ``None`` uses the constructor value.

        Returns:
            Numpy array of SHAP values, `(batch, d)` for a single-output
            model.
        """
        return self.explainer.shap_values(
            input.cpu().numpy(),
            npermutations=(
                self.npermutations if npermutations is None else npermutations
            ),
        )
