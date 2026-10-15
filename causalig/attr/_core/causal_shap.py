"""Causal SHAP attribution built on top of a structural causal model."""

from collections.abc import Callable
from functools import partial

import torch

try:
    import shap
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "CausalSHAP requires the optional 'shap' dependency. "
        "Install it with `pip install causalig[shap]` (or `pip install shap`)."
    ) from e

from causalig.attr._core.attribution import Attribution


class CausalMasker:
    """
    SHAP masker that uses a structural causal model (SCM) to generate
    interventional background samples.

    For each SHAP coalition (defined by a boolean feature mask), the masker
    performs a ``do``-intervention on the observed features and samples the
    remaining features from the SCM, marginalising over noise. This respects
    the causal structure of the data rather than breaking feature correlations
    arbitrarily.

    Args:
        scm: Structural causal model with a ``forward_do(X_input, noise,
            do_nodes, do_values)`` method.
        noise_dist: Callable ``(batch, p, device) -> Tensor`` that samples
            exogenous noise for the SCM.
        n_samples: Number of interventional samples to draw per coalition.
            Higher values reduce variance at the cost of compute. Default 50.
        device: Device to run computations on. Defaults to CPU.
        background: Optional background dataset `(n, d)` used as the
            observational starting distribution before intervening. If
            ``None``, a zero background is used.
    """

    def __init__(
        self,
        scm,
        noise_dist: Callable[[int, int], torch.Tensor],
        n_samples: int = 50,
        device: torch.device | None = None,
        background: torch.Tensor | None = None,
        seed: int | None = None,
    ):
        self.scm = scm
        self.noise_dist = noise_dist
        self.n_samples = n_samples
        self.device = device or torch.device("cpu")

        if background is not None:
            self.background = torch.as_tensor(background, dtype=torch.float32)
            if self.background.shape[0] > n_samples:
                g = torch.Generator()
                if seed is not None:
                    g.manual_seed(seed)
                indices = torch.randperm(self.background.shape[0], generator=g)[
                    :n_samples
                ]
                self.background = self.background[indices]
            self.background = self.background.to(self.device)
        else:
            self.background = None

    def __call__(self, mask, x):
        """
        Generate interventional samples for a SHAP coalition.

        Called by SHAP's ``PermutationExplainer`` for each coalition subset.

        Args:
            mask: Boolean array of length `d`; ``True`` means the feature is
                present in the coalition (i.e. intervened on).
            x: Numpy array of length `d`; the instance being explained.

        Returns:
            Numpy array of shape `(n_samples, d)` containing samples from
            the interventional distribution ``P(X | do(X_mask = x_mask))``.
        """
        mask = torch.as_tensor(mask, dtype=torch.bool, device=self.device)
        x = torch.as_tensor(x, dtype=torch.float32, device=self.device)

        batch = self.n_samples
        p = len(mask)

        X_input = (
            self.background
            if self.background is not None
            else torch.zeros(batch, p, device=self.device)
        )

        noise = self.noise_dist(batch, p, device=self.device)
        do_nodes = torch.where(mask)[0]

        with torch.inference_mode():
            if do_nodes.numel() == 0:
                X = self.scm.forward_do(
                    X_input=X_input, noise=noise, do_nodes=None, do_values=None
                )
            else:
                do_values = x[do_nodes].unsqueeze(0).expand(batch, -1)
                X = self.scm.forward_do(
                    X_input=X_input, noise=noise, do_nodes=do_nodes, do_values=do_values
                )

        return X.cpu().numpy()


def gaussian_noise(
    batch: int,
    p: int,
    sigma: float = 1e-6,
    device: torch.device | None = None,
) -> torch.Tensor:
    """
    Sample isotropic Gaussian noise for SCM exogenous variables.

    Args:
        batch: Number of noise samples to draw.
        p: Number of features (noise dimensions).
        sigma: Standard deviation. Defaults to ``1e-6`` (near-zero noise,
            used when the SCM is evaluated near-deterministically).
        device: Target device.

    Returns:
        Tensor of shape `(batch, p)`.
    """
    return torch.randn(batch, p, device=device) * sigma


class CausalSHAP(Attribution):
    """
    SHAP attributions using causal interventional background samples.

    Wraps SHAP's ``PermutationExplainer`` with a `CausalMasker` so that
    each coalition's background distribution is generated via ``do``-calculus
    on the provided SCM rather than by marginalising over observational data.
    This gives attributions that respect the causal graph and are not
    confounded by spurious correlations.

    Args:
        forward_func: Trained PyTorch model (must have ``.parameters()``).
        scm: Structural causal model used by `CausalMasker`.
        background: Optional background dataset `(n, d)` tensor. Subsampled
            to `n_samples` rows at construction time.
        n_samples: Number of interventional samples per coalition. Default 50.
        npermutations: Number of permutations for SHAP estimation. Default 10.
        noise_sigma: Standard deviation of exogenous SCM noise. Default 0.01.
        seed: Seed for the permutation sampling (and background subsampling).
            Pass one for reproducible attributions: shap reseeds NumPy's global
            RNG when the explainer is created, from fresh entropy if ``None``.
            The SCM noise is drawn from torch's global RNG, so also call
            ``torch.manual_seed``.
    """

    def __init__(
        self,
        forward_func,
        scm,
        background=None,
        n_samples: int = 50,
        npermutations: int = 10,
        noise_sigma: float = 0.01,
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

        noise_dist = partial(gaussian_noise, sigma=noise_sigma)
        self.masker = CausalMasker(
            scm,
            noise_dist,
            n_samples=n_samples,
            device=device,
            background=background,
            seed=seed,
        )
        self.explainer = shap.PermutationExplainer(
            model_wrapper, masker=self.masker, seed=seed
        )

    def attribute(self, input, npermutations: int | None = None):
        """
        Compute CausalSHAP attributions.

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
