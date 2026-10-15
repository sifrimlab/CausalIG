"""
Path sampling utilities for attribution methods.
"""

from abc import ABC, abstractmethod

import torch


class PathSampler(ABC):
    """
    Abstract base class for path sampling strategies.
    """

    @abstractmethod
    def sample(
        self, baseline: torch.Tensor, input: torch.Tensor, n_steps: int
    ) -> torch.Tensor:
        """
        Generate path samples between baseline and input.

        Args:
            baseline: Baseline tensor of shape (d,) or (batch, d)
            input: Input tensor of shape (d,) or (batch, d)
            n_steps: Number of steps along the path

        Returns:
            Path samples tensor of shape (n_steps, batch, d) or (n_steps, d)
        """
        raise NotImplementedError("Subclasses must implement sample()")


class LinearPathSampler(PathSampler):
    """
    Linear interpolation path sampler.
    """

    def sample(
        self,
        baseline: torch.Tensor,
        input: torch.Tensor,
        n_steps: int,
        return_alphas: bool = False,
    ) -> torch.Tensor:
        """
        Generate deterministic linear path samples using evenly spaced
        interpolation in alpha.

        Args:
            baseline: Baseline tensor of shape (d,) or (B, d)
            input: Input tensor of shape (d,) or (I, d)
            n_steps: Number of interpolation steps

        Returns:
            Pairwise path samples tensor of shape (n_steps, B, I, d), where
            each baseline is paired with each input.
        """
        if baseline.dim() == 1:
            baseline = baseline.unsqueeze(0)
        if input.dim() == 1:
            input = input.unsqueeze(0)

        B, d = baseline.shape
        I, _ = input.shape  # noqa: E741  (I = number of inputs, matches docstring)

        alphas = torch.linspace(0.0, 1.0, steps=n_steps, device=baseline.device)
        alphas = alphas.view(n_steps, 1, 1, 1)
        baseline_exp = baseline.view(1, B, 1, d)
        input_exp = input.view(1, 1, I, d)
        path_samples = baseline_exp + alphas * (input_exp - baseline_exp)

        if return_alphas:
            return path_samples, alphas
        return path_samples

    def sample_pairwise_stochastic(
        self,
        baseline: torch.Tensor,
        input: torch.Tensor,
        n_steps: int,
        alpha_min: float = 0.0,
        seed: int = None,
        ordered: bool = True,
        include_endpoints: bool = False,
        stratified: bool = False,
    ) -> torch.Tensor:
        """
        Generate stochastic linear path samples between each baseline and
        every input using independent uniform samples per baseline.

        For each baseline b and step s:
            alpha[s, b] ~ Uniform(0, 1)

        Paths are generated as:
            baseline[b] + alpha[s, b] * (input[j] - baseline[b])

        Args:
            baseline: Baseline tensor of shape (d,) or (B, d)
            input: Input tensor of shape (d,) or (I, d)
            n_steps: Number of stochastic samples per baseline
            alpha_min: Minimum alpha value to ensure samples are not too close to baseline
            seed: Optional random seed for reproducibility
            ordered: Whether to order the samples
            include_endpoints: Whether to force alpha=0 and alpha=1 as first/last samples
            stratified: Whether to sample one alpha per bin (reduces tiny interval gaps)
        Returns:
            Path samples tensor of shape (n_steps, B, I, d)
        """
        if baseline.dim() == 1:
            baseline = baseline.unsqueeze(0)
        if input.dim() == 1:
            input = input.unsqueeze(0)

        B, d = baseline.shape
        I, _ = input.shape  # noqa: E741  (I = number of inputs, matches docstring)

        device = baseline.device

        generator = None
        if seed is not None:
            generator = torch.Generator(device=device)
            generator.manual_seed(seed)

        if stratified:
            if include_endpoints:
                assert n_steps >= 2, "n_steps must be at least 2 to include endpoints"
                n_inner = n_steps - 2
                if n_inner > 0:
                    jitter = torch.rand(
                        n_inner,
                        B,
                        device=device,
                        generator=generator,
                    )
                    bin_ids = torch.arange(
                        n_inner, device=device, dtype=jitter.dtype
                    ).unsqueeze(1)
                    inner = (bin_ids + jitter) / (n_steps - 1)
                    alphas = torch.cat(
                        [
                            torch.zeros(1, B, device=device, dtype=inner.dtype),
                            inner,
                            torch.ones(1, B, device=device, dtype=inner.dtype),
                        ],
                        dim=0,
                    )
                else:
                    alphas = torch.cat(
                        [
                            torch.zeros(1, B, device=device, dtype=baseline.dtype),
                            torch.ones(1, B, device=device, dtype=baseline.dtype),
                        ],
                        dim=0,
                    )
            else:
                jitter = torch.rand(
                    n_steps,
                    B,
                    device=device,
                    generator=generator,
                )
                bin_ids = torch.arange(
                    n_steps, device=device, dtype=jitter.dtype
                ).unsqueeze(1)
                alphas = (bin_ids + jitter) / n_steps
                alphas = alphas * (1 - alpha_min) + alpha_min
        else:
            # Independent stochastic alpha for each baseline
            alphas = (
                torch.rand(
                    n_steps,
                    B,
                    device=device,
                    generator=generator,
                )
                * (1 - alpha_min)
                + alpha_min
            )

            if include_endpoints:
                assert n_steps >= 2, "n_steps must be at least 2 to include endpoints"
                alphas = torch.cat(
                    [
                        torch.zeros(1, B, device=device),
                        alphas[1:-1],
                        torch.ones(1, B, device=device),
                    ],
                    dim=0,
                )

        alphas = alphas.view(n_steps, B, 1, 1)
        if ordered:
            alphas, _ = torch.sort(alphas, dim=0)

        baseline = baseline.view(1, B, 1, d)
        input = input.view(1, 1, I, d)

        path_samples = baseline + alphas * (input - baseline)

        return path_samples, alphas


def linear_path_samples(
    baseline: torch.Tensor, input: torch.Tensor, n_steps: int
) -> torch.Tensor:
    """
    Generate linear path samples between baseline and input.

    This is a convenience function that uses LinearPathSampler.
    Maintains backward compatibility with the original function.

    Args:
        baseline: Baseline tensor of shape (d,) or (batch, d)
        input: Input tensor of shape (d,) or (batch, d)
        n_steps: Number of steps along the path

    Returns:
        Path samples tensor of shape (n_steps, batch, d) or (n_steps, d)
    """
    sampler = LinearPathSampler()
    return sampler.sample(baseline, input, n_steps)
