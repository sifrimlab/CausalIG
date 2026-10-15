"""
Base class for attribution methods, following Captum's architecture.
"""

from abc import ABC, abstractmethod
from graphlib import CycleError, TopologicalSorter

import torch


class Attribution(ABC):
    """
    Abstract base class for all attribution methods in this package.

    Subclasses implement `attribute()` and inherit input validation and
    shape-normalisation utilities. The convention throughout is that a single
    sample may be passed as a 1-D tensor `(d,)`; the base class promotes it to
    `(1, d)` before computation and squeezes it back on return so callers never
    need to manage the batch dimension themselves.

    Args:
        forward_func: Model or callable to attribute. Must accept a
            `(batch, d)` float tensor and return a `(batch, C)` or
            `(batch,)` tensor.
    """

    def __init__(self, forward_func):
        self.forward_func = forward_func

    @abstractmethod
    def attribute(
        self, inputs: torch.Tensor, **kwargs
    ) -> torch.Tensor | tuple[torch.Tensor, ...]:
        """
        Compute attributions for the given inputs.

        Args:
            inputs: Input tensor of shape `(d,)` or `(batch, d)`.
            **kwargs: Additional arguments specific to each subclass.

        Returns:
            Attribution tensor of the same shape as `inputs`, or a tuple
            `(direct, indirect)` when `separate_effects=True`.
        """
        raise NotImplementedError

    def _format_input(self, inputs: torch.Tensor) -> tuple[torch.Tensor, bool]:
        """
        Promote a 1-D tensor to `(1, d)` and record whether it was 1-D.

        Args:
            inputs: Tensor of shape `(d,)` or `(batch, d)`.

        Returns:
            `(normalized, was_single)` where `normalized` is always `(batch, d)`
            and `was_single` is `True` when the original tensor was 1-D.
        """
        if inputs.dim() == 1:
            return inputs.unsqueeze(0), True
        return inputs, False

    def _format_output(
        self, attributions: torch.Tensor | tuple[torch.Tensor, ...], was_single: bool
    ) -> torch.Tensor | tuple[torch.Tensor, ...]:
        """
        Squeeze the batch dimension when the original input was 1-D.

        Args:
            attributions: Tensor `(batch, d)` or tuple of such tensors.
            was_single: Value returned by `_format_input`.

        Returns:
            Tensor `(d,)` / `(batch, d)` or matching tuple.
        """
        if was_single:
            if isinstance(attributions, tuple):
                return tuple(r.squeeze(0) for r in attributions)
            return attributions.squeeze(0)
        return attributions

    def _validate_inputs(
        self,
        inputs: torch.Tensor,
        ordering: list[int],
        baselines: torch.Tensor | None = None,
    ) -> None:
        """
        Sanity-check inputs before attribution begins.

        Checks that `inputs` is a non-empty tensor, that `ordering` length
        matches the feature dimension, and that `baselines` (if provided)
        has a compatible feature dimension and lives on the same device.

        Args:
            inputs: Input tensor.
            ordering: Topological ordering of feature indices.
            baselines: Optional baseline tensor.

        Raises:
            ValueError: On any shape or device mismatch.
        """
        if not isinstance(inputs, torch.Tensor):
            raise ValueError("inputs must be a torch.Tensor")

        if inputs.numel() == 0:
            raise ValueError("inputs cannot be empty")

        if len(ordering) != inputs.shape[-1]:
            raise ValueError("Ordering length must match the feature dimension")

        if baselines is not None:
            if not isinstance(baselines, torch.Tensor):
                raise ValueError("baselines must be a torch.Tensor")

            inputs_normalized, _ = self._format_input(inputs)
            baselines_normalized, _ = self._format_input(baselines)

            if inputs_normalized.shape[-1] != baselines_normalized.shape[-1]:
                raise ValueError(
                    f"inputs shape {inputs_normalized.shape} and baselines shape "
                    f"{baselines_normalized.shape} must have the same feature dimension"
                )

            if inputs.device != baselines.device:
                raise ValueError(
                    f"inputs device {inputs.device} and baselines device "
                    f"{baselines.device} must match"
                )

    @staticmethod
    def _resolve_ordering(
        ordering: list[int] | None, adjacency_matrix: torch.Tensor | None
    ) -> list[int]:
        """
        Return `ordering`, or derive a topological ordering from the graph.

        Args:
            ordering: Topological ordering of feature indices, or ``None``.
            adjacency_matrix: `(d, d)` tensor, `[i, j]` true if feature `i`
                causes feature `j`. Required when `ordering` is ``None``.

        Raises:
            ValueError: If both are ``None``, or the graph has a cycle.
        """
        if ordering is not None:
            return ordering
        if adjacency_matrix is None:
            raise ValueError(
                "Pass `adjacency_matrix` (preferred) or `ordering`: without a "
                "graph, the ordering defines which features each feature may "
                "depend on."
            )
        adjacency = torch.as_tensor(adjacency_matrix).bool().cpu()
        parents = {
            j: torch.nonzero(adjacency[:, j]).flatten().tolist()
            for j in range(adjacency.shape[1])
        }
        try:
            return list(TopologicalSorter(parents).static_order())
        except CycleError as e:
            raise ValueError(
                "adjacency_matrix contains a cycle; CIG/CEG need a DAG"
            ) from e

    @staticmethod
    def _require(value, name: str):
        """Raise a ``TypeError`` naming `name` if `value` is ``None``."""
        if value is None:
            raise TypeError(f"attribute() missing required argument: '{name}'")
        return value

    def _convergence_delta(
        self,
        input_normalized: torch.Tensor,
        baselines_normalized: torch.Tensor,
        target: int | None,
        total_attributions: torch.Tensor,
    ) -> torch.Tensor:
        """
        Completeness (convergence) delta.

        Measures how well the summed attributions account for the model
        output difference between `input` and the average baseline:

            delta = sum_i(attributions_i) - (f(input) - mean_b f(baseline_b))

        A delta close to zero indicates the completeness axiom holds; for
        finite `n_steps` / single-step sampling this is only approximately
        satisfied.

        Args:
            input_normalized: Input tensor `(batch, d)`.
            baselines_normalized: Baseline tensor `(n_baselines, d)`.
            target: Output index attributed, or ``None`` if the full model
                output was used.
            total_attributions: Direct + indirect attributions, summed
                shape `(batch, d, C)` (before any `C == 1` squeeze).

        Returns:
            Delta tensor of shape `(batch, C)`, matching
            `total_attributions` with the feature dimension reduced.
        """
        with torch.no_grad():
            f_input = self.forward_func(input_normalized)
            f_baselines = self.forward_func(baselines_normalized)
            if target is not None:
                f_input = f_input[:, target]
                f_baselines = f_baselines[:, target]

        expected_diff = f_input - f_baselines.mean(dim=0)
        attribution_sum = total_attributions.sum(dim=1)
        return attribution_sum - expected_diff.reshape(attribution_sum.shape)
