import torch

from causalig.attr._core.attribution import Attribution
from causalig.attr._core.causal_expected_gradients import (
    CausalExpectedGradientsMultiStep,
)


class CausalIntegratedGradients(Attribution):
    """
    Causal Integrated Gradients (CIG).

    CIG is the special case of `CausalExpectedGradientsMultiStep` with a
    single baseline: it integrates the causal gradient along the linear path
    from `baseline` to `input`, conditioning on a surrogate at every step to
    separate direct and indirect effects.

    **Direct effect** of feature *i*: contribution along the residual path
    ``h = x - surrogate(x)``, capturing the portion of the gradient that is
    *not* mediated through the surrogate.

    **Indirect effect** of feature *i*: contribution along the surrogate
    output path, capturing effects propagated through causally downstream
    features.

    This class delegates entirely to `CausalExpectedGradientsMultiStep` and
    exists as a convenience wrapper with the single-baseline API
    (``baseline`` instead of ``baselines``).

    Args:
        forward_func: Model to attribute. Must accept `(batch, d)` float
            tensors and return `(batch, C)` or `(batch,)` tensors.
    """

    def __init__(self, forward_func):
        super().__init__(forward_func)
        self._ceg = CausalExpectedGradientsMultiStep(forward_func)

    def attribute(
        self,
        input: torch.Tensor,
        surrogate,
        ordering: list[int] | None = None,
        baseline: torch.Tensor | None = None,
        n_steps: int = 50,
        target: int = None,
        separate_effects: bool = False,
        adjacency_matrix: torch.Tensor | None = None,
        step_chunk_size: int | None = None,
        dim_chunk_size: int | None = None,
        has_closed_form_jacobian: bool = False,
        include_second_order_indirect: bool = False,
        return_convergence_delta: bool = False,
    ) -> (
        torch.Tensor
        | tuple[torch.Tensor, torch.Tensor]
        | tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ):
        """
        Compute CIG attributions.

        Args:
            input: Input tensor `(d,)` or `(batch, d)`.
            surrogate: Trained masked surrogate (autoencoder / regressor).
                Must accept `(batch, d)` and return `(batch, d)`.
            ordering: Topological ordering of feature indices, length `d`.
                Only needed when `adjacency_matrix` is ``None``; otherwise
                it is derived from the graph.
            baseline: Baseline tensor `(d,)`, shared by every input in the
                batch. The integration path runs from `baseline` to `input`.
            n_steps: Number of Riemann steps along the integration path.
                More steps give a more accurate approximation. Default 50.
            target: Output index to attribute for multi-class models.
                If ``None`` all outputs are attributed simultaneously.
            separate_effects: If ``True`` return ``(direct, indirect)``
                instead of their sum.
            adjacency_matrix: Boolean tensor `(d, d)` where entry `[i, j]`
                means feature *i* causally influences feature *j*. Used to
                restrict which features the surrogate conditions on.
            step_chunk_size: Process this many path steps at a time to
                reduce peak memory. ``None`` processes all steps at once.
            dim_chunk_size: Process this many feature dimensions at a time
                when computing surrogate Jacobians. ``None`` processes all
                dimensions at once.
            has_closed_form_jacobian: Set ``True`` when the surrogate provides
                an exact input Jacobian in closed form (e.g.
                `MaskedGlobalLinear`), enabling a faster analytic Jacobian
                computation path instead of autograd.
            include_second_order_indirect: If ``True``, add a second-hop
                term that captures effects mediated through two surrogate
                applications. Defaults to ``False``.
            return_convergence_delta: If ``True``, also return a completeness
                delta: ``sum(attributions) - (f(input) - f(baseline))``. A
                delta near zero confirms the attributions satisfy the
                completeness axiom; it shrinks as `n_steps` increases.

        Returns:
            Attribution tensor of shape matching `input`, or tuple
            ``(direct, indirect)`` when ``separate_effects=True``. If
            ``return_convergence_delta=True``, the delta tensor is appended
            as the last element of the returned tuple (attributions are
            wrapped in a tuple if not already one).
        """
        baseline = self._require(baseline, "baseline")
        return self._ceg.attribute(
            input=input,
            surrogate=surrogate,
            ordering=ordering,
            baselines=baseline.unsqueeze(0),
            n_steps=n_steps,
            target=target,
            separate_effects=separate_effects,
            adjacency_matrix=adjacency_matrix,
            step_chunk_size=step_chunk_size,
            dim_chunk_size=dim_chunk_size,
            has_closed_form_jacobian=has_closed_form_jacobian,
            include_second_order_indirect=include_second_order_indirect,
            return_convergence_delta=return_convergence_delta,
        )
