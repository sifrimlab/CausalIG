import torch

from causalig.attr._core.attribution import Attribution
from causalig.attr._utils.path import LinearPathSampler
from causalig.attr._utils.surrogate import compute_surrogate_gradients


def _compute_causal_gradients(
    forward_func,
    x_flat: torch.Tensor,
    C: int,
    target: int | None,
    surrogate,
    ordering: list[int],
    d: int,
    device: torch.device,
    adjacency_matrix: torch.Tensor | None,
    dim_chunk_size: int,
    has_closed_form_jacobian: bool,
    include_second_order_indirect: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """
    Shared forward pass for causal gradient attribution methods.

    Computes model gradients and surrogate-chained gradients for a flat
    batch of interpolated points `x_flat`. Both `CausalExpectedGradients`
    and `CausalExpectedGradientsMultiStep` call this; each then applies its
    own path-derivative and accumulation logic.

    Args:
        forward_func: Model callable accepting `(N, d)` and returning
            `(N, C)` or `(N,)`.
        x_flat: Flat batch of interpolated points `(N, d)`, must have
            ``requires_grad=True`` before calling.
        C: Number of output classes (1 for scalar / single-target).
        target: Output index when ``C == 1`` and model has multiple outputs.
        surrogate: Trained masked surrogate (autoencoder / regressor).
        ordering: Topological ordering of feature indices, length `d`.
        d: Number of input features.
        device: Compute device.
        adjacency_matrix: Optional causal adjacency tensor `(d, d)`.
        dim_chunk_size: Number of output dims per surrogate Jacobian pass.
        has_closed_form_jacobian: ``True`` if the surrogate provides an exact
            input Jacobian in closed form (e.g. `MaskedGlobalLinear`).
        include_second_order_indirect: If ``True`` also compute the
            second-hop indirect term `grads_f_ae_ae`.

    Returns:
        grads_f: Model gradients `(N, d, C)`.
        grads_f_ae: Surrogate-chained gradients `(N, d, C)`.
        x_ae_out_sel: Surrogate outputs `(N, d)`.
        grads_f_ae_ae: Second-order indirect term `(N, d, C)`, or ``None``.
    """
    N = x_flat.shape[0]

    # MODEL GRADIENTS
    if C != 1:
        grads = []
        for c in range(C):
            y_c = forward_func(x_flat)[:, c]
            grads_c = torch.autograd.grad(
                y_c, x_flat, grad_outputs=torch.ones_like(y_c), retain_graph=True
            )[0]
            grads.append(grads_c)
        grads_f = torch.stack(grads, dim=-1)  # (N, d, C)
    else:
        if target is not None:
            y = forward_func(x_flat)[:, target]  # (N,)
        else:
            y = forward_func(x_flat)  # (N, d_out)
        grads_f = torch.autograd.grad(
            y, x_flat, grad_outputs=torch.ones_like(y), retain_graph=False
        )[
            0
        ]  # (N, d)

    grads_f = grads_f.reshape(N, d, C).detach()  # (N, d, C)

    # SURROGATE GRADIENTS
    x_ae_out_sel = torch.zeros(N, d, device=device)
    grads_f_ae = torch.zeros(N, d, C, device=device)

    for dim_start in range(0, d, dim_chunk_size):
        dim_end = min(dim_start + dim_chunk_size, d)
        cs = dim_end - dim_start

        grads_ae_chunk, x_ae_out_sel_chunk = compute_surrogate_gradients(
            surrogate,
            x_flat,
            ordering,
            d,
            device,
            adjacency_matrix=adjacency_matrix,
            dim_start=dim_start,
            dim_chunk_size=cs,
            has_closed_form_jacobian=has_closed_form_jacobian,
        )

        x_ae_out_sel[:, dim_start : dim_start + cs] = x_ae_out_sel_chunk.detach()
        grads_ae_chunk = grads_ae_chunk.detach()  # (N, cs, d)
        grads_f_ae += (
            grads_f[:, dim_start : dim_start + cs].unsqueeze(-2)
            * grads_ae_chunk.unsqueeze(-1)
        ).sum(dim=1)

    if not include_second_order_indirect:
        return grads_f, grads_f_ae, x_ae_out_sel, None

    grads_f_ae_ae = torch.zeros(N, d, C, device=device)
    for mid_start in range(0, d, dim_chunk_size):
        mid_end = min(mid_start + dim_chunk_size, d)
        mcs = mid_end - mid_start

        grads_ae_mid_chunk, _ = compute_surrogate_gradients(
            surrogate,
            x_flat,
            ordering,
            d,
            device,
            adjacency_matrix=adjacency_matrix,
            dim_start=mid_start,
            dim_chunk_size=mcs,
            has_closed_form_jacobian=has_closed_form_jacobian,
        )
        grads_ae_mid_chunk = grads_ae_mid_chunk.detach()  # (N, mcs, d)
        grads_f_ae_ae += (
            grads_f_ae[:, mid_start:mid_end].unsqueeze(-2)
            * grads_ae_mid_chunk.unsqueeze(-1)
        ).sum(dim=1)

    return grads_f, grads_f_ae, x_ae_out_sel, grads_f_ae_ae


class CausalExpectedGradients(Attribution):
    """
    Causal Expected Gradients (CEG) — single integration step per baseline.

    For each baseline *b* sampled from `baselines`, a single interpolated
    point ``x_alpha = b + alpha * (input - b)`` (with ``alpha`` drawn
    uniformly) is used to estimate the gradient contribution. Attributions
    are averaged over all baselines.

    This is a cheaper alternative to `CausalExpectedGradientsMultiStep` as
    described in the original paper introducing Expected Gradients.
    Use the multi-step variant when higher integration accuracy is required.

    Attributions are decomposed into:

    - **Direct effects**: gradient along the residual path
      ``h = x - surrogate(x)``.
    - **Indirect effects**: gradient along the surrogate output path,
      capturing effects mediated through downstream features.

    Args:
        forward_func: Model to attribute. Must accept `(batch, d)` float
            tensors and return `(batch, C)` or `(batch,)` tensors.
    """

    def attribute(
        self,
        input: torch.Tensor,
        surrogate,
        ordering: list[int] | None = None,
        baselines: torch.Tensor | None = None,
        target: int = None,
        separate_effects: bool = False,
        adjacency_matrix: torch.Tensor | None = None,
        baselines_chunk_size: int | None = None,
        dim_chunk_size: int | None = None,
        has_closed_form_jacobian: bool = False,
        include_second_order_indirect: bool = False,
        return_convergence_delta: bool = False,
        seed: int | None = None,
    ) -> (
        torch.Tensor
        | tuple[torch.Tensor, torch.Tensor]
        | tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ):
        """
        Compute CEG attributions (single-step per baseline).

        Args:
            input: Input tensor `(d,)` or `(batch, d)`.
            surrogate: Trained masked surrogate (autoencoder / regressor).
                Must accept `(batch, d)` and return `(batch, d)`.
            ordering: Topological ordering of feature indices, length `d`.
                Only needed when `adjacency_matrix` is ``None``; otherwise
                it is derived from the graph.
            baselines: Tensor of baselines `(n_baselines, d)` or `(d,)`.
                Each row is one reference point; attributions are averaged
                over all rows.
            target: Output index to attribute for multi-class models.
                ``None`` attributes all outputs simultaneously.
            separate_effects: If ``True`` return ``(direct, indirect)``
                instead of their sum.
            adjacency_matrix: Boolean tensor `(d, d)` where `[i, j]` means
                feature *i* causally influences feature *j*.
            baselines_chunk_size: Number of baselines to process at a time.
                ``None`` processes all baselines in one pass.
            dim_chunk_size: Number of feature dimensions to process at a
                time when computing surrogate Jacobians. ``None`` uses all.
            has_closed_form_jacobian: Set ``True`` when the surrogate provides
                an exact input Jacobian in closed form (e.g.
                `MaskedGlobalLinear`) to use the faster analytic Jacobian path
                instead of autograd.
            include_second_order_indirect: If ``True``, add a second-hop
                indirect term that captures effects propagated through two
                successive surrogate applications. Defaults to ``False``.
            return_convergence_delta: If ``True``, also return a completeness
                delta: ``sum(attributions) - (f(input) - mean_b f(baseline_b))``.
                A delta near zero confirms the attributions satisfy the
                completeness axiom.
            seed: Random seed for stochastic alpha sampling.

        Returns:
            Attribution tensor matching `input` shape, or tuple
            ``(direct, indirect)`` when ``separate_effects=True``. If
            ``return_convergence_delta=True``, the delta tensor is appended
            as the last element of the returned tuple (attributions are
            wrapped in a tuple if not already one).
        """
        baselines = self._require(baselines, "baselines")
        ordering = self._resolve_ordering(ordering, adjacency_matrix)
        self._validate_inputs(input, ordering, baselines)

        input_normalized, was_single = self._format_input(input)
        baselines_normalized, _ = self._format_input(baselines)

        batch = input_normalized.size(0)
        d = input_normalized.size(-1)
        device = input_normalized.device

        # Ensure baseline lives on the same device as the input
        baselines_normalized = baselines_normalized.to(device)

        x, alphas = LinearPathSampler().sample_pairwise_stochastic(
            baselines_normalized, input_normalized, n_steps=1, seed=seed
        )
        x = x.reshape(len(baselines_normalized), batch, d)
        alphas = alphas.reshape(len(baselines_normalized))

        if target is None:
            with torch.no_grad():
                C = self.forward_func(input_normalized[:1]).shape[-1]
        else:
            C = 1

        baselines_chunk_size = baselines_chunk_size or baselines_normalized.size(0)
        dim_chunk_size = dim_chunk_size or d

        # h at each baseline: residual not captured by the surrogate
        baseline_ae_out = compute_surrogate_gradients(
            surrogate,
            x_flat=baselines_normalized,
            ordering=ordering,
            d=d,
            device=device,
            adjacency_matrix=adjacency_matrix,
            skip_gradients=True,
            has_closed_form_jacobian=has_closed_form_jacobian,
        )[1]
        h_baselines = baselines_normalized - baseline_ae_out

        direct_accum = torch.zeros((batch, d, C), device=device)
        indirect_accum = torch.zeros((batch, d, C), device=device)

        for b_start in range(0, baselines_normalized.size(0), baselines_chunk_size):
            b_end = min(b_start + baselines_chunk_size, baselines_normalized.size(0))
            b_chunk_size = b_end - b_start

            x_chunk = x[b_start:b_end]  # (chunk, batch, d)
            x_flat = x_chunk.reshape(-1, d).clone()  # (chunk*batch, d)
            x_flat.requires_grad_(True)

            grads_f, grads_f_ae, x_ae_out_sel, grads_f_ae_ae = (
                _compute_causal_gradients(
                    self.forward_func,
                    x_flat,
                    C,
                    target,
                    surrogate,
                    ordering,
                    d,
                    device,
                    adjacency_matrix,
                    dim_chunk_size,
                    has_closed_form_jacobian,
                    include_second_order_indirect,
                )
            )

            grads_f = grads_f.reshape(b_chunk_size, batch, d, C)  # (chunk, batch, d, C)
            grads_f_ae = grads_f_ae.reshape(
                b_chunk_size, batch, d, C
            )  # (chunk, batch, d, C)
            x_ae_out_sel = x_ae_out_sel.reshape(
                b_chunk_size, batch, d
            )  # (chunk, batch, d)

            # PATH GRADIENTS
            h = x_chunk - x_ae_out_sel

            # Linear approximation to dh/dalpha between baseline and interpolated point
            grads_h = h - h_baselines[b_start:b_end].unsqueeze(1)
            # Full dxi/dalpha
            grads_x_alpha = x_chunk - baselines_normalized[b_start:b_end].unsqueeze(1)

            # Divide by distance of path
            grads_h /= alphas[b_start:b_end].reshape(-1, 1, 1)
            grads_x_alpha /= alphas[b_start:b_end].reshape(-1, 1, 1)

            # COMPUTE THE EXPECTED GRADIENTS
            # Direct effects: average over baselines, keep per-sample
            direct_accum += (grads_f * grads_h.unsqueeze(-1)).mean(dim=0)
            # Indirect effects: average over baselines, keep per-sample
            if grads_f_ae_ae is not None:
                indirect_accum += (
                    grads_f_ae * grads_h.unsqueeze(-1)
                    + grads_f_ae_ae * grads_x_alpha.unsqueeze(-1)
                ).mean(dim=0)
            else:
                indirect_accum += (grads_f_ae * grads_x_alpha.unsqueeze(-1)).mean(dim=0)

        # Normalize by number of chunks
        n_chunks = (
            baselines_normalized.size(0) + baselines_chunk_size - 1
        ) // baselines_chunk_size
        direct_effects = direct_accum / n_chunks
        indirect_effects = indirect_accum / n_chunks

        if return_convergence_delta:
            delta = self._convergence_delta(
                input_normalized,
                baselines_normalized,
                target,
                direct_effects + indirect_effects,
            )
            if C == 1:
                delta = delta.squeeze(-1)

        if C == 1:
            direct_effects = direct_effects.squeeze(-1)
            indirect_effects = indirect_effects.squeeze(-1)

        result = (
            (direct_effects, indirect_effects)
            if separate_effects
            else direct_effects + indirect_effects
        )
        result = self._format_output(result, was_single)

        if return_convergence_delta:
            delta = self._format_output(delta, was_single)
            return (*result, delta) if separate_effects else (result, delta)
        return result


class CausalExpectedGradientsMultiStep(Attribution):
    """
    Causal Expected Gradients — multi-step integration, averaged over baselines (CEG-MS).

    For each baseline *b* in `baselines`, the causal gradient is integrated
    along the full linear path from *b* to `input` using `n_steps` Riemann
    steps, mirroring `CausalIntegratedGradients`. Attributions are then
    averaged over all baselines.

    `CausalIntegratedGradients` is the special case with a single baseline.

    Attributions are decomposed into:

    - **Direct effects**: accumulated gradient along ``h = x - surrogate(x)``.
    - **Indirect effects**: accumulated gradient along the surrogate output,
      capturing effects mediated through causally downstream features.

    Args:
        forward_func: Model to attribute. Must accept `(batch, d)` float
            tensors and return `(batch, C)` or `(batch,)` tensors.
    """

    def attribute(
        self,
        input: torch.Tensor,
        surrogate,
        ordering: list[int] | None = None,
        baselines: torch.Tensor | None = None,
        n_steps: int = 50,
        target: int = None,
        separate_effects: bool = False,
        adjacency_matrix: torch.Tensor | None = None,
        baselines_chunk_size: int | None = None,
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
        Compute CEG-MS attributions.

        Args:
            input: Input tensor `(d,)` or `(batch, d)`.
            surrogate: Trained masked surrogate (autoencoder / regressor).
                Must accept `(batch, d)` and return `(batch, d)`.
            ordering: Topological ordering of feature indices, length `d`.
                Only needed when `adjacency_matrix` is ``None``; otherwise
                it is derived from the graph.
            baselines: Baseline distribution `(n_baselines, d)` or `(d,)`.
                Attributions are averaged over all baselines.
            n_steps: Number of Riemann integration steps per baseline path.
                More steps give a more accurate approximation. Default 50.
            target: Output index to attribute for multi-class models.
                ``None`` attributes all outputs simultaneously.
            separate_effects: If ``True`` return ``(direct, indirect)``
                instead of their sum.
            adjacency_matrix: Boolean tensor `(d, d)` where `[i, j]` means
                feature *i* causally influences feature *j*.
            baselines_chunk_size: Number of baselines to process per pass.
                ``None`` processes all baselines at once.
            step_chunk_size: Number of integration steps to process per
                pass. Must be > 1 or ``None``. ``None`` processes all steps
                at once.
            dim_chunk_size: Number of feature dimensions per surrogate
                Jacobian pass. ``None`` uses all dimensions at once.
            has_closed_form_jacobian: Set ``True`` when the surrogate provides
                an exact input Jacobian in closed form (e.g.
                `MaskedGlobalLinear`) to use the faster analytic Jacobian path
                instead of autograd.
            include_second_order_indirect: If ``True``, add a second-hop
                indirect term that captures effects propagated through two
                successive surrogate applications. Defaults to ``False``.
            return_convergence_delta: If ``True``, also return a completeness
                delta: ``sum(attributions) - (f(input) - mean_b f(baseline_b))``.
                A delta near zero confirms the attributions satisfy the
                completeness axiom; it shrinks as `n_steps` increases.

        Returns:
            Attribution tensor matching `input` shape, or tuple
            ``(direct, indirect)`` when ``separate_effects=True``. If
            ``return_convergence_delta=True``, the delta tensor is appended
            as the last element of the returned tuple (attributions are
            wrapped in a tuple if not already one).
        """
        baselines = self._require(baselines, "baselines")
        ordering = self._resolve_ordering(ordering, adjacency_matrix)
        self._validate_inputs(input, ordering, baselines)
        assert (
            step_chunk_size is None or step_chunk_size > 1
        ), "step_chunk_size must be > 1 or None"

        input_normalized, was_single = self._format_input(input)
        baselines_normalized, _ = self._format_input(baselines)

        batch = input_normalized.size(0)
        d = input_normalized.size(-1)
        device = input_normalized.device

        # Ensure baseline lives on the same device as the input
        baselines_normalized = baselines_normalized.to(device)

        x = LinearPathSampler().sample(baselines_normalized, input_normalized, n_steps)
        step_lengths = 1.0 / (n_steps - 1)

        if target is None:
            with torch.no_grad():
                C = self.forward_func(input_normalized[:1]).shape[-1]
        else:
            C = 1

        direct_accum = torch.zeros((batch, d, C), device=device)
        indirect_accum = torch.zeros((batch, d, C), device=device)

        baselines_chunk_size = baselines_chunk_size or baselines_normalized.size(0)
        step_chunk_size = step_chunk_size or n_steps
        dim_chunk_size = dim_chunk_size or d

        for b_start in range(0, baselines_normalized.size(0), baselines_chunk_size):
            b_end = min(b_start + baselines_chunk_size, baselines_normalized.size(0))
            b_chunk_size = b_end - b_start

            # Merge baseline chunk into batch dim: (n_steps, b_chunk*batch, d)
            x_b_chunk = x[:, b_start:b_end].reshape(n_steps, b_chunk_size * batch, d)

            prev_x = None
            prev_h = None

            for step_start in range(0, n_steps, step_chunk_size):
                step_end = min(step_start + step_chunk_size, n_steps)
                steps = step_end - step_start

                x_chunk = x_b_chunk[step_start:step_end]  # (steps, b_chunk*batch, d)
                x_flat = x_chunk.reshape(steps * b_chunk_size * batch, d).clone()
                x_flat.requires_grad_(True)

                grads_f, grads_f_ae, x_ae_out_sel, grads_f_ae_ae = (
                    _compute_causal_gradients(
                        self.forward_func,
                        x_flat,
                        C,
                        target,
                        surrogate,
                        ordering,
                        d,
                        device,
                        adjacency_matrix,
                        dim_chunk_size,
                        has_closed_form_jacobian,
                        include_second_order_indirect,
                    )
                )

                grads_f = grads_f.reshape(
                    steps, b_chunk_size * batch, d, C
                )  # (steps, b_chunk*batch, d, C)
                grads_f_ae = grads_f_ae.reshape(
                    steps, b_chunk_size * batch, d, C
                )  # (steps, b_chunk*batch, d, C)
                x_ae_out_sel = x_ae_out_sel.reshape(
                    steps, b_chunk_size * batch, d
                )  # (steps, b_chunk*batch, d)
                if grads_f_ae_ae is not None:
                    grads_f_ae_ae = grads_f_ae_ae.reshape(
                        steps, b_chunk_size * batch, d, C
                    )

                # PATH GRADIENTS
                h = x_chunk - x_ae_out_sel

                # Full dxi/dalpha — finite differences across steps
                grads_h = h[1:] - h[:-1]
                grads_x_alpha = x_chunk[1:] - x_chunk[:-1]

                # Fix for first step
                if prev_x is None:
                    grads_h = torch.cat([grads_h[0:1], grads_h], dim=0)
                    grads_x_alpha = torch.cat(
                        [grads_x_alpha[0:1], grads_x_alpha], dim=0
                    )
                else:
                    grads_h = torch.cat([h[0:1] - prev_h, grads_h], dim=0)
                    grads_x_alpha = torch.cat(
                        [x_chunk[0:1] - prev_x, grads_x_alpha], dim=0
                    )

                grads_h = grads_h / step_lengths
                grads_x_alpha = grads_x_alpha / step_lengths

                prev_x = x_chunk[-1].detach()
                prev_h = h[-1].detach()

                # COMPUTE THE INTEGRATED GRADIENTS
                # Direct effects: sum over path steps, keep per-sample
                direct_accum += (
                    (grads_f * grads_h.unsqueeze(-1))
                    .sum(dim=0)
                    .reshape(b_chunk_size, batch, d, C)
                    .sum(dim=0)
                )

                # Indirect effects
                if grads_f_ae_ae is not None:
                    indirect_accum += (
                        (
                            grads_f_ae * grads_h.unsqueeze(-1)
                            + grads_f_ae_ae * grads_x_alpha.unsqueeze(-1)
                        )
                        .sum(dim=0)
                        .reshape(b_chunk_size, batch, d, C)
                        .sum(dim=0)
                    )
                else:
                    indirect_accum += (
                        (grads_f_ae * grads_x_alpha.unsqueeze(-1))
                        .sum(dim=0)
                        .reshape(b_chunk_size, batch, d, C)
                        .sum(dim=0)
                    )

        # Normalize by number of steps and baselines
        direct_effects = direct_accum / (n_steps * baselines_normalized.size(0))
        indirect_effects = indirect_accum / (n_steps * baselines_normalized.size(0))

        if return_convergence_delta:
            delta = self._convergence_delta(
                input_normalized,
                baselines_normalized,
                target,
                direct_effects + indirect_effects,
            )
            if C == 1:
                delta = delta.squeeze(-1)

        if C == 1:
            direct_effects = direct_effects.squeeze(-1)
            indirect_effects = indirect_effects.squeeze(-1)

        result = (
            (direct_effects, indirect_effects)
            if separate_effects
            else direct_effects + indirect_effects
        )
        result = self._format_output(result, was_single)

        if return_convergence_delta:
            delta = self._format_output(delta, was_single)
            return (*result, delta) if separate_effects else (result, delta)
        return result
