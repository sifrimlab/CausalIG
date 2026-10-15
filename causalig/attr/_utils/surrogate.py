"""
Autoencoder interface and gradient computation utilities for attribution methods.
"""

from abc import ABC, abstractmethod

import numpy as np
import torch

# Mask value for autoencoder inputs
MASK_VAL = 0.0


class ModelInterface(ABC):
    """
    Abstract interface for models used in attribution methods.

    This interface decouples attribution methods from specific model
    implementations, allowing for different model types. Subclassing it is
    optional: any callable with the same behaviour can be used as a surrogate.

    A surrogate maps `(batch, d)` to `(batch, d)`, where output column `j`
    predicts feature `j` from its causal parents. On the default autograd
    path, features that are not parents of `j` arrive set to `MASK_VAL`.
    """

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.

        Args:
            x: Input tensor of shape (batch, features)

        Returns:
            Reconstructed output tensor of shape (batch, features)
        """
        raise NotImplementedError("Subclasses must implement forward()")

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Make the interface callable."""
        return self.forward(x)


def compute_surrogate_gradients(
    autoencoder: ModelInterface,
    x_flat: torch.Tensor,
    ordering: list[int],
    d: int,
    device: torch.device,
    adjacency_matrix: torch.Tensor | None = None,
    skip_gradients: bool = False,
    dim_start: int | None = 0,
    dim_chunk_size: int | None = None,
    has_closed_form_jacobian: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Evaluate a surrogate and its input Jacobian for a chunk of output features.

    For every output feature `j` in `[dim_start, dim_start + dim_chunk_size)`,
    returns the surrogate's prediction of `x_j` and its gradient with respect
    to the inputs `j` may depend on: its parents in `adjacency_matrix` or, if
    no adjacency is given, every feature before `j` in `ordering`. Features
    without parents get a zero prediction and a zero Jacobian row.

    Two strategies are supported:

    - Autograd (default): for each child `j` the input is copied, every
      non-parent of `j` is set to `MASK_VAL`, the surrogate is called on the
      masked copies and column `j` of its output is differentiated.
    - Closed form (``has_closed_form_jacobian=True``): the surrogate is called
      once on the unmasked input and must provide
      ``compute_input_jacobian(x, dim_start, dim_end) -> (N, chunk, d)``.
      The surrogate itself must then ensure output `j` reads only parents.

    Args:
        autoencoder: The surrogate, a callable mapping `(N, d)` to `(N, d)`.
        x_flat: Points at which to evaluate it, `(N, d)`.
        ordering: Topological ordering of feature indices, used to build the
            dependency mask only when `adjacency_matrix` is ``None``.
        d: Number of features.
        device: Device for the outputs.
        adjacency_matrix: Optional `(d, d)` tensor, `[i, j]` true if feature
            `i` causes feature `j`.
        skip_gradients: If ``True`` only compute the predictions and return
            ``None`` for the Jacobian.
        dim_start: First output feature of the chunk.
        dim_chunk_size: Number of output features in the chunk. ``None``
            means all remaining features.
        has_closed_form_jacobian: Use the surrogate's
            ``compute_input_jacobian`` instead of autograd.

    Returns:
        ``(jacobian, predictions)``: the masked Jacobian
        `(N, chunk, d)` with entry `[n, j - dim_start, i]` equal to
        ``d surrogate(x)[n, j] / d x[n, i]`` (or ``None`` when
        `skip_gradients`), and the detached predictions `(N, chunk)`.

    Raises:
        TypeError: If `has_closed_form_jacobian` is ``True`` and the
            surrogate has no ``compute_input_jacobian``.
        ValueError: If `has_closed_form_jacobian` is ``True`` and the
            Jacobian it returns is non-zero for a feature that is not a
            parent of the output. Only the reported Jacobian is checked, so
            a surrogate whose ``compute_input_jacobian`` misreports its
            forward pass is not caught. The check is skipped when
            `skip_gradients` is ``True``.
    """
    N = x_flat.shape[0]

    dim_chunk_size = dim_chunk_size or d
    dim_end = min(dim_start + dim_chunk_size, d)
    dim_chunk_size = dim_end - dim_start

    # ---- mask construction ----
    if adjacency_matrix is not None:
        mask_block_full = adjacency_matrix.T.to(device).bool()
    else:
        mask_block_full = torch.tril(
            torch.ones(d, d, dtype=torch.bool, device=device),
            diagonal=-1,
        )
        inverse = np.argsort(ordering)
        mask_block_full = mask_block_full[inverse][:, inverse]

    mask_block = mask_block_full[dim_start:dim_end]  # [chunk, d]
    zero_row_mask_block = (mask_block == 0).all(dim=1)  # [chunk]

    # ---- outputs ----
    sel = torch.zeros(N, dim_chunk_size, device=device, dtype=x_flat.dtype)

    # ---- gradients ----
    grads_ae_out = (
        None
        if skip_gradients
        else torch.zeros(N, dim_chunk_size, d, device=device, dtype=x_flat.dtype)
    )
    # ============================================================
    # CLOSED-FORM JACOBIAN
    # ============================================================
    if has_closed_form_jacobian:
        if skip_gradients:
            with torch.no_grad():
                x_out = autoencoder(x_flat)  # [N, d]
        else:
            if not hasattr(autoencoder, "compute_input_jacobian"):
                raise TypeError(
                    "has_closed_form_jacobian=True requires the surrogate to "
                    "implement compute_input_jacobian(x, dim_start, dim_end); "
                    f"{type(autoencoder).__name__} does not."
                )
            with torch.no_grad():
                grads_chunk = autoencoder.compute_input_jacobian(
                    x_flat,
                    dim_start,
                    dim_end,
                )  # [N, chunk, d]

            # inputs are not masked on this path, so a non-zero derivative
            # w.r.t. a non-parent means the surrogate reads beyond the parents
            # (parentless children are zeroed below and are not checked)
            leaked = (
                (grads_chunk != 0)
                & ~mask_block.unsqueeze(0)
                & ~zero_row_mask_block.view(1, -1, 1)
            )
            if leaked.any():
                _, j_local, i = leaked.nonzero()[0].tolist()
                raise ValueError(
                    "has_closed_form_jacobian=True but the surrogate's output "
                    f"{dim_start + j_local} depends on feature {i}, which is not "
                    "one of its parents. Use a surrogate built on the same "
                    "graph, or has_closed_form_jacobian=False to mask inputs."
                )

            grads_ae_out.copy_(grads_chunk)

            # apply dependency mask
            grads_ae_out *= mask_block.unsqueeze(0)

            x_out = autoencoder(x_flat)  # [N, d]

        # select outputs per-dimension (no reshape tricks)
        for j_local, j in enumerate(range(dim_start, dim_end)):
            if not zero_row_mask_block[j_local]:
                sel[:, j_local] = x_out[:, j]

        if not skip_gradients:
            grads_ae_out[:, zero_row_mask_block, :] = 0.0

        return grads_ae_out, sel.detach()

    # ============================================================
    # AUTOGRAD PATH (keeps structure but fixes bugs)
    # ============================================================
    else:
        mask_block_reduced = mask_block[~zero_row_mask_block]  # [chunk_reduced, d]
        d_chunk_reduced = mask_block_reduced.shape[0]

        x_ae = x_flat.repeat_interleave(
            d_chunk_reduced, dim=0
        )  # (N * chunk_reduced, d)

        mask = mask_block_reduced.repeat(N, 1)  # (N * chunk_reduced, d)

        if skip_gradients:
            with torch.no_grad():
                x_ae_out = autoencoder(x_ae.masked_fill(~mask, MASK_VAL))
        else:
            x_ae = x_ae.detach().requires_grad_(True)
            x_ae = x_ae.masked_fill(~mask, MASK_VAL)

            x_ae_out = autoencoder(x_ae)

        # ---- output selection ----
        # x_ae_out is [N * d_chunk_reduced, d], laid out sample-major: sample n
        # occupies rows [n * d_chunk_reduced : (n + 1) * d_chunk_reduced], one row
        # per predictable child (children with >=1 parent). Fill sel column-wise
        # so the mapping is correct even when d_chunk_reduced != dim_chunk_size
        # (i.e. when some child in the chunk is a parentless root). Parentless
        # children keep sel == 0.
        idx_full = (~zero_row_mask_block).nonzero(as_tuple=False).squeeze(1)

        for i, j_local in enumerate(idx_full):
            j_local = int(j_local)
            rows = torch.arange(N, device=device) * d_chunk_reduced + i
            sel[:, j_local] = x_ae_out[rows, dim_start + j_local]

        # ---- gradients ----
        if skip_gradients:
            return None, sel.detach()

        gmask = torch.zeros(d_chunk_reduced, d, device=device, dtype=x_ae_out.dtype)
        gmask[torch.arange(d_chunk_reduced), idx_full + dim_start] = 1.0
        gmask = gmask.repeat(N, 1)

        grads_ae = torch.autograd.grad(
            outputs=x_ae_out,
            inputs=x_ae,
            grad_outputs=gmask,
            retain_graph=False,
        )[0]

        grads_ae = grads_ae.detach()
        grads_ae[~mask] = 0.0

        # ---- restore full shape ----
        grads_full = torch.zeros(
            N * dim_chunk_size, d, device=device, dtype=grads_ae.dtype
        )
        grads_full[~zero_row_mask_block.repeat(N)] = grads_ae

        grads_ae_out = grads_full.view(N, dim_chunk_size, d)

        return grads_ae_out, sel.detach()
