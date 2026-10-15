"""
MaskedGlobalLinear surrogate model and its training loop.
"""

import logging
import math
import os
from collections.abc import Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, random_split


# ============================================================================ #
# Basis expansion
# ============================================================================ #
class BasisExpansion(nn.Module):
    """Per-feature, scale-adaptive basis expansion on standardized inputs.

    Components, concatenated along the last dim::

        polynomial:       z, z^2, ..., z^polynomial_degree         (D terms)
        truncated power:  ReLU(z - c_k)^p   for k=1..n_knots       (K terms)
        Fourier:          sin(f·z), cos(f·z)  for f in frequencies (2F terms)

    Total `n_basis = D + K + 2F`.

    Inputs are standardized as ``z = (x - x_loc) / x_scale`` and clipped to
    ``[-activation_clip, activation_clip]`` before expansion.

    Args:
        n_features: Number of input features `N`.
        n_knots: Number of truncated-power terms `K` per feature.
        knot_power: Power `p` of the truncated-power terms.
        polynomial_degree: Highest polynomial power `D`.
        fourier_features: Number of sin/cos frequency pairs `F`; frequency
            `k` is ``k * pi / fourier_scale``.
        fourier_scale: Period scale of the Fourier terms.
        learnable_knots: Train the knot positions instead of keeping them
            fixed.
        knot_init: Optional `(N, K)` initial knot positions. Defaults to `K`
            evenly spaced values on ``[-2, 2]`` for every feature.
        x_loc: Per-feature centre `(N,)` for standardization. Defaults to 0.
        x_scale: Per-feature scale `(N,)` for standardization. Defaults to 1.
        activation_clip: Clip standardized inputs to this magnitude; ``None``
            disables clipping. The derivative is zero where clipping is
            active.
    """

    def __init__(
        self,
        n_features: int,
        n_knots: int = 4,
        knot_power: int = 3,
        polynomial_degree: int = 2,
        fourier_features: int = 0,
        fourier_scale: float = 2.0,
        learnable_knots: bool = True,
        knot_init: torch.Tensor | None = None,
        x_loc: torch.Tensor | None = None,
        x_scale: torch.Tensor | None = None,
        activation_clip: float | None = 8.0,
    ) -> None:
        super().__init__()
        self.n_features = n_features
        self.n_knots = n_knots
        self.knot_power = knot_power
        self.polynomial_degree = polynomial_degree
        self.fourier_features = fourier_features
        self.activation_clip = activation_clip

        # standardization buffers (chain-rule applied in derivative)
        self.register_buffer(
            "x_loc",
            (x_loc if x_loc is not None else torch.zeros(n_features)).float(),
        )
        self.register_buffer(
            "x_scale",
            (x_scale if x_scale is not None else torch.ones(n_features))
            .float()
            .clamp_min(1e-6),
        )

        if n_knots > 0:
            if knot_init is None:
                base = torch.linspace(-2.0, 2.0, steps=n_knots)
                knots = base.unsqueeze(0).expand(n_features, -1).contiguous()
            else:
                knots = knot_init.float()
            if learnable_knots:
                self.knots = nn.Parameter(knots)
            else:
                self.register_buffer("knots", knots)

        if fourier_features > 0:
            freqs = torch.arange(1, fourier_features + 1).float() * (
                math.pi / fourier_scale
            )
            self.register_buffer("freqs", freqs)

        self.n_basis = polynomial_degree + n_knots + 2 * fourier_features
        if self.n_basis < 1:
            raise ValueError("BasisExpansion must have at least one component.")

    def _z(self, x: torch.Tensor) -> torch.Tensor:
        z = (x - self.x_loc) / self.x_scale
        if self.activation_clip is not None:
            z = z.clamp(min=-self.activation_clip, max=self.activation_clip)
        return z

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Expand `x` of shape `(B, N)` into basis values `(B, N, n_basis)`."""
        z = self._z(x)
        bases = []

        # polynomial: build z, z^2, ... incrementally
        if self.polynomial_degree > 0:
            zd = z
            for _ in range(self.polynomial_degree):
                bases.append(zd)
                zd = zd * z

        # truncated power
        if self.n_knots > 0:
            diff = F.relu(z.unsqueeze(-1) - self.knots.unsqueeze(0))
            tp = diff.pow(self.knot_power)
            bases.extend(tp.unbind(-1))

        # Fourier
        if self.fourier_features > 0:
            wz = z.unsqueeze(-1) * self.freqs  # [B, N, F]
            bases.extend(torch.sin(wz).unbind(-1))
            bases.extend(torch.cos(wz).unbind(-1))

        return torch.stack(bases, dim=-1)  # [B, N, n_basis]

    def derivative(self, x: torch.Tensor) -> torch.Tensor:
        """Exact derivative of each basis value with respect to `x`, `(B, N, n_basis)`."""
        z_raw = (x - self.x_loc) / self.x_scale
        z = self._z(x)
        inv_s = 1.0 / self.x_scale  # [N]
        dbases = []

        # polynomial: d/dx z^d = d * z^(d-1) / scale
        zd_minus_1 = torch.ones_like(z)  # z^0
        for d in range(1, self.polynomial_degree + 1):
            dbases.append(d * zd_minus_1 * inv_s)
            zd_minus_1 = zd_minus_1 * z  # z^(d-1) -> z^d

        # truncated power: d/dx ReLU(z-c)^p = p * ReLU(z-c)^(p-1) * 1[z>c] / scale
        # For p >= 2 the indicator is implicit (ReLU^(p-1) is 0 below the knot);
        # for p == 1 it must be applied explicitly.
        if self.n_knots > 0:
            diff = z.unsqueeze(-1) - self.knots.unsqueeze(0)
            tp_d = (
                self.knot_power
                * F.relu(diff).pow(self.knot_power - 1)
                * inv_s.unsqueeze(-1)
            )
            if self.knot_power == 1:
                tp_d = tp_d * (diff > 0).to(diff.dtype)
            dbases.extend(tp_d.unbind(-1))

        # Fourier: d/dx sin(wz) = w cos(wz)/scale, d/dx cos(wz) = -w sin(wz)/scale
        if self.fourier_features > 0:
            wz = z.unsqueeze(-1) * self.freqs
            scale = self.freqs * inv_s.unsqueeze(-1)  # [N, F]
            dbases.extend((torch.cos(wz) * scale).unbind(-1))
            dbases.extend((-torch.sin(wz) * scale).unbind(-1))

        dphi = torch.stack(dbases, dim=-1)

        # chain rule through the clamp in `_z`: d clamp(z)/dx = 0 where the
        # input is saturated, so every basis derivative is zero there too.
        if self.activation_clip is not None:
            gate = (z_raw.abs() <= self.activation_clip).to(dphi.dtype)
            dphi = dphi * gate.unsqueeze(-1)

        return dphi


# ============================================================================ #
# Masked global linear model
# ============================================================================ #
class MaskedGlobalLinear(nn.Module):
    """Sparse linear (optionally basis-expanded) parent→child surrogate.

    Reconstructs each node from its direct parents over a directed graph.

    Mask convention: parents are rows, children are columns. For child ``j`` the
    prediction aggregates over parents ``i`` where ``mask[i, j] == 1``::

        pred[j] = bias[j] + sum_{i: mask[i,j]=1} <weight[i,j,:], phi(x_i)>

    where ``phi`` is the per-feature :class:`~causalig.models.surrogate.BasisExpansion` (or the identity,
    i.e. ``phi(x_i) = x_i``, when ``use_basis_expansion=False``). Only edges
    present in ``adjacency_matrix`` carry parameters, so the model is sparse in
    the number of edges ``E`` rather than dense in ``N^2``.

    Because the model is linear in ``phi``, :meth:`compute_input_jacobian`
    returns the exact input Jacobian in closed form (via
    :meth:`~causalig.models.surrogate.BasisExpansion.derivative`), which the causal-attribution methods use
    through ``compute_surrogate_gradients(..., has_closed_form_jacobian=True)``.

    Args:
        adjacency_matrix: `(N, N)` array or tensor; ``adjacency_matrix[i, j]``
            non-zero means `i` is a parent of `j`. Must contain at least one
            edge.
        predict_only_children: If ``True`` (default), parentless features are
            output as 0; if ``False`` they are copied from the input.
        use_basis_expansion: Expand each parent with :class:`~causalig.models.surrogate.BasisExpansion`
            (nonlinear per-edge functions). ``False`` gives a linear model.
        X_train: Optional training data `(n, N)`; its per-feature mean and
            standard deviation standardize the basis inputs. Recommended when
            `use_basis_expansion` is ``True``.
        n_knots: Truncated-power terms per feature (basis only).
        knot_power: Power of the truncated-power terms (basis only).
        polynomial_degree: Highest polynomial power (basis only).
        fourier_features: Number of sin/cos pairs (basis only).
        learnable_knots: Train the knot positions (basis only).
    """

    def __init__(
        self,
        adjacency_matrix: np.ndarray,
        predict_only_children: bool = True,
        use_basis_expansion: bool = True,
        X_train: torch.Tensor | None = None,  # for data-driven basis init (optional)
        n_knots=5,
        knot_power=2,
        polynomial_degree=3,
        fourier_features=0,
        learnable_knots=True,
    ) -> None:
        super().__init__()

        if (
            adjacency_matrix.ndim != 2
            or adjacency_matrix.shape[0] != adjacency_matrix.shape[1]
        ):
            raise ValueError("adjacency_matrix must be square [N, N].")

        self.n_features = N = adjacency_matrix.shape[0]
        self.predict_only_children = predict_only_children
        self.use_basis_expansion = use_basis_expansion
        self.n_knots = n_knots
        self.knot_power = knot_power
        self.polynomial_degree = polynomial_degree
        self.fourier_features = fourier_features
        self.learnable_knots = learnable_knots

        # mask & predictable children
        self.register_buffer(
            "mask",
            torch.as_tensor(adjacency_matrix, dtype=torch.float32),
        )
        self.register_buffer(
            "predictable_children",
            torch.where(self.mask.abs().sum(dim=0) > 0)[0],
        )
        if len(self.predictable_children) == 0:
            raise ValueError("No predictable children.")

        src, dst = self.mask.nonzero(as_tuple=True)  # [E], [E]
        src = src.long().contiguous()
        dst = dst.long().contiguous()
        E = len(src)
        self.register_buffer("edge_src", src)
        self.register_buffer("edge_dst", dst)

        # basis
        if use_basis_expansion:
            x_loc = None
            x_scale = None

            if X_train is not None:
                X_train = torch.as_tensor(X_train, dtype=torch.float32)
                x_loc = X_train.mean(dim=0)
                x_scale = X_train.std(dim=0).clamp_min(1e-4)

            # dyngen
            self.basis = BasisExpansion(
                n_features=N,
                n_knots=self.n_knots,
                knot_power=self.knot_power,
                polynomial_degree=self.polynomial_degree,
                fourier_features=self.fourier_features,
                learnable_knots=self.learnable_knots,
                x_loc=x_loc,
                x_scale=x_scale,
                activation_clip=8.0,
            )
            K = self.basis.n_basis
        else:
            self.basis = None
            K = 1
        self.n_basis = K

        # linear / basis weights & bias (mask applied at use time)
        self.weight = nn.Parameter(torch.empty(E, K))
        self.bias = nn.Parameter(torch.zeros(N))
        nn.init.xavier_uniform_(self.weight)

        logging.getLogger(__name__).debug(
            "MaskedGlobalLinear | N=%d | E=%d | K=%d", N, E, K
        )

    # ------------------------------------------------------------------ #
    # forward
    # ------------------------------------------------------------------ #
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Predict every child from its parents: `(B, N)` -> `(B, N)`."""
        if x.ndim != 2 or x.shape[1] != self.n_features:
            raise ValueError(f"x must be [B, {self.n_features}]; got {tuple(x.shape)}.")

        B, N = x.shape
        # basis
        phi = (
            self.basis(x) if self.use_basis_expansion else x.unsqueeze(-1)
        )  # [B, N, K]
        # gather only active edges
        phi_e = phi[:, self.edge_src]  # [B, E, K]
        # [B,E,K] * [E,K]
        contrib = (phi_e * self.weight.unsqueeze(0)).sum(-1)

        # scatter-add into children
        pred = torch.zeros(B, N, device=x.device, dtype=x.dtype)
        pred.index_add_(dim=1, index=self.edge_dst, source=contrib)

        # bias
        pred += self.bias

        recon = torch.zeros_like(x) if self.predict_only_children else x.clone()
        recon[:, self.predictable_children] = pred[:, self.predictable_children]

        return recon

    # ------------------------------------------------------------------ #
    # input Jacobian
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def compute_input_jacobian(
        self,
        x: torch.Tensor,
        dim_start: int,
        dim_end: int,
    ) -> torch.Tensor:
        """d pred[:, dim_start:dim_end] / d x   →   shape [B, chunk, N]."""
        device = x.device
        chunk = torch.arange(dim_start, dim_end, device=device)

        # 1) linear / basis: grads[b, j, n] = sum_kb phi'[b, n, kb] * weight[n, j, kb] * mask[n, j]
        if self.use_basis_expansion:
            phi_prime = self.basis.derivative(x)  # [B, N, K]
        else:
            phi_prime = torch.ones(
                x.shape[0], self.n_features, 1, device=device, dtype=x.dtype
            )
        grads = x.new_zeros(x.shape[0], len(chunk), self.n_features)  # [B, chunk, N]
        dst_mask = torch.isin(self.edge_dst, chunk)  # [1, chunk, E]
        edge_src = self.edge_src[dst_mask]  # [E_chunk]
        edge_dst = self.edge_dst[dst_mask] - dim_start  # [E_chunk]
        edge_w = self.weight[dst_mask]  # [E_chunk, K]

        contrib = (phi_prime[:, edge_src] * edge_w.unsqueeze(0)).sum(-1)  # [B, E_chunk]

        grads[:, edge_dst, edge_src] += contrib  # scatter-add into grads

        return grads


def train_surrogate_model(
    surrogate,
    X,
    epochs=100,
    batch_size=256,
    lr=1e-3,
    weight_decay=1e-5,
    train_frac=0.8,
    val_frac=0.1,
    seed=42,
    device="cuda",
    grad_clip_norm=1.0,
    num_workers=4,
    logger=None,
    save_dir=None,
):
    """Train a masked graph-structured surrogate (e.g. :class:`MaskedGlobalLinear`).

    Minimizes the MSE between ``surrogate(x)`` and ``x`` on the columns in
    ``surrogate.predictable_children`` (the children reconstructed from their
    parents), on unmasked inputs. This is only meaningful for surrogates whose
    output `j` reads nothing but the parents of `j` by construction.

    Uses AdamW with a per-step cosine schedule (down to ``lr / 100``),
    gradient clipping, and keeps the weights with the best validation loss.
    Batches with non-finite losses or gradients are skipped. `weight_decay`
    is not applied to parameters whose name contains ``bias`` or ``knots``.

    Args:
        surrogate: Model to train; must expose ``predictable_children``.
        X: Training data `(n_samples, d)`, a tensor or array (cast to
            ``float32``).
        epochs: Number of passes over the training split.
        batch_size: Mini-batch size.
        lr: Peak learning rate.
        weight_decay: AdamW weight decay.
        train_frac: Fraction of the rows of `X` used for training.
        val_frac: Fraction used for validation (model selection); the rest is
            the test split.
        seed: Seed for the random split.
        device: Training device. Defaults to ``"cuda"``.
        grad_clip_norm: Maximum gradient norm.
        num_workers: ``DataLoader`` worker processes.
        logger: Logger for per-epoch losses; defaults to this module's logger.
        save_dir: If given, the best ``surrogate.weight`` is saved there as
            ``best_masked_weight.npy`` (requires a ``weight`` attribute).

    Returns:
        The trained surrogate, on `device`, in eval mode.
    """
    device = torch.device(device)
    surrogate = surrogate.to(device)
    logger = logger or logging.getLogger(__name__)
    # ----- Splits -----
    X = torch.as_tensor(X, dtype=torch.float32)
    n = len(X)
    train_size = int(train_frac * n)
    val_size = int(val_frac * n)
    test_size = n - train_size - val_size
    train_set, val_set, test_set = random_split(
        X,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(seed),
    )

    dl_kwargs = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=num_workers > 0,
    )
    train_loader = DataLoader(train_set, shuffle=True, drop_last=True, **dl_kwargs)
    val_loader = DataLoader(val_set, shuffle=False, **dl_kwargs)
    test_loader = DataLoader(test_set, shuffle=False, **dl_kwargs)

    # ------------------------------------------------------------------ #
    # parameter groups
    # ------------------------------------------------------------------ #
    decay = []
    no_decay = []

    for name, p in surrogate.named_parameters():
        if not p.requires_grad:
            continue
        if "bias" in name or "knots" in name:
            no_decay.append(p)
        else:
            decay.append(p)
    optimizer = optim.AdamW(
        [
            {
                "params": decay,
                "weight_decay": weight_decay,
            },
            {
                "params": no_decay,
                "weight_decay": 0.0,
            },
        ],
        lr=lr,
        betas=(0.9, 0.99),
        eps=1e-8,
    )
    total_steps = epochs * len(train_loader)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=total_steps,
        eta_min=lr * 1e-2,
    )
    pc = surrogate.predictable_children

    # ------------------------------------------------------------------ #
    # eval
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def evaluate(loader):
        surrogate.eval()
        total = 0.0
        count = 0
        for x in loader:
            x = x.to(
                device,
                non_blocking=True,
            )
            out = surrogate(x)
            loss = F.mse_loss(
                out[:, pc],
                x[:, pc],
            )
            total += loss.item() * x.size(0)
            count += x.size(0)
        return total / count

    # ------------------------------------------------------------------ #
    # training
    # ------------------------------------------------------------------ #
    best_val = float("inf")
    best_state = None
    for epoch in range(epochs):
        surrogate.train()
        running = 0.0
        seen = 0
        for x in train_loader:
            x = x.to(
                device,
                non_blocking=True,
            )
            optimizer.zero_grad(set_to_none=True)
            out = surrogate(x)
            loss = F.mse_loss(
                out[:, pc],
                x[:, pc],
            )
            if not torch.isfinite(loss):
                logger.info("non-finite loss detected")
                continue
            loss.backward()
            # ---------------------------------------------------------- #
            # gradient sanity check
            # ---------------------------------------------------------- #
            bad_grad = False
            for p in surrogate.parameters():
                if p.grad is None:
                    continue
                if not torch.isfinite(p.grad).all():
                    bad_grad = True
                    break
            if bad_grad:
                logger.warning("non-finite gradients detected")
                optimizer.zero_grad(set_to_none=True)
                continue
            # ---------------------------------------------------------- #
            # clipping
            # ---------------------------------------------------------- #
            torch.nn.utils.clip_grad_norm_(
                surrogate.parameters(),
                max_norm=grad_clip_norm,
            )
            optimizer.step()
            scheduler.step()
            running += loss.item() * x.size(0)
            seen += x.size(0)

        train_loss = running / max(seen, 1)
        val_loss = evaluate(val_loader)
        improved = val_loss < best_val
        marker = " *" if improved else ""

        logger.info(
            f"Epoch {epoch + 1:03d} | "
            f"Train {train_loss:.6f} | "
            f"Val {val_loss:.6f} | "
            f"LR {optimizer.param_groups[0]['lr']:.2e}"
            f"{marker}"
        )
        if improved:
            best_val = val_loss
            best_state = {
                k: v.detach().cpu().clone() for k, v in surrogate.state_dict().items()
            }

    # ------------------------------------------------------------------ #
    # restore best
    # ------------------------------------------------------------------ #
    # ----- Restore best & persist -----
    if best_state is not None:
        surrogate.load_state_dict(best_state)
        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)
            np.save(
                os.path.join(save_dir, "best_masked_weight.npy"),
                surrogate.weight.detach().cpu().numpy(),
            )
    test_loss = evaluate(test_loader)
    logger.info(f"Best Val loss: {best_val:.6f}, Test Loss: {test_loss:.6f}")
    surrogate.eval()
    return surrogate


def create_autoencoder_from_scm(scm: Callable, structural: bool = False) -> Callable:
    """
    Create an autoencoder wrapper from a Structural Causal Model (SCM).

    Args:
        scm: SCM callable that takes input tensor and returns output tensor
        structural: Whether to use structural forward pass in SCM. Use
            ``True`` when the result is passed to CIG/CEG as a surrogate: each
            node's mechanism is then applied to the parent values it is given.
            With ``False``, non-root parents are re-simulated from their own
            (masked) ancestors, so indirect effects are lost on graphs deeper
            than one edge.

    Returns:
        Callable autoencoder function
    """

    def autoencoder(x: torch.Tensor) -> torch.Tensor:
        return scm(x, structural=structural)

    return autoencoder
