"""Visualization utilities for attribution results."""

import matplotlib.pyplot as plt
import numpy as np
import torch

METHOD_DISPLAY_NAMES = {
    "CIG": "Causal Integrated Gradients",
    "CEG": "Causal Expected Gradients",
    "IG": "Integrated Gradients",
    "CondSHAP": "Conditional SHAP",
    "CausalSHAP": "Causal Shapley",
    # run_surrogate_comparison variants (surrogate-label-suffixed method names)
    "CIG_oracle": "CIG (oracle SCM surrogate)",
    "CIG_learned": "CIG (learned surrogate, GAM)",
    "CIG_learned_linear": "CIG (learned surrogate, linear)",
    "IG_oracle": "Integrated Gradients",
}

METHOD_COLORS = {
    "IG": "#8C3BFF",  # purple
    "CIG": "#FFA9FF",  # light pink
    "CEG": "#00B159",  # green
    "CondSHAP": "#FF8C00",  # amber orange — far from purple, pink, green
    "CausalSHAP": "#0077BB",  # medium blue — far from all others
    # run_surrogate_comparison variants: same pink/purple families as CIG/IG,
    # with saturation increasing as the surrogate gets less flexible (oracle
    # SCM -> learned GAM -> learned linear).
    "CIG_oracle": "#FFA9FF",
    "CIG_learned": "#C71585",
    "CIG_learned_linear": "#800040",
    "IG_oracle": "#8C3BFF",
}

NODE_SET_COLORS = {
    "direct_nodes": "green",
    "all_ancestors": "blue",
    "eldest_ancestors": "purple",
}

# Vertical offset (in the imshow sample axis) at which each node set's markers
# are drawn below the x-axis, so overlapping node sets remain distinguishable.
NODE_SET_MARKER_Y_OFFSETS = {
    "direct_nodes": -1,
    "all_ancestors": -3,
    "eldest_ancestors": -5,
}


def plot_cate_comparison(
    attr: torch.Tensor,
    cate: torch.Tensor,
    method_name: str,
    node_sets: dict[str, list[int]],
    save_path: str | None = None,
    figsize: tuple = (10, 4),
) -> plt.Figure:
    """
    Side-by-side heatmap of a method's attributions vs. ground-truth CATE,
    with feature-set membership marked below the x-axis of each panel.

    Args:
        attr: Attribution tensor, shape (samples, features)
        cate: Ground-truth CATE tensor, shape (samples, features)
        method_name: Attribution method key (e.g. "CIG"), used for the title
        node_sets: Mapping from node-set name (e.g. "direct_nodes") to the
            list of feature indices in that set
        save_path: Path to save the figure (optional)
        figsize: Figure size tuple (default: (10, 4))

    Returns:
        matplotlib Figure object
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    display_name = METHOD_DISPLAY_NAMES.get(method_name, method_name)

    im1 = axes[0].imshow(attr.cpu().numpy(), cmap="coolwarm", aspect="auto")
    axes[0].set_title(f"{display_name} Attributions")
    axes[0].set_xlabel("Features")
    axes[0].set_ylabel("Samples")

    axes[1].imshow(cate.cpu().numpy(), cmap="coolwarm", aspect="auto")
    axes[1].set_title("CATE")
    axes[1].set_xlabel("Features")

    for node_set_name, indices in node_sets.items():
        color = NODE_SET_COLORS.get(node_set_name, "black")
        y = NODE_SET_MARKER_Y_OFFSETS.get(node_set_name, -1)
        for idx in indices:
            axes[0].plot(idx, y, "o", color=color, markersize=2)
            axes[1].plot(idx, y, "o", color=color, markersize=2)

    fig.colorbar(im1, ax=axes, shrink=0.8)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig


def plot_metric_vs_num_features(
    features: list[int],
    metric_means_by_method: dict[str, list[float]],
    metric_stds_by_method: dict[str, list[float]] | None = None,
    metric_name: str = "mse",
    save_path: str | None = None,
    figsize: tuple = (9, 5),
    cig_edge_drop_overlays: (
        dict[float, tuple[list[float], list[float] | None]] | None
    ) = None,
    cig_overlay_method: str = "remove",
) -> plt.Figure:
    """Plot a causal metric (MSE/Pearson/Spearman) vs number of features.

    Args:
        cig_edge_drop_overlays: Optional robustness overlay for CIG only,
            mapping ``fraction -> (means, stds)`` (same length/order as
            `features`; `stds` may be ``None``). Drawn as faded, increasingly
            dashed lines in CIG's color, to show how CIG degrades under graph
            misspecification, on the same axes used for the main scaling
            comparison.
        cig_overlay_method: Which `experiments.utils.graph_perturbation.
            PerturbationMethod` value `cig_edge_drop_overlays` was generated
            with — only used to word the legend label appropriately.
    """
    fig, ax = plt.subplots(figsize=figsize)
    x = np.asarray(features, dtype=float)

    for method, means in metric_means_by_method.items():
        color = METHOD_COLORS.get(method)
        y = np.asarray(means, dtype=float)
        ax.plot(
            x,
            y,
            marker="o",
            linestyle="-",
            label=METHOD_DISPLAY_NAMES.get(method, method),
            color=color,
        )

        if metric_stds_by_method is not None and method in metric_stds_by_method:
            std = np.asarray(metric_stds_by_method[method], dtype=float)
            lower = y - std
            upper = y + std
            ax.fill_between(x, lower, upper, alpha=0.2, color=color)

    if cig_edge_drop_overlays:
        overlay_phrase = {
            "remove": "edges hidden",
            "add_dag": "spurious edges added",
            "reverse": "edges reversed",
        }.get(cig_overlay_method, f"{cig_overlay_method} perturbation")

        cig_color = METHOD_COLORS.get("CIG")
        overlay_linestyles = ["--", ":", "-."]
        for i, edge_drop_frac in enumerate(sorted(cig_edge_drop_overlays)):
            means, stds = cig_edge_drop_overlays[edge_drop_frac]
            y = np.asarray(means, dtype=float)
            ax.plot(
                x,
                y,
                marker="o",
                markersize=4,
                linestyle=overlay_linestyles[i % len(overlay_linestyles)],
                alpha=0.9,
                color=cig_color,
                label=f"CIG ({edge_drop_frac:.0%} {overlay_phrase})",
            )
            if stds is not None:
                std = np.asarray(stds, dtype=float)
                ax.fill_between(x, y - std, y + std, alpha=0.18, color=cig_color)

    ax.set_xscale("log")
    if metric_name.lower() == "mse":
        ax.set_yscale("log")
    elif metric_name.lower() in ("spearman", "pearson"):
        ax.set_ylim(0, 1.05)

    label_map = {
        "spearman": "Spearman Correlation",
        "pearson": "Pearson Correlation",
        "mse": "MSE",
    }
    ylabel = label_map.get(metric_name.lower(), metric_name.upper())
    ax.set_xlabel("Number of Features", fontsize=14)
    ax.set_ylabel(ylabel, fontsize=14)
    ax.tick_params(axis="both", labelsize=12)
    ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.6)
    ax.legend(fontsize=12)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig


def plot_metric_all_node_sets(
    features: list[int],
    metric_means_by_node_set: dict[str, dict[str, list[float]]],
    metric_stds_by_node_set: dict[str, dict[str, list[float]]] | None = None,
    metric_name: str = "mse",
    save_path: str | None = None,
    figsize: tuple = (15, 4),
) -> plt.Figure:
    """
    Plot a causal metric vs number of features, with one subplot per node set.

    Args:
        features: Sorted list of feature counts (x-axis)
        metric_means_by_node_set: Mapping from node-set name to a mapping of
            method name -> list of metric means (one per entry in `features`)
        metric_stds_by_node_set: Same shape as `metric_means_by_node_set`,
            with standard deviations instead of means (optional)
        metric_name: Which metric is being plotted (mse, pearson, spearman)
        save_path: Path to save the figure (optional)
        figsize: Figure size tuple (default: (15, 4))

    Returns:
        matplotlib Figure object
    """
    node_set_names = list(metric_means_by_node_set.keys())
    fig, axes = plt.subplots(1, len(node_set_names), figsize=figsize)
    if len(node_set_names) == 1:
        axes = [axes]
    fig.suptitle(f"{metric_name.upper()} vs Number of Features (Different Node Sets)")

    for ax, node_set_name in zip(axes, node_set_names):
        means_by_method = metric_means_by_node_set[node_set_name]
        stds_by_method = (
            metric_stds_by_node_set.get(node_set_name, {})
            if metric_stds_by_node_set is not None
            else {}
        )

        for method, means in means_by_method.items():
            color = METHOD_COLORS.get(method)
            ax.errorbar(
                features,
                means,
                yerr=stds_by_method.get(method),
                marker="o",
                linestyle="-",
                label=METHOD_DISPLAY_NAMES.get(method, method),
                color=color,
                capsize=5,
                capthick=2,
            )

        ax.set_xlabel("Number of Features")
        ax.set_ylabel(metric_name.upper())
        ax.set_title(node_set_name.replace("_", " ").title())
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig


def plot_runtime_vs_num_features(
    features: list[int],
    runtime_means_by_method: dict[str, list[float]],
    runtime_stds_by_method: dict[str, list[float]] | None = None,
    save_path: str | None = None,
    figsize: tuple = (9, 5),
) -> plt.Figure:
    """Plot runtime (seconds) vs number of features for attribution methods."""
    fig, ax = plt.subplots(figsize=figsize)
    x = np.asarray(features, dtype=float)

    methods = list(runtime_means_by_method.keys())
    n = len(methods)
    # Spread methods symmetrically in log-y space so overlapping lines separate
    log_offsets = np.linspace(-0.04, 0.04, n) if n > 1 else [0.0]

    for offset, method in zip(log_offsets, methods):
        means = runtime_means_by_method[method]
        y = np.asarray(means, dtype=float) * (10**offset)
        ax.plot(
            x,
            y,
            marker="o",
            linestyle="-",
            label=METHOD_DISPLAY_NAMES.get(method, method),
            color=METHOD_COLORS.get(method),
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Number of Features", fontsize=14)
    ax.set_ylabel("Runtime (s)", fontsize=14)
    ax.tick_params(axis="both", labelsize=12)
    ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.6)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig


def save_legend(
    method_names: list[str],
    save_path: str,
    figsize: tuple = (4, 2.2),
) -> plt.Figure:
    """Save a standalone legend figure with fully spelled-out method names."""
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_visible(False)

    handles = []
    labels = []
    for method in method_names:
        (handle,) = ax.plot(
            [], [], marker="o", linestyle="-", color=METHOD_COLORS.get(method)
        )
        handles.append(handle)
        labels.append(METHOD_DISPLAY_NAMES.get(method, method))

    fig.legend(handles, labels, loc="center", fontsize=12, frameon=True)
    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig
