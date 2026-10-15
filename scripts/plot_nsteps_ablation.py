"""
Plot completeness delta vs. `n_steps`, for each num_features.

Reads the `_nsteps<N>`-suffixed result files produced by
`run_synthetic_nsteps_ablation.py`'s `run_nsteps_sweep` and plots relative
completeness delta (log-y) against the
number of integration steps $K$ (log-x), one line per num_features, with a
shaded std band. On a log-log plot, O(1/K) convergence already shows up as a
straight line of slope -1, so no separate reference line is drawn.

Pass `--learned-results-dir` to overlay a second surrogate variant's curve
(solid = the `--results-dir` variant, typically the oracle SCM surrogate;
dashed = `--learned-results-dir`, typically a learned `MaskedGlobalLinear`
surrogate) on the same axes, to compare their convergence directly.
"""

import argparse
import json
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments.visualization.plots import METHOD_COLORS, METHOD_DISPLAY_NAMES

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"

RESULT_NAME_PATTERN = re.compile(
    r"^synthetic_causal_metrics_multiseed_(?P<num_features>\d+)_noise.*"
    r"_nsteps(?P<n_steps>\d+)\.json$"
)


def _load_nsteps_summaries(results_dir: Path) -> list[dict]:
    """Load every `_nsteps<N>`-suffixed multiseed result file in a directory."""
    payloads = []
    for json_path in sorted(
        results_dir.glob("synthetic_causal_metrics_multiseed_*.json")
    ):
        match = RESULT_NAME_PATTERN.match(json_path.name)
        if match is None:
            continue

        with open(json_path) as f:
            payload = json.load(f)

        if payload.get("mode") != "causal_multi_seed":
            continue

        # Prefer explicit fields (added by run_nsteps_sweep); fall back to
        # the filename for result files that predate them.
        payload.setdefault("num_features", int(match.group("num_features")))
        payload.setdefault("attribution_n_steps", int(match.group("n_steps")))
        payloads.append(payload)

    return payloads


def _collect_delta_by_nsteps(
    payloads: list[dict],
    method: str,
) -> tuple[
    list[int],
    dict[int, dict[int, tuple[float, float] | None]],
    dict[int, tuple[float, float] | None],
]:
    """
    Collect `method`'s completeness delta keyed by [num_features][n_steps],
    plus an IG reference value keyed by num_features (taken from whichever
    n_steps entry has it first, since it shouldn't vary with n_steps).
    """
    features = set()
    delta_by_features: dict[int, dict[int, tuple[float, float] | None]] = {}
    ig_ref_by_features: dict[int, tuple[float, float] | None] = {}

    for payload in payloads:
        n_features = payload["num_features"]
        n_steps = payload["attribution_n_steps"]
        features.add(n_features)
        timings = payload.get("timings_seconds", {})

        entry = timings.get(f"completeness_delta.{method}")
        delta_by_features.setdefault(n_features, {})[n_steps] = (
            None
            if entry is None
            else (float(entry.get("mean", 0.0)), float(entry.get("std", 0.0)))
        )

        if n_features not in ig_ref_by_features:
            ig_entry = timings.get("completeness_delta.IG")
            if ig_entry is not None:
                ig_ref_by_features[n_features] = (
                    float(ig_entry.get("mean", 0.0)),
                    float(ig_entry.get("std", 0.0)),
                )

    return sorted(features), delta_by_features, ig_ref_by_features


def _plot_one_variant(
    ax: plt.Axes,
    n_steps_sorted: list[int],
    delta_by_features: dict[int, dict[int, tuple[float, float] | None]],
    colors: list,
    method_label: str,
    linestyle: str,
    variant_suffix: str | None,
    show_num_features: bool,
) -> None:
    for color, n_features in zip(colors, sorted(delta_by_features.keys())):
        row = delta_by_features[n_features]
        xs = np.array(
            [n for n in n_steps_sorted if row.get(n) is not None], dtype=float
        )
        if len(xs) == 0:
            continue
        means = np.array([row[int(n)][0] for n in xs])
        stds = np.array([row[int(n)][1] for n in xs])

        label_parts = [method_label]
        if show_num_features:
            label_parts.append(f"d={n_features}")
        if variant_suffix:
            label_parts.append(variant_suffix)
        label = " ".join(label_parts)

        ax.plot(xs, means, marker="o", linestyle=linestyle, label=label, color=color)
        lower = np.clip(means - stds, a_min=means.min() * 1e-3, a_max=None)
        ax.fill_between(xs, lower, means + stds, alpha=0.15, color=color)


def plot_completeness_delta_vs_nsteps(
    n_steps_sorted: list[int],
    delta_by_features: dict[int, dict[int, tuple[float, float] | None]],
    method: str = "CIG",
    save_path: str | None = None,
    figsize: tuple = (9, 5),
    learned_n_steps_sorted: list[int] | None = None,
    learned_delta_by_features: (
        dict[int, dict[int, tuple[float, float] | None]] | None
    ) = None,
) -> plt.Figure:
    """
    Plot `method`'s completeness delta vs. `n_steps`, one line per
    num_features. If `learned_delta_by_features` is given, overlay it as a
    second, dashed set of lines (typically a learned surrogate, vs. the
    primary solid lines' oracle surrogate) for direct comparison.
    """
    fig, ax = plt.subplots(figsize=figsize)

    features_sorted = sorted(delta_by_features.keys())
    base_color = METHOD_COLORS.get(method, "#8C3BFF")
    if len(features_sorted) > 1:
        cmap = plt.get_cmap("viridis")
        colors = [
            cmap(i / (len(features_sorted) - 1)) for i in range(len(features_sorted))
        ]
    else:
        colors = [base_color]

    method_label = METHOD_DISPLAY_NAMES.get(method, method)
    has_learned = learned_delta_by_features is not None
    show_num_features = len(features_sorted) > 1

    _plot_one_variant(
        ax,
        n_steps_sorted,
        delta_by_features,
        colors,
        method_label,
        linestyle="-",
        variant_suffix="(oracle surrogate)" if has_learned else None,
        show_num_features=show_num_features,
    )
    if has_learned:
        _plot_one_variant(
            ax,
            learned_n_steps_sorted or n_steps_sorted,
            learned_delta_by_features,
            colors,
            method_label,
            linestyle="--",
            variant_suffix="(learned surrogate)",
            show_num_features=show_num_features,
        )

    all_n_steps = sorted(set(n_steps_sorted) | set(learned_n_steps_sorted or []))
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks(all_n_steps)
    ax.set_xticklabels([str(int(n)) for n in all_n_steps])
    ax.xaxis.set_minor_formatter(mticker.NullFormatter())
    ax.set_xlabel("Number of integration steps ($K$)", fontsize=14)
    ax.set_ylabel("Relative completeness delta", fontsize=14)
    ax.tick_params(axis="both", labelsize=12)
    ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.6)
    if has_learned or show_num_features:
        ax.legend(fontsize=10)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot completeness delta vs n_steps, for each num_features."
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=RESULTS_DIR / "synthetic_nsteps_ablation",
        help="Directory containing the _nsteps<N>-suffixed multiseed result "
        "files produced by run_synthetic_nsteps_ablation.py (oracle surrogate "
        "variant by default).",
    )
    parser.add_argument(
        "--learned-results-dir",
        type=Path,
        default=None,
        help="If given, overlay this second directory's results (typically "
        "the learned-surrogate variant, e.g. "
        "results/synthetic_nsteps_ablation_learned_surrogate) as dashed "
        "lines on the same axes, to compare convergence across surrogate types.",
    )
    parser.add_argument(
        "--method", default="CIG", help="Method to plot (default: CIG)."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output image path. Default: <results-dir>/completeness_delta_vs_nsteps_<method>.png",
    )
    args = parser.parse_args()

    payloads = _load_nsteps_summaries(args.results_dir)
    if not payloads:
        raise RuntimeError(f"No _nsteps<N> result files found in {args.results_dir}")

    _, delta_by_features, _ = _collect_delta_by_nsteps(payloads, method=args.method)
    n_steps_sorted = sorted({n for row in delta_by_features.values() for n in row})

    learned_delta_by_features = None
    learned_n_steps_sorted = None
    suffix = ""
    if args.learned_results_dir:
        learned_payloads = _load_nsteps_summaries(args.learned_results_dir)
        if not learned_payloads:
            raise RuntimeError(
                f"No _nsteps<N> result files found in {args.learned_results_dir}"
            )
        _, learned_delta_by_features, _ = _collect_delta_by_nsteps(
            learned_payloads, method=args.method
        )
        learned_n_steps_sorted = sorted(
            {n for row in learned_delta_by_features.values() for n in row}
        )
        suffix = "_vs_learned"

    save_path = args.output or (
        args.results_dir / f"completeness_delta_vs_nsteps_{args.method}{suffix}.png"
    )
    plot_completeness_delta_vs_nsteps(
        n_steps_sorted,
        delta_by_features,
        method=args.method,
        save_path=str(save_path),
        learned_n_steps_sorted=learned_n_steps_sorted,
        learned_delta_by_features=learned_delta_by_features,
    )
    print(f"Saved plot to: {save_path}")
    plt.show()


if __name__ == "__main__":
    main()
