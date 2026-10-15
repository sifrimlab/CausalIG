"""
Plot the per-sample distribution of completeness-axiom delta.

Complements print_synthetic_completeness_table.py / plot_nsteps_ablation.py,
which report a single batch-aggregated WAPE scalar per (config, seed). This
reads the raw per-sample `|delta_i|` values (added by
`compute_attributions`'s `completeness_delta_samples`, via
`_per_sample_completeness_delta` in experiments/synthetic.py) from one
multiseed result file's `per_seed` list, pools them across all seeds, and
plots their distribution — answering "is completeness roughly uniform
across explained samples, or concentrated on a few hard ones?" rather than
"how big is the aggregate error."

Deliberately plots raw |delta_i|, not a per-sample ratio to |expected_diff_i|
— a per-sample ratio reintroduces exactly the near-zero-denominator
instability WAPE was built to avoid.

Pass `--compare-method` (e.g. `--method CIG --compare-method IG`) to overlay
a second method's distribution on the same axes — the standard way to check
whether a method's per-sample spread is unusual, by comparing it against a
trusted baseline (e.g. IG) computed on the very same samples.

Summary statistics are reported as median/IQR/min-max rather than mean±std:
these distributions are right-skewed (mean > median), so a small number of
seeds gives a much more stable read on the median and quartiles than on the
mean and standard deviation.
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments.visualization.plots import METHOD_COLORS, METHOD_DISPLAY_NAMES

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def _load_pooled_abs_delta(result_path: Path, method: str) -> np.ndarray:
    """Pool |delta_i| across every seed's batch in one multiseed result file."""
    with open(result_path) as f:
        payload = json.load(f)

    values = []
    for seed_result in payload.get("per_seed", []):
        samples = seed_result.get("completeness_delta_samples", {}).get(method)
        if samples:
            values.extend(samples["abs_delta"])

    if not values:
        raise RuntimeError(
            f"No completeness_delta_samples for method={method!r} in {result_path} "
            "— rerun the experiment that produced it with the updated compute_attributions."
        )
    return np.asarray(values, dtype=float)


def _summary_row(method: str, abs_delta: np.ndarray) -> dict:
    p25, p75 = np.percentile(abs_delta, [25, 75])
    return {
        "method": method,
        "n": len(abs_delta),
        "median": np.median(abs_delta),
        "p25": p25,
        "p75": p75,
        "iqr": p75 - p25,
        "min": abs_delta.min(),
        "max": abs_delta.max(),
        "max_over_min": abs_delta.max() / abs_delta.min(),
    }


def print_summary_table(rows: list[dict]) -> None:
    print("| Method | n | median | p25 | p75 | IQR | min-max | max/min |")
    print("|---|---|---|---|---|---|---|---|")
    for r in rows:
        print(
            f"| {r['method']} | {r['n']} | {r['median']:.2e} | {r['p25']:.2e} | "
            f"{r['p75']:.2e} | {r['iqr']:.2e} | {r['min']:.2e} - {r['max']:.2e} | "
            f"{r['max_over_min']:.2e}x |"
        )


def plot_per_sample_histogram(
    abs_delta_by_method: dict[str, np.ndarray],
    save_path: str | None = None,
    figsize: tuple = (10, 5.5),
) -> plt.Figure:
    """One method: single-color histogram with median/mean lines. Two methods
    (`--compare-method`): overlaid histograms with a max/min ratio in the
    legend, for comparing proportional spread across methods."""
    fig, ax = plt.subplots(figsize=figsize)
    multi = len(abs_delta_by_method) > 1

    for method, abs_delta in abs_delta_by_method.items():
        color = METHOD_COLORS.get(method, "#8C3BFF")
        positive = abs_delta[abs_delta > 0]
        log_bins = np.logspace(np.log10(positive.min()), np.log10(positive.max()), 30)
        method_label = METHOD_DISPLAY_NAMES.get(method, method)

        if multi:
            label = f"{method_label} (max/min={abs_delta.max() / abs_delta.min():.1e}x)"
            ax.hist(
                positive,
                bins=log_bins,
                color=color,
                alpha=0.65,
                edgecolor="white",
                label=label,
            )
        else:
            ax.hist(positive, bins=log_bins, color=color, alpha=0.75, edgecolor="white")
            median = np.median(abs_delta)
            mean = abs_delta.mean()
            ax.axvline(
                median,
                color="black",
                linestyle="--",
                linewidth=1.2,
                label=f"median={median:.2e}",
            )
            ax.axvline(
                mean,
                color="black",
                linestyle=":",
                linewidth=1.2,
                label=f"mean={mean:.2e}",
            )

    ax.set_xscale("log")
    xlabel = (
        "Per-sample absolute convergence delta"
        if multi
        else (
            f"{METHOD_DISPLAY_NAMES.get(next(iter(abs_delta_by_method)), next(iter(abs_delta_by_method)))} "
            "per-sample absolute convergence delta"
        )
    )
    ax.set_xlabel(xlabel, fontsize=14)
    ax.set_ylabel("Count", fontsize=14)
    ax.tick_params(axis="both", labelsize=12)
    ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.4)
    ax.legend(fontsize=11)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot the per-sample |delta_i| distribution from one "
        "multiseed completeness-delta result file."
    )
    parser.add_argument(
        "result_file",
        type=Path,
        nargs="?",
        default=None,
        help="Path to a synthetic_causal_metrics_multiseed_*.json file "
        "(from run_synthetic_completeness_comparison.py, "
        "run_synthetic_nsteps_ablation.py, or similar). If omitted, the "
        "--num-features file in --results-dir is used.",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=RESULTS_DIR / "synthetic_completeness_comparison",
        help="Directory searched when no result_file is given.",
    )
    parser.add_argument(
        "--num-features",
        type=int,
        default=774,
        help="Feature count to plot when no result_file is given (default: 774).",
    )
    parser.add_argument(
        "--method", default="CIG", help="Method to plot (default: CIG)."
    )
    parser.add_argument(
        "--compare-method",
        default=None,
        help="If given, overlay this second method's distribution (e.g. "
        "--method CIG --compare-method IG) to compare proportional spread.",
    )
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    if args.result_file is None:
        matches = sorted(
            args.results_dir.glob(
                f"synthetic_causal_metrics_multiseed_{args.num_features}_noise*.json"
            )
        )
        if len(matches) != 1:
            raise SystemExit(
                f"Expected one result file for {args.num_features} features in "
                f"{args.results_dir}, found {len(matches)}; pass result_file explicitly."
            )
        args.result_file = matches[0]

    methods = [args.method] + ([args.compare_method] if args.compare_method else [])
    abs_delta_by_method = {
        m: _load_pooled_abs_delta(args.result_file, m) for m in methods
    }

    rows = [_summary_row(m, d) for m, d in abs_delta_by_method.items()]
    print_summary_table(rows)

    suffix = "_".join(methods)
    save_path = args.output or args.result_file.with_name(
        f"{args.result_file.stem}_per_sample_{suffix}.png"
    )
    plot_per_sample_histogram(abs_delta_by_method, save_path=str(save_path))
    print(f"\nSaved plot to: {save_path}")
    plt.show()


if __name__ == "__main__":
    main()
