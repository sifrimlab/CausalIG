"""
Plot metrics vs number of features from `run_surrogate_comparison` results
(produced by scripts/run_synthetic_surrogate_comparison.py), i.e. results
comparing CIG with a surrogate trained on simulated data against CIG with
the true SCM as surrogate.
"""

import argparse
import json
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments.visualization.plots import (
    plot_metric_all_node_sets,
    plot_metric_vs_num_features,
    save_legend,
)

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"

RESULT_NAME_PATTERN = re.compile(
    r"^synthetic_surrogate_comparison_(?P<num_features>\d+)_noise.*\.json$"
)


def _load_surrogate_comparison_summaries(results_dir: Path) -> dict[int, dict]:
    summaries = {}
    for json_path in sorted(results_dir.glob("synthetic_surrogate_comparison_*.json")):
        match = RESULT_NAME_PATTERN.match(json_path.name)
        if match is None:
            continue

        with open(json_path) as f:
            payload = json.load(f)

        if payload.get("mode") != "causal_multi_seed":
            continue

        num_features = int(match.group("num_features"))
        summaries[num_features] = payload

    return summaries


def _collect_metric_curves(
    summaries: dict[int, dict],
    metric_name: str,
    node_set: str,
) -> tuple[list[int], dict[str, list[float]], dict[str, list[float]]]:
    features_sorted = sorted(summaries.keys())

    method_names = sorted(
        {
            method
            for payload in summaries.values()
            for method in payload.get("metrics_by_node_set", {})
            .get(node_set, {})
            .keys()
        }
    )

    means = {method: [] for method in method_names}
    stds = {method: [] for method in method_names}

    for n_features in features_sorted:
        metrics = summaries[n_features].get("metrics_by_node_set", {}).get(node_set, {})
        for method in method_names:
            entry = metrics.get(method, {}).get(metric_name)
            if entry is None:
                means[method].append(float("nan"))
                stds[method].append(float("nan"))
            else:
                means[method].append(float(entry.get("mean", float("nan"))))
                stds[method].append(float(entry.get("std", float("nan"))))

    return features_sorted, means, stds


def _collect_metric_curves_all_node_sets(
    summaries: dict[int, dict],
    metric: str,
) -> tuple[
    list[int], dict[str, dict[str, list[float]]], dict[str, dict[str, list[float]]]
]:
    node_sets = ["direct_nodes", "eldest_ancestors", "all_nodes"]

    features = None
    means_by_node_set = {}
    stds_by_node_set = {}

    for node_set in node_sets:
        features, metric_means, metric_stds = _collect_metric_curves(
            summaries, metric_name=metric, node_set=node_set
        )
        means_by_node_set[node_set] = metric_means
        stds_by_node_set[node_set] = metric_stds

    return features, means_by_node_set, stds_by_node_set


def _filter_methods(
    means_by_method: dict[str, list[float]],
    stds_by_method: dict[str, list[float]] | None,
    methods: list[str] | None,
) -> tuple[dict[str, list[float]], dict[str, list[float]] | None]:
    """Restrict method dicts to `methods`, preserving that list's order. No-op if None."""
    if methods is None:
        return means_by_method, stds_by_method

    filtered_means = {m: means_by_method[m] for m in methods if m in means_by_method}
    filtered_stds = (
        {m: stds_by_method[m] for m in methods if m in stds_by_method}
        if stds_by_method is not None
        else None
    )
    return filtered_means, filtered_stds


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot surrogate-comparison metrics vs feature size."
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=RESULTS_DIR / "synthetic_surrogate_comparison",
        help="Directory containing synthetic_surrogate_comparison_*.json files.",
    )
    parser.add_argument(
        "--metric",
        choices=["mse", "pearson", "spearman"],
        default="spearman",
        help="Which causal metric to plot vs number of features.",
    )
    parser.add_argument(
        "--node-set",
        choices=["direct_nodes", "eldest_ancestors", "all_nodes", "all"],
        default="all",
        help="Which node set to plot. 'all' creates subplots for all node sets.",
    )
    parser.add_argument(
        "--output-prefix",
        type=str,
        default="surrogate_comparison",
        help="Prefix for output figure names.",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=None,
        metavar="METHOD",
        help="Restrict plotted methods to these (e.g. --methods CIG_oracle "
        "CIG_learned). Default: plot every method present in the result files.",
    )
    args = parser.parse_args()

    summaries = _load_surrogate_comparison_summaries(args.results_dir)
    if len(summaries) < 2:
        raise RuntimeError(
            f"Need at least 2 multiseed result files in {args.results_dir}, "
            f"found {len(summaries)}"
        )

    if args.node_set == "all":
        features, means_by_node_set, stds_by_node_set = (
            _collect_metric_curves_all_node_sets(summaries, metric=args.metric)
        )
        for node_set in means_by_node_set:
            means_by_node_set[node_set], stds_by_node_set[node_set] = _filter_methods(
                means_by_node_set[node_set], stds_by_node_set[node_set], args.methods
            )
        metric_out = (
            args.results_dir / f"{args.output_prefix}_{args.metric}_all_node_sets.png"
        )
        plot_metric_all_node_sets(
            features=features,
            metric_means_by_node_set=means_by_node_set,
            metric_stds_by_node_set=stds_by_node_set,
            metric_name=args.metric,
            save_path=str(metric_out),
        )
        all_methods = sorted({m for d in means_by_node_set.values() for m in d})
        legend_out = args.results_dir / f"{args.output_prefix}_legend.png"
        save_legend(method_names=all_methods, save_path=str(legend_out))

        print(f"Saved all node sets plot to: {metric_out}")
        print(f"Saved legend to: {legend_out}")
        plt.show()
    else:
        features, metric_means, metric_stds = _collect_metric_curves(
            summaries, metric_name=args.metric, node_set=args.node_set
        )
        metric_means, metric_stds = _filter_methods(
            metric_means, metric_stds, args.methods
        )

        methods_suffix = f"_{'-'.join(args.methods)}" if args.methods else ""
        metric_out = (
            args.results_dir
            / f"{args.output_prefix}_{args.metric}_{args.node_set}_vs_features"
            f"{methods_suffix}.png"
        )
        plot_metric_vs_num_features(
            features=features,
            metric_means_by_method=metric_means,
            metric_stds_by_method=metric_stds,
            metric_name=args.metric,
            save_path=str(metric_out),
        )

        legend_out = (
            args.results_dir / f"{args.output_prefix}_legend{methods_suffix}.png"
        )
        save_legend(method_names=list(metric_means.keys()), save_path=str(legend_out))

        print(f"Saved metric plot to: {metric_out}")
        print(f"Saved legend to: {legend_out}")
        plt.show()


if __name__ == "__main__":
    main()
