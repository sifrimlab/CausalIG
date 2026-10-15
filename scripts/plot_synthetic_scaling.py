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
    plot_runtime_vs_num_features,
    save_legend,
)

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"

RESULT_NAME_PATTERN = re.compile(
    r"^synthetic_causal_metrics_multiseed_(?P<num_features>\d+)_noise.*\.json$"
)

PERTURBATION_RESULT_PATTERN = re.compile(
    r"^synthetic_causal_metrics_multiseed_(?P<num_features>\d+)_noise.*"
    r"_(?:edgedrop|adddag|reverse|addcyclic)(?P<edge_drop_frac>[0-9.]+)\.json$"
)


def _load_multiseed_summaries(results_dir: Path) -> dict[int, dict]:
    summaries = {}
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

        # Graph-perturbation robustness variants (drop/add/reverse) share the
        # base filename prefix but carry a nonzero edge_drop_frac; exclude
        # them from the main curves.
        if payload.get("edge_drop_frac", 0.0):
            continue

        num_features = int(match.group("num_features"))
        summaries[num_features] = payload

    return summaries


def _load_cig_edge_drop_overlays(
    results_dir: Path,
    features: list[int],
    node_set: str,
    metric_name: str,
    perturbation_method: str = "remove",
) -> dict[float, tuple[list[float], list[float]]]:
    """
    Load CIG-only metric curves for each fraction of `perturbation_method`
    found in results_dir.

    Produced by `run_synthetic_scaling_edge_robustness.py`. Returns a mapping
    fraction -> (means, stds), aligned to `features` (NaN where a given
    num_features has no matching result file).
    """
    raw: dict[float, dict[int, tuple[float, float]]] = {}

    for json_path in sorted(
        results_dir.glob("synthetic_causal_metrics_multiseed_*.json")
    ):
        match = PERTURBATION_RESULT_PATTERN.match(json_path.name)
        if match is None:
            continue

        with open(json_path) as f:
            payload = json.load(f)

        if payload.get("mode") != "causal_multi_seed":
            continue
        if payload.get("graph_perturbation_method", "remove") != perturbation_method:
            continue

        num_features = int(match.group("num_features"))
        edge_drop_frac = float(
            payload.get("edge_drop_frac", match.group("edge_drop_frac"))
        )

        entry = (
            payload.get("metrics_by_node_set", {})
            .get(node_set, {})
            .get("CIG", {})
            .get(metric_name)
        )
        if entry is None:
            continue

        raw.setdefault(edge_drop_frac, {})[num_features] = (
            float(entry.get("mean", float("nan"))),
            float(entry.get("std", 0.0)),
        )

    overlays = {}
    for edge_drop_frac, means_by_features in raw.items():
        means = [
            means_by_features.get(nf, (float("nan"), float("nan")))[0]
            for nf in features
        ]
        stds = [
            means_by_features.get(nf, (float("nan"), float("nan")))[1]
            for nf in features
        ]
        overlays[edge_drop_frac] = (means, stds)

    return overlays


def _collect_metric_curves(
    summaries: dict[int, dict],
    metric_name: str,
    node_set: str,
) -> (list[int], dict[str, list[float]], dict[str, list[float]]):
    """
    Collect metric curves from multi-seed summaries.

    Args:
        summaries: Dictionary of summaries keyed by num_features
        metric_name: Name of metric to extract (mse, pearson, spearman)
        node_set: Which node set to use (direct_nodes, eldest_ancestors, all_nodes)

    Returns:
        Tuple of (features_sorted, means_dict, stds_dict)
    """
    features_sorted = sorted(summaries.keys())

    method_names = set()
    for payload in summaries.values():
        method_names.update(payload["metrics_by_node_set"].get(node_set, {}).keys())

    method_names = sorted(method_names)

    means = {method: [] for method in method_names}
    stds = {method: [] for method in method_names}

    for n_features in features_sorted:
        payload = summaries[n_features]

        metrics = payload["metrics_by_node_set"].get(node_set, {})

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
    """Collect metric curves for all three node sets, keyed by node-set name."""
    node_sets = ["direct_nodes", "eldest_ancestors", "all_nodes"]

    features = None
    means_by_node_set = {}
    stds_by_node_set = {}

    for node_set in node_sets:
        features, metric_means, metric_stds = _collect_metric_curves(
            summaries,
            metric_name=metric,
            node_set=node_set,
        )
        means_by_node_set[node_set] = metric_means
        stds_by_node_set[node_set] = metric_stds

    return features, means_by_node_set, stds_by_node_set


def _collect_runtime_curves(
    summaries: dict[int, dict],
) -> (list[int], dict[str, list[float]], dict[str, list[float]]):
    features_sorted = sorted(summaries.keys())

    method_names = sorted(
        {
            timing_key.split(".", 1)[1]
            for payload in summaries.values()
            for timing_key in payload.get("timings_seconds", {}).keys()
            if timing_key.startswith("attributions_seconds.")
            and "prepare_autoencoder" not in timing_key
            and "prepare_surrogate" not in timing_key
        }
    )

    means = {method: [] for method in method_names}
    stds = {method: [] for method in method_names}

    for n_features in features_sorted:
        payload = summaries[n_features]
        timings = payload.get("timings_seconds", {})
        for method in method_names:
            timing_entry = timings.get(f"attributions_seconds.{method}")
            if timing_entry is None:
                means[method].append(float("nan"))
                stds[method].append(float("nan"))
            else:
                means[method].append(float(timing_entry.get("mean", float("nan"))))
                stds[method].append(float(timing_entry.get("std", float("nan"))))

    return features_sorted, means, stds


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
        description="Plot causal multi-seed metrics and runtimes vs feature size."
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=RESULTS_DIR / "synthetic_general_sim_autoencoder",
        help="Directory containing synthetic_causal_metrics_multiseed_*.json files.",
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
        default="eldest_ancestors",
        help="Which node set to plot. 'all' creates subplots for all node sets.",
    )
    parser.add_argument(
        "--output-prefix",
        type=str,
        default="causal_scaling",
        help="Prefix for output figure names.",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=None,
        metavar="METHOD",
        help="Restrict plotted methods to these (e.g. --methods CIG IG). "
        "Default: plot every method present in the result files.",
    )
    parser.add_argument(
        "--edge-drop-overlay",
        action="store_true",
        help="Overlay CIG curves at different perturbation fractions (from "
        "run_synthetic_scaling_edge_robustness.py) on top of the main figure. "
        "Off by default so this script still reproduces the plain scaling "
        "figure even when --results-dir also contains perturbation result files.",
    )
    parser.add_argument(
        "--overlay-method",
        choices=["remove", "add_dag", "reverse"],
        default="remove",
        help="Which graph_perturbation_method's CIG curves to overlay "
        "(only used with --edge-drop-overlay). Default 'remove' matches the "
        "original drop-only ablation.",
    )
    parser.add_argument(
        "--overlay-only",
        action="store_true",
        help="Draw only the --overlay-method fraction curves (25/50/75%%) — "
        "no base CIG/IG/CEG scaling lines, no runtime plot, no legend. "
        "Implies --edge-drop-overlay. Use for a clean per-method figure "
        "(e.g. one plot each for remove/add_dag/reverse).",
    )
    args = parser.parse_args()
    if args.overlay_only:
        args.edge_drop_overlay = True

    summaries = _load_multiseed_summaries(args.results_dir)
    summaries = {k: v for k, v in summaries.items() if k >= 100}
    if len(summaries) < 2:
        raise RuntimeError(
            f"Need at least 2 multiseed result files in {args.results_dir}, found {len(summaries)}"
        )

    if args.node_set == "all":
        # Plot all node sets in a 1x3 grid of subplots
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
        print(f"Saved all node sets plot to: {metric_out}")
        plt.show()
    else:
        # Plot single node set
        features, metric_means, metric_stds = _collect_metric_curves(
            summaries,
            metric_name=args.metric,
            node_set=args.node_set,
        )
        _, runtime_means, runtime_stds = _collect_runtime_curves(summaries)

        if args.overlay_only:
            metric_means, metric_stds = {}, {}
        else:
            metric_means, metric_stds = _filter_methods(
                metric_means, metric_stds, args.methods
            )
            runtime_means, runtime_stds = _filter_methods(
                runtime_means, runtime_stds, args.methods
            )

        cig_edge_drop_overlays = (
            _load_cig_edge_drop_overlays(
                args.results_dir,
                features,
                node_set=args.node_set,
                metric_name=args.metric,
                perturbation_method=args.overlay_method,
            )
            if args.edge_drop_overlay
            else {}
        )
        overlay_suffix = (
            f"_{args.overlay_method}_overlay" if cig_edge_drop_overlays else ""
        )
        methods_suffix = (
            "_overlay_only"
            if args.overlay_only
            else (f"_{'-'.join(args.methods)}" if args.methods else "")
        )

        metric_out = (
            args.results_dir
            / f"{args.output_prefix}_{args.metric}_{args.node_set}_vs_features"
            f"{methods_suffix}{overlay_suffix}.png"
        )

        plot_metric_vs_num_features(
            features=features,
            metric_means_by_method=metric_means,
            metric_stds_by_method=metric_stds,
            metric_name=args.metric,
            save_path=str(metric_out),
            cig_edge_drop_overlays=cig_edge_drop_overlays or None,
            cig_overlay_method=args.overlay_method,
        )
        print(f"Saved metric plot to: {metric_out}")

        if not args.overlay_only:
            runtime_out = (
                args.results_dir
                / f"{args.output_prefix}_runtime_vs_features{methods_suffix}.png"
            )
            plot_runtime_vs_num_features(
                features=features,
                runtime_means_by_method=runtime_means,
                runtime_stds_by_method=runtime_stds,
                save_path=str(runtime_out),
            )
            print(f"Saved runtime plot to: {runtime_out}")

            legend_out = (
                args.results_dir / f"{args.output_prefix}_legend{methods_suffix}.png"
            )
            save_legend(
                method_names=list(metric_means.keys()),
                save_path=str(legend_out),
            )
            print(f"Saved legend to: {legend_out}")

        plt.show()


if __name__ == "__main__":
    main()
