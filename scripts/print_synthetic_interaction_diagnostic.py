"""
Print pooled per-type summary statistics from the additive-surrogate
interaction indirect-effect diagnostic.

Reads `multiseed_results.json` (produced by
`run_synthetic_interaction_indirect_effect_multiseed.py`), pools the
per-node correlation and relative-error statistics across all seeds by node
type ("interaction" vs. "linear" control), and prints a summary table.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments.diagnostics.stats_utils import summarize_correlations

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def _load_per_node_results(results_dir: Path) -> list[dict]:
    path = results_dir / "multiseed_results.json"
    if not path.exists():
        raise RuntimeError(
            f"No results file found at {path} -- run "
            "run_synthetic_interaction_indirect_effect_multiseed.py first."
        )
    with open(path) as f:
        payload = json.load(f)
    return payload["per_node"]


def summarize_by_type(per_node: list[dict]) -> dict[str, dict]:
    """Pool per-node correlation and relative-error stats by node type."""
    summaries = {}
    for type_name in sorted({r["type"] for r in per_node}):
        type_results = [r for r in per_node if r["type"] == type_name]
        pooled_stats = [
            {"x": {"r": r["r"], "p": r["p"], "n": r["n"]}} for r in type_results
        ]
        summary = summarize_correlations(pooled_stats, "x")

        rel_err = np.array([r["relative_error"] for r in type_results])
        rel_err = rel_err[~np.isnan(rel_err)]
        summary["median_relative_error"] = (
            float(np.median(rel_err)) if len(rel_err) else None
        )
        summary["mean_relative_error"] = (
            float(np.mean(rel_err)) if len(rel_err) else None
        )
        summary["frac_relative_error_gt_0.5"] = (
            float(np.mean(rel_err > 0.5)) if len(rel_err) else None
        )
        summary["frac_relative_error_gt_1.0"] = (
            float(np.mean(rel_err > 1.0)) if len(rel_err) else None
        )
        summaries[type_name] = summary
    return summaries


def _format_table(summaries: dict[str, dict]) -> str:
    header = (
        "| type | n_nodes | mean\\|r\\| | frac. significant (p<0.05) "
        "| median rel. error | frac >50% off |"
    )
    sep = "|---|---|---|---|---|---|"
    lines = [header, sep]
    for type_name, s in summaries.items():
        lines.append(
            f"| {type_name} | {s['n_nodes']} | {s['mean_abs_r']:.3f} | "
            f"{s['frac_significant']:.1%} | {s['median_relative_error']:.3f} | "
            f"{s['frac_relative_error_gt_0.5']:.1%} |"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Print pooled per-type summary from the interaction "
        "indirect-effect multiseed diagnostic."
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=RESULTS_DIR / "synthetic_interaction_indirect_effect_multiseed",
        help="Directory containing multiseed_results.json.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path to also write the table to a file.",
    )
    args = parser.parse_args()

    per_node = _load_per_node_results(args.results_dir)
    summaries = summarize_by_type(per_node)
    table = _format_table(summaries)
    print(table)

    if args.output:
        args.output.write_text(table + "\n")
        print(f"\nSaved table to: {args.output}")


if __name__ == "__main__":
    main()
