"""
Print a completeness-axiom table from synthetic scaling multi-seed results.

Reports, per method and number of features, how well attributions sum to the
model output difference between input and baseline:

    delta = sum(attributions) - (f(input) - f(baseline))

expressed relative to that output difference, aggregated as WAPE
(sum(|delta|) / sum(|expected_diff|) across samples and seeds) rather than a
mean of per-sample ratios — see `_relative_completeness_delta` in
`experiments/synthetic.py` for why. Values near zero confirm the
completeness axiom holds; larger values flag a numerical approximation
issue (e.g. too few integration steps).

Requires results produced by `run_synthetic_completeness_comparison.py`,
whose CIG/CEG/IG attribution calls pass `return_convergence_delta=True`
(see `experiments/synthetic.py`).
"""

import argparse
from pathlib import Path

from plot_synthetic_scaling import _load_multiseed_summaries

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def _collect_completeness_delta(
    summaries: dict[int, dict],
    methods: list[str] | None = None,
) -> tuple[list[str], dict[int, dict[str, tuple[float, float] | None]]]:
    """Collect mean/std completeness delta per method, keyed by num_features."""
    features_sorted = sorted(summaries.keys())

    method_names = set()
    for payload in summaries.values():
        timings = payload.get("timings_seconds", {})
        for key in timings:
            if key.startswith("completeness_delta."):
                method_names.add(key.split(".", 1)[1])
    if methods is not None:
        method_names &= set(methods)
    method_names = sorted(method_names)

    rows = {}
    for n_features in features_sorted:
        timings = summaries[n_features].get("timings_seconds", {})
        row = {}
        for method in method_names:
            entry = timings.get(f"completeness_delta.{method}")
            row[method] = (
                None
                if entry is None
                else (float(entry.get("mean", 0.0)), float(entry.get("std", 0.0)))
            )
        rows[n_features] = row

    return method_names, rows


def _format_table(
    method_names: list[str], rows: dict[int, dict[str, tuple[float, float] | None]]
) -> str:
    """
    Plain-text scientific notation (e.g. "9.69e-03 ± 6.46e-03") rather than
    percentages — CIG and IG routinely differ by several orders of magnitude,
    and a fixed `.3%` format silently rounds IG's ~1e-7 values to "0.000%".
    """
    header = "| num_features | " + " | ".join(method_names) + " |"
    sep = "|---" * (len(method_names) + 1) + "|"
    lines = [header, sep]
    for n_features, method_values in rows.items():
        cells = []
        for method in method_names:
            value = method_values[method]
            cells.append("—" if value is None else f"{value[0]:.2e} ± {value[1]:.2e}")
        lines.append(f"| {n_features} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _format_scientific_latex(value: float) -> str:
    """Render a float as LaTeX scientific notation, e.g. 1.14e-02 -> '1.14 \\times 10^{-2}'."""
    mantissa, exponent = f"{value:.2e}".split("e")
    return f"{mantissa} \\times 10^{{{int(exponent)}}}"


def _format_latex_table(
    method_names: list[str], rows: dict[int, dict[str, tuple[float, float] | None]]
) -> str:
    """
    Render as a LaTeX table (booktabs style — needs \\usepackage{booktabs}).

    Cells report relative completeness delta as mean $\\pm$ std, in scientific
    notation because CIG and IG typically differ by several orders of
    magnitude (e.g. ~1e-2 vs ~1e-7), which a fixed-decimal or percentage
    format would flatten to 0.000 for the smaller method.
    """
    col_spec = "r" + "c" * len(method_names)
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\begin{tabular}{" + col_spec + "}",
        "\\toprule",
        "num\\_features & " + " & ".join(method_names) + " \\\\",
        "\\midrule",
    ]
    for n_features in sorted(rows):
        method_values = rows[n_features]
        cells = []
        for method in method_names:
            value = method_values[method]
            if value is None:
                cells.append("--")
            else:
                mean_str = _format_scientific_latex(value[0])
                std_str = _format_scientific_latex(value[1])
                cells.append(f"${mean_str} \\pm {std_str}$")
        lines.append(f"{n_features} & " + " & ".join(cells) + " \\\\")
    lines += [
        "\\bottomrule",
        "\\end{tabular}",
        "\\caption{Relative completeness delta "
        "($|\\text{sum(attributions)} - (f(x) - f(\\text{baseline}))| / "
        "|f(x) - f(\\text{baseline})|$), mean $\\pm$ std across seeds.}",
        "\\label{tab:completeness-delta}",
        "\\end{table}",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Print a completeness-delta table (relative to model output "
        "difference) from synthetic scaling multi-seed results."
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=RESULTS_DIR / "synthetic_completeness_comparison",
        help="Directory containing synthetic_causal_metrics_multiseed_*.json files "
        "produced by run_synthetic_completeness_comparison.py.",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=None,
        metavar="METHOD",
        help="Restrict the table to these methods (e.g. --methods CIG CEG IG). "
        "Default: every method with a recorded completeness delta.",
    )
    parser.add_argument(
        "--format",
        choices=["markdown", "latex"],
        default="markdown",
        help="Output format for the table. Default: markdown.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path to also write the table to a file.",
    )
    args = parser.parse_args()

    summaries = _load_multiseed_summaries(args.results_dir)
    if not summaries:
        raise RuntimeError(f"No multiseed result files found in {args.results_dir}")

    method_names, rows = _collect_completeness_delta(summaries, methods=args.methods)
    if not method_names:
        raise RuntimeError(
            "No completeness_delta entries found in results — pass "
            "--results-dir pointing at output from run_synthetic_completeness_comparison.py, "
            "or rerun whatever produced these result files with an updated "
            "compute_attributions if they predate the completeness_delta feature."
        )

    table = (
        _format_latex_table(method_names, rows)
        if args.format == "latex"
        else _format_table(method_names, rows)
    )
    print(table)

    if args.output:
        args.output.write_text(table + "\n")
        print(f"\nSaved table to: {args.output}")


if __name__ == "__main__":
    main()
