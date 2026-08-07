"""Aggregate run metrics into CSV and Markdown experiment tables."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

METRICS = (
    "accuracy",
    "ece",
    "binary_brier",
    "multiclass_brier",
    "multiclass_nll",
    "aurc",
    "mean_error_confidence",
    "format_validity",
)


def compare_runs(named_paths: list[tuple[str, Path]], output_csv: Path) -> Path:
    rows = []
    for name, path in named_paths:
        metrics = json.loads(path.read_text(encoding="utf-8"))
        rows.append({"run": name, **{key: metrics.get(key) for key in METRICS}})
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["run", *METRICS])
        writer.writeheader()
        writer.writerows(rows)

    markdown_path = output_csv.with_suffix(".md")
    lines = ["| Run | " + " | ".join(METRICS) + " |", "|---|" + "---:|" * len(METRICS)]
    for row in rows:
        values = [
            "" if row[key] is None else f"{float(row[key]):.4f}"
            for key in METRICS
        ]
        lines.append(f"| {row['run']} | " + " | ".join(values) + " |")
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return markdown_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run", action="append", nargs=2, metavar=("NAME", "METRICS_JSON"), required=True
    )
    parser.add_argument("--output", default="results/experiment_matrix.csv")
    args = parser.parse_args()
    markdown = compare_runs(
        [(name, Path(path)) for name, path in args.run],
        Path(args.output),
    )
    print(markdown)


if __name__ == "__main__":
    main()
