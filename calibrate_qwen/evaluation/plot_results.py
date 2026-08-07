"""Render reliability and risk-coverage figures from a metrics JSON file."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def plot_metrics(metrics_path: str | Path, output_dir: str | Path, run_name: str) -> list[Path]:
    metrics = json.loads(Path(metrics_path).read_text(encoding="utf-8"))
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []

    bins = [item for item in metrics["calibration_bins"] if item["count"]]
    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    ax.plot([0, 1], [0, 1], color="#777777", linestyle="--", linewidth=1)
    ax.plot(
        [item["confidence"] for item in bins],
        [item["accuracy"] for item in bins],
        marker="o",
        color="#176B87",
        linewidth=2,
    )
    ax.set(xlabel="Mean confidence", ylabel="Empirical accuracy", xlim=(0, 1), ylim=(0, 1))
    ax.set_title(f"Reliability diagram: {run_name}")
    ax.grid(alpha=0.2)
    reliability_path = output_dir / f"{run_name}_reliability.png"
    fig.tight_layout()
    fig.savefig(reliability_path, dpi=180)
    plt.close(fig)
    paths.append(reliability_path)

    curve = metrics["risk_coverage"]
    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    ax.plot(
        [point["coverage"] for point in curve],
        [point["risk"] for point in curve],
        color="#B23A48",
        linewidth=2,
    )
    ax.set(xlabel="Coverage", ylabel="Risk", xlim=(0, 1), ylim=(0, 1))
    ax.set_title(f"Risk-coverage curve: {run_name}")
    ax.grid(alpha=0.2)
    risk_path = output_dir / f"{run_name}_risk_coverage.png"
    fig.tight_layout()
    fig.savefig(risk_path, dpi=180)
    plt.close(fig)
    paths.append(risk_path)
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", required=True)
    parser.add_argument("--output-dir", default="results/figures")
    parser.add_argument("--run-name", required=True)
    args = parser.parse_args()
    for path in plot_metrics(args.metrics, args.output_dir, args.run_name):
        print(path)


if __name__ == "__main__":
    main()
