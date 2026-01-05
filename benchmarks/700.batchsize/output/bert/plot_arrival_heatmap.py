#!/usr/bin/env python3
"""
Render a heatmap showing per-batch inference latency as a function of
latency budget (columns) and achieved batch size (rows) for the
arrival-time experiment outputs.

Example:
    python plot_arrival_heatmap.py \
        --inputs result_arrival_50.json result_arrival_100.json ... \
        --output arrival_heatmap.png
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np


def _unwrap_result(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Drill into nested result/output/result structure produced by benchmarks.
    """
    candidate: Any = payload
    for key in ("result", "output", "result"):
        if isinstance(candidate, dict) and key in candidate:
            candidate = candidate[key]
    if not isinstance(candidate, dict) or "batch_profiles" not in candidate:
        raise KeyError("Unable to find batch_profiles in JSON payload.")
    return candidate


def _load_file(path: Path) -> Dict[str, Any]:
    with path.open("r") as fh:
        payload = json.load(fh)
    return _unwrap_result(payload)


def build_heatmap(
    inputs: List[Path],
    latency_budgets: List[int],
) -> Tuple[np.ndarray, List[int]]:
    data: Dict[Tuple[int, int], List[float]] = {}
    budget_set = set(latency_budgets)

    for path in inputs:
        result = _load_file(path)
        budget = int(round(result["simulation"].get("latency_budget_ms", 0)))
        if budget not in budget_set:
            continue

        for profile in result["batch_profiles"]:
            size = int(profile["size"])
            latency_ms = profile["inference_time_us"] / 1000.0
            data.setdefault((size, budget), []).append(latency_ms)

    batch_sizes = sorted({size for (size, _budget) in data.keys()})
    matrix = np.empty((len(batch_sizes), len(latency_budgets)), dtype=float)
    matrix[:] = np.nan
    for i, size in enumerate(batch_sizes):
        for j, budget in enumerate(latency_budgets):
            values = data.get((size, budget))
            if values:
                matrix[i, j] = float(np.median(values))
    return matrix, batch_sizes


def plot_heatmap(
    matrix: np.ndarray,
    batch_sizes: List[int],
    latency_budgets: List[int],
    output: Path,
    cmap: str = "viridis",
):
    fig, ax = plt.subplots(figsize=(8, 4))

    masked = np.ma.masked_invalid(matrix)
    heat = ax.imshow(masked, aspect="auto", cmap=cmap, origin="lower")

    ax.set_xticks(range(len(latency_budgets)))
    ax.set_xticklabels(latency_budgets)
    ax.set_xlabel("Latency budget (ms)")
    ax.set_yticks(range(len(batch_sizes)))
    ax.set_yticklabels(batch_sizes)
    ax.set_ylabel("Batch size")
    ax.set_title("BERT arrival-time latency heatmap")

    cbar = fig.colorbar(heat, ax=ax)
    cbar.set_label("Median inference latency (ms)")

    for i, size in enumerate(batch_sizes):
        for j, budget in enumerate(latency_budgets):
            value = matrix[i, j]
            if np.isnan(value):
                continue
            ax.text(
                j,
                i,
                f"{value:.2f}",
                ha="center",
                va="center",
                color="white" if value > np.nanmean(matrix) else "black",
                fontsize=8,
            )

    fig.tight_layout()
    if output:
        fig.savefig(output, dpi=200)
        print(f"Saved heatmap to {output}")
    else:
        plt.show()


def parse_args():
    parser = argparse.ArgumentParser(description="Plot BERT arrival-time latency heatmap.")
    parser.add_argument(
        "--dir",
        type=Path,
        help="Directory containing result_arrival_*.json files. Defaults to this script's directory.",
    )
    parser.add_argument(
        "--inputs",
        nargs="+",
        type=Path,
        default=[],
        help="Explicit list of JSON files (skips globbing when provided).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional path to save the heatmap (PNG). Displays interactively if omitted.",
    )
    parser.add_argument(
        "--latency-budgets",
        nargs="+",
        type=int,
        default=[50, 100, 300, 600, 900, 1200],
        help="Latency budgets in ms (columns) to include.",
    )
    parser.add_argument("--cmap", default="viridis", help="Matplotlib colormap name.")
    return parser.parse_args()


def main():
    args = parse_args()
    input_paths: List[Path] = args.inputs
    if not input_paths:
        search_dir = args.dir or Path(__file__).parent
        input_paths = sorted(
            p for p in search_dir.glob("result_arrival_*.json") if "old" not in p.stem
        )
        if not input_paths:
            raise FileNotFoundError(f"No result_arrival_*.json files found in {search_dir}")

    matrix, batch_sizes = build_heatmap(input_paths, args.latency_budgets)
    plot_heatmap(matrix, batch_sizes, args.latency_budgets, args.output, args.cmap)


if __name__ == "__main__":
    main()
