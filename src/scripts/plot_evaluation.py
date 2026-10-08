"""
File: plot_evaluation.py
Path: src/scripts/plot_evaluation.py

Purpose:
  CLI utility to generate visualizations from evaluation results. Plots metrics,
  compares model performance, and generates publication-ready charts.

Original Author(s):
  - Nathan Meiners
  - AI Assistant

AI Tools Used:
  - GitHub Copilot - Initial implementation
  - Claude Sonnet 4.6 - Documentation

Editors:
  - AI Assistant (2026-04-20) — Added file header and relocated from scripts/ to src/scripts/

Last Editor:
  - AI Assistant

Last Edit Date:
  2026-04-20

Assumptions & Constraints:
  - Input evaluation results must be in JSON format (evaluation.json)
  - Matplotlib and pandas required for visualization
  - Output directory must be writable

Related Docs:
  - docs/evaluation_workflow.md
  - docs/src/evaluation/evaluation.md
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_evaluation_results(input_path: Path) -> pd.DataFrame:
    """
    Name: load_evaluation_results

    Purpose:
      Loads evaluation results from JSON file and normalizes into DataFrame format.

    Inputs:
      - input_path: Path — path to evaluation results directory or file

    Outputs:
      - pd.DataFrame — normalized evaluation results with flattened metric columns

    Raises / Errors:
      - FileNotFoundError: if evaluation.json not found

    Notes:
      - Handles both directory input (looks for evaluation.json) and file input
      - Flattens nested metric structure for easier plotting
    """
    if input_path.is_dir():
        input_path = input_path / "evaluation.json"

    if input_path.suffix.lower() == ".csv":
        return pd.read_csv(input_path)

    if not input_path.exists():
        raise FileNotFoundError(f"Evaluation file not found: {input_path}")

    with input_path.open("r", encoding="utf-8") as handle:
        raw_results = json.load(handle)

    df = pd.json_normalize(raw_results)
    df = df.rename(columns=lambda name: name.replace("metrics.", ""))

    return df


def plot_metric_bars(df: pd.DataFrame, x_col: str, metrics: list[str], title: str, output_path: Path) -> None:
    """
    Name: plot_metric_bars

    Purpose:
      Creates side-by-side bar charts comparing metrics across models or datasets.

    Inputs:
      - df: pd.DataFrame — evaluation results DataFrame
      - x_col: str — column name for x-axis grouping
      - metrics: list[str] — metric column names to plot
      - title: str — chart title
      - output_path: Path — where to save the figure

    Outputs:
      - Saves PNG file to output_path

    Raises / Errors:
      - KeyError: if metric columns not found in DataFrame

    Notes:
      - Creates subplots for multiple metrics
      - Saves to PNG format
    """
    x_labels = df[x_col].astype(str)
    figure, axes = plt.subplots(1, len(metrics), figsize=(6 * len(metrics), 5), constrained_layout=True)
    if len(metrics) == 1:
        axes = [axes]

    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        ax.bar(x_labels, df[metric], color='steelblue', alpha=0.7)
        ax.set_xlabel(x_col)
        ax.set_ylabel(metric)
        ax.set_title(f"{metric}")

    figure.suptitle(title)
    figure.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot evaluation results")
    parser.add_argument("input_dir", help="Path to evaluation results directory")
    parser.add_argument("--output-dir", default="plots", help="Output directory for plots")
    args = parser.parse_args()

    df = load_evaluation_results(Path(args.input_dir))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    # Example: plot NDCG@5 across models
    if "ndcg@5" in df.columns:
        plot_metric_bars(df, "model_name", ["ndcg@5"], "NDCG@5 Comparison", output_dir / "ndcg_comparison.png")
