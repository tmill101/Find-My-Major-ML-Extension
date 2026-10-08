"""
File: visualization.py
Path: src/evaluation/visualization.py

Purpose:
  Generates bar charts and a combined metrics overview from evaluation results.
  Called by main.py when output.visual_output is true. Saves PNGs to the run
  output directory alongside evaluation.json.

  The companion notebook notebooks/visualize_results.ipynb contains the same
  logic and can be run interactively in JupyterLab.

Original Author(s):
  - AI Assistant (Claude Sonnet 4.6)

AI Tools Used:
  - Claude Sonnet 4.6 - Code generation and documentation

Editors:
  - AI Assistant (2026-04-13) — Initial implementation

Last Editor:
  - AI Assistant

Last Edit Date:
  2026-04-13

Assumptions & Constraints:
  - evaluation.json exists in the output directory before this module is called
  - matplotlib Agg backend is used (non-interactive, safe for pipeline execution)
  - All metric keys follow the pattern <metric_name>@<k> (e.g. ndcg@5)

Related Docs:
  - docs/src/evaluation/evaluation.md
"""

import json
from pathlib import Path
from typing import Any, Callable, Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _build_dataframe(results: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Name: _build_dataframe

    Purpose:
      Flattens a list of evaluation result dicts into a DataFrame suitable
      for charting. Appends a numeric suffix to the label when multiple
      results share the same model name.

    Inputs:
      - results: List[Dict[str, Any]] — raw evaluation results from evaluation.json

    Outputs:
      - pd.DataFrame — one row per result with flattened metric columns and a
        unique 'label' column for chart x-axis ticks

    Raises / Errors:
      - None

    Notes:
      - Metric keys nested under 'metrics' are promoted to top-level columns.
      - memory_bytes is converted to memory_kb for readability.
    """
    rows = []
    for entry in results:
        row: Dict[str, Any] = {
            "model": entry.get("model", "unknown"),
            "dataset": entry.get("dataset", "unknown"),
            "latency_ms": entry.get("latency_ms", float("nan")),
            "memory_kb": entry.get("memory_bytes", float("nan")) / 1024,
            "model_size_mb": entry.get("model_size_mb", float("nan")),
            "samples_evaluated": entry.get("samples_evaluated", 0),
        }
        row.update(entry.get("metrics", {}))
        rows.append(row)

    df = pd.DataFrame(rows)

    counts = df.groupby("model").cumcount()
    model_counts = df["model"].value_counts()
    df["label"] = df.apply(
        lambda r: r["model"] if model_counts[r["model"]] == 1
                  else f"{r['model']}_{counts[r.name] + 1}",
        axis=1,
    )
    return df


def _bar_chart(
    labels: List[str],
    values: List[float],
    title: str,
    ylabel: str,
    color_map: Callable,
    out_path: Path,
) -> None:
    """
    Name: _bar_chart

    Purpose:
      Renders a single sorted bar chart and saves it as a PNG.

    Inputs:
      - labels: List[str] — x-axis bar labels (model names)
      - values: List[float] — bar heights (metric scores or latency)
      - title: str — chart title
      - ylabel: str — y-axis label
      - color_map: Callable — matplotlib colormap (e.g. plt.cm.Blues)
      - out_path: Path — destination PNG file path

    Outputs:
      - None (writes PNG to out_path)

    Raises / Errors:
      - None

    Notes:
      - Value labels are printed above each bar.
      - Y-axis ceiling is capped at 1.0 for metric scores; uncapped for latency.
    """
    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 0.8), 5))
    colors = color_map(np.linspace(0.4, 0.85, len(labels)))
    bars = ax.bar(labels, values, color=colors[::-1])

    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(values) * 0.01,
            f"{val:.4f}",
            ha="center", va="bottom", fontsize=8,
        )

    ax.set_xlabel("Model")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_ylim(0, min(1.0, max(values) * 1.2) if max(values) <= 1 else max(values) * 1.2)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_path.name}")


def _heatmap(df: pd.DataFrame, metric_cols: List[str], out_path: Path) -> None:
    """
    Name: _heatmap

    Purpose:
      Renders a model × metric heatmap so all scores can be compared in a
      single view without flipping between individual bar charts.

    Inputs:
      - df: pd.DataFrame — flattened results DataFrame from _build_dataframe
      - metric_cols: List[str] — metric column names to include (e.g. ['ndcg@5', 'precision@5'])
      - out_path: Path — destination PNG file path

    Outputs:
      - None (writes PNG to out_path)

    Raises / Errors:
      - None

    Notes:
      - Cell values are printed inside each tile for readability.
      - Rows (models) are sorted by the first metric column descending.
    """
    plot_df = df.set_index("label")[metric_cols].dropna(how="all")
    plot_df = plot_df.sort_values(metric_cols[0], ascending=False)

    fig, ax = plt.subplots(figsize=(max(6, len(metric_cols) * 1.8), max(4, len(plot_df) * 0.6)))
    im = ax.imshow(plot_df.values, aspect="auto", cmap="YlGn", vmin=0, vmax=1)

    ax.set_xticks(range(len(metric_cols)))
    ax.set_xticklabels([c.upper() for c in metric_cols], rotation=30, ha="right")
    ax.set_yticks(range(len(plot_df)))
    ax.set_yticklabels(plot_df.index)

    for row in range(len(plot_df)):
        for col in range(len(metric_cols)):
            val = plot_df.values[row, col]
            if not np.isnan(val):
                ax.text(col, row, f"{val:.4f}", ha="center", va="center",
                        fontsize=8, color="black" if val < 0.7 else "white")

    plt.colorbar(im, ax=ax, fraction=0.03, pad=0.04)
    ax.set_title("Metrics Heatmap — All Models")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_path.name}")


def _scatter_tradeoff(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    x_label: str,
    y_label: str,
    title: str,
    out_path: Path,
) -> None:
    """
    Name: _scatter_tradeoff

    Purpose:
      Renders a labeled scatter plot to visualise the tradeoff between two
      model attributes (e.g. latency vs. accuracy, model size vs. accuracy).
      Models in the top-left corner are best: high accuracy, low cost.

    Inputs:
      - df: pd.DataFrame — flattened results DataFrame from _build_dataframe
      - x_col: str — column name for x-axis (e.g. 'latency_ms')
      - y_col: str — column name for y-axis (e.g. 'ndcg@5')
      - x_label: str — x-axis display label
      - y_label: str — y-axis display label
      - title: str — chart title
      - out_path: Path — destination PNG file path

    Outputs:
      - None (writes PNG to out_path)

    Raises / Errors:
      - None (skipped silently if either column has no valid data)

    Notes:
      - Each point is labelled with the model name.
      - A dashed reference line marks the mean of each axis.
    """
    plot_df = df[["label", x_col, y_col]].dropna()
    if plot_df.empty:
        return

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.scatter(plot_df[x_col], plot_df[y_col], color="#4c72b0", s=80, zorder=3)

    for _, row in plot_df.iterrows():
        ax.annotate(
            row["label"],
            (row[x_col], row[y_col]),
            textcoords="offset points",
            xytext=(6, 4),
            fontsize=8,
        )

    ax.axvline(plot_df[x_col].mean(), color="grey", linestyle="--", linewidth=0.8, alpha=0.6)
    ax.axhline(plot_df[y_col].mean(), color="grey", linestyle="--", linewidth=0.8, alpha=0.6)

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_path.name}")


def generate_visualizations(output_dir: Path) -> None:
    """
    Name: generate_visualizations

    Purpose:
      Loads evaluation.json from output_dir and writes PNG charts into
      the same directory. Generates one bar chart per metric family (NDCG,
      Precision, Recall, Latency), a combined grouped bar chart, a metrics
      heatmap, and tradeoff scatter plots (latency vs accuracy, model size
      vs accuracy).

    Inputs:
      - output_dir: Path — run output directory containing evaluation.json

    Outputs:
      - None (writes PNGs to output_dir)

    Raises / Errors:
      - None (missing or empty results file prints a warning and returns early)

    Notes:
      - Chart filenames use the pattern <metric>_comparison.png.
      - Each call overwrites existing PNGs in output_dir.
      - Tradeoff scatter plots are skipped silently when only one model is present.
    """
    results_file = output_dir / "evaluation.json"
    if not results_file.exists():
        print(f"Warning: {results_file} not found, skipping visualizations.")
        return

    with open(results_file) as f:
        raw: List[Dict[str, Any]] = json.load(f)

    if not raw:
        print("Warning: evaluation.json is empty, skipping visualizations.")
        return

    df = _build_dataframe(raw)

    metric_groups: Dict[str, Any] = {
        "ndcg": plt.cm.Blues,
        "precision": plt.cm.Greens,
        "recall": plt.cm.Oranges,
    }

    for prefix, cmap in metric_groups.items():
        for col in [c for c in df.columns if c.startswith(prefix)]:
            col_df = df[["label", col]].dropna().sort_values(col, ascending=False)
            if col_df.empty:
                continue
            _bar_chart(
                labels=col_df["label"].tolist(),
                values=col_df[col].tolist(),
                title=f"{col.upper()} — Model Comparison",
                ylabel=col.upper(),
                color_map=cmap,
                out_path=output_dir / f"{col.replace('@', '_at_')}_comparison.png",
            )

    lat_df = df[["label", "latency_ms"]].dropna().sort_values("latency_ms")
    if not lat_df.empty:
        _bar_chart(
            labels=lat_df["label"].tolist(),
            values=lat_df["latency_ms"].tolist(),
            title="Inference Latency — Model Comparison",
            ylabel="Latency (ms)",
            color_map=plt.cm.Purples,
            out_path=output_dir / "latency_comparison.png",
        )

    metric_cols = [c for c in df.columns if c.startswith(("ndcg", "precision", "recall"))]
    if metric_cols:
        plot_df = df.set_index("label")[metric_cols].dropna(how="all")
        x = np.arange(len(plot_df))
        width = 0.8 / len(metric_cols)
        palette = ["#4c72b0", "#55a868", "#c44e52", "#8172b2", "#937860"]

        fig, ax = plt.subplots(figsize=(max(10, len(plot_df) * 1.2), 6))
        for i, col in enumerate(metric_cols):
            offset = (i - len(metric_cols) / 2 + 0.5) * width
            ax.bar(x + offset, plot_df[col], width, label=col.upper(),
                   color=palette[i % len(palette)], alpha=0.85)

        ax.set_xticks(x)
        ax.set_xticklabels(plot_df.index, rotation=45, ha="right")
        ax.set_ylabel("Score")
        ax.set_title("All Metrics — Model Comparison")
        ax.set_ylim(0, 1)
        ax.legend()
        plt.tight_layout()

        out_path = output_dir / "all_metrics_comparison.png"
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"  Saved: {out_path.name}")

        _heatmap(df, metric_cols, output_dir / "metrics_heatmap.png")

    primary_metric = next((c for c in df.columns if c.startswith("ndcg")), None)
    if primary_metric:
        _scatter_tradeoff(
            df=df,
            x_col="latency_ms",
            y_col=primary_metric,
            x_label="Latency (ms)",
            y_label=primary_metric.upper(),
            title=f"Latency vs. {primary_metric.upper()} — Accuracy / Speed Tradeoff",
            out_path=output_dir / "tradeoff_latency_vs_accuracy.png",
        )
        _scatter_tradeoff(
            df=df,
            x_col="model_size_mb",
            y_col=primary_metric,
            x_label="Model Size (MB)",
            y_label=primary_metric.upper(),
            title=f"Model Size vs. {primary_metric.upper()} — Efficiency Tradeoff",
            out_path=output_dir / "tradeoff_size_vs_accuracy.png",
        )
