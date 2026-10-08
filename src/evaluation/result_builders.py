"""
File: result_builders.py
Path: src/evaluation/result_builders.py

Purpose:
  Provides utilities for building, validating, and persisting evaluation results.
  Handles model result aggregation, prediction contract validation, and file I/O.

Original Author(s):
  - AI Assistant (GitHub Copilot)

AI Tools Used:
  - GitHub Copilot - Code generation and documentation

Editors:
  - AI Assistant (2026-04-20) — Extracted from evaluator.py for modularity

Last Editor:
  - AI Assistant

Last Edit Date:
  2026-04-20

Assumptions & Constraints:
  - Results must have required fields (Title, Career Category, Match_Score)
  - Model size is computed via src/evaluation/benchmark.py
  - Results are serialized to JSON with indent for readability

Related Docs:
  - docs/src/evaluation/evaluation.md
  - docs/data_contracts.md
"""

import json
import re
from pathlib import Path
from typing import Dict, Any, List

import pandas as pd

from src.evaluation.benchmark import get_model_size_mb


_REQUIRED_PREDICTION_COLUMNS = ("Title", "Career Category", "Match_Score")


def collect_prediction_contract_violations(
    predictions: pd.DataFrame,
    top_n_jobs: int,
) -> Dict[str, int]:
    """
    Validate model predictions against data contract.

    Inputs:
      - predictions: pd.DataFrame — model predictions
      - top_n_jobs: int — expected maximum row count

    Outputs:
      - Dict[str, int] — violation counts (empty if valid)
    """
    violations: Dict[str, int] = {}

    if len(predictions) > top_n_jobs:
        violations["row_count_exceeds_top_n_jobs"] = 1

    null_in_required = int(predictions[list(_REQUIRED_PREDICTION_COLUMNS)].isnull().any().any())
    if null_in_required:
        violations["null_in_required_columns"] = 1

    scores = pd.to_numeric(predictions["Match_Score"], errors="coerce")
    if scores.isnull().any():
        violations["non_numeric_match_score"] = 1
    elif ((scores < 0.0) | (scores > 1.0)).any():
        violations["match_score_out_of_range"] = 1

    return violations


def merge_violation_counts(
    aggregate: Dict[str, int],
    sample_violations: Dict[str, int],
) -> None:
    """
    Merge sample-level violations into aggregate violation counts.

    Inputs:
      - aggregate: Dict[str, int] — accumulator for violations
      - sample_violations: Dict[str, int] — violations from one sample

    Outputs:
      - None (modifies aggregate in-place)
    """
    for key, value in sample_violations.items():
        aggregate[key] = aggregate.get(key, 0) + value


def build_model_result(
    all_metrics: list[Dict[str, Any]],
    latencies: list[float],
    memory_usages: list[float],
    model: Any,
    constraint_violations: Dict[str, int],
    samples_evaluated: int,
) -> Dict[str, Any]:
    """
    Aggregate metrics and benchmarks into a model result dictionary.

    Inputs:
      - all_metrics: list[Dict[str, Any]] — per-sample metrics
      - latencies: list[float] — per-sample latencies (ms)
      - memory_usages: list[float] — per-sample memory (bytes)
      - model: Any — model instance
      - constraint_violations: Dict[str, int] — accumulated violations
      - samples_evaluated: int — count of evaluated samples

    Outputs:
      - Dict[str, Any] — aggregated result

    Raises:
      - ValueError: if no samples were evaluated
    """
    if not all_metrics:
        raise ValueError("No test samples were evaluated. Check max_test_samples and test dataset size.")

    avg_metrics: Dict[str, float] = {}
    for key in all_metrics[0].keys():
        avg_metrics[key] = sum(m[key] for m in all_metrics) / len(all_metrics)

    avg_latency = sum(latencies) / len(latencies)
    avg_memory = sum(memory_usages) / len(memory_usages)
    model_size = get_model_size_mb(model)

    return {
        'metrics': avg_metrics,
        'latency_ms': avg_latency,
        'memory_bytes': avg_memory,
        'model_size_mb': model_size,
        'constraint_violations': constraint_violations,
        'samples_evaluated': samples_evaluated,
    }


def sanitize_path_component(value: str) -> str:
    """
    Sanitize identifiers before using them in filesystem paths.

    Inputs:
      - value: str — identifier to sanitize

    Outputs:
      - str — sanitized identifier safe for use in paths
    """
    sanitized = re.sub(r"[^A-Za-z0-9_-]+", "_", value).strip("_")
    return sanitized or "unknown"


def write_aggregate_results(results: List[Dict[str, Any]], output_path: Path) -> None:
    """
    Persist aggregate evaluation output in run-level evaluation.json.

    Inputs:
      - results: List[Dict[str, Any]] — evaluation results
      - output_path: Path — output file path

    Outputs:
      - None (writes to file)
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, default=str)


def write_single_experiment_result(
    result: Dict[str, Any],
    output_path: Path,
    run_id: str,
    job_index: int,
) -> None:
    """
    Persist each dataset/model pair output as an individual JSON artifact.

    Inputs:
      - result: Dict[str, Any] — experiment result
      - output_path: Path — base output path
      - run_id: str — experiment run ID
      - job_index: int — job index for ordering

    Outputs:
      - None (writes to file)
    """
    dataset_name = sanitize_path_component(str(result.get('dataset', 'dataset')))
    model_name = sanitize_path_component(str(result.get('model', 'model')))
    safe_run_id = sanitize_path_component(run_id)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    per_experiment_dir = output_path.parent / 'per_experiment'
    file_name = f"{job_index:03d}_{dataset_name}_{model_name}.json"
    per_experiment_path = per_experiment_dir / file_name

    per_experiment_dir.mkdir(parents=True, exist_ok=True)
    with open(per_experiment_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, default=str)
