"""
File: __init__.py
Path: src/evaluation/__init__.py

Purpose:
  Exposes public interfaces for the evaluation module.

Original Author(s):
  - AI Assistant (GitHub Copilot)

AI Tools Used:
  - GitHub Copilot - Code generation and documentation

Editors:
  - AI Assistant (2026-03-23) — Initial implementation

Last Editor:
  - AI Assistant

Last Edit Date:
  2026-03-23

Assumptions & Constraints:
  - Only public interfaces are exposed
  - Internal modules are not imported

Related Docs:
  - docs/src/evaluation/evaluation.md
"""

from .evaluator import evaluate_experiment, evaluate_model, Dataset, save_evaluation_results
from .metrics import compute_all_metrics, cosine_similarity, ndcg_at_k, precision_at_k, recall_at_k
from .benchmark import benchmark_model_inference, get_model_size_mb
from .reporting import format_evaluation_results, save_results_to_file, save_results_to_csv, load_results_from_file
from .fairness import (compute_category_distribution, compute_fairness_score,
                      detect_overrepresented_categories, detect_underrepresented_categories)
from .config_resolution import (resolve_parallel_jobs, resolve_metric_selection, validate_k_values,
                                resolve_top_k, filter_metric_results)
from .result_builders import (collect_prediction_contract_violations, merge_violation_counts,
                              build_model_result, sanitize_path_component, write_aggregate_results,
                              write_single_experiment_result)

__all__ = [
    # Core orchestration
    'evaluate_experiment',
    'evaluate_model',
    'Dataset',
    'save_evaluation_results',
    # Metrics
    'compute_all_metrics',
    'cosine_similarity',
    'ndcg_at_k',
    'precision_at_k',
    'recall_at_k',
    # Benchmarking
    'benchmark_model_inference',
    'get_model_size_mb',
    # Reporting
    'format_evaluation_results',
    'save_results_to_file',
    'save_results_to_csv',
    'load_results_from_file',
    # Fairness analysis
    'compute_category_distribution',
    'compute_fairness_score',
    'detect_overrepresented_categories',
    'detect_underrepresented_categories',
    # Configuration and result building helpers
    'resolve_parallel_jobs',
    'resolve_metric_selection',
    'validate_k_values',
    'resolve_top_k',
    'filter_metric_results',
    'collect_prediction_contract_violations',
    'merge_violation_counts',
    'build_model_result',
    'sanitize_path_component',
    'write_aggregate_results',
    'write_single_experiment_result',
]