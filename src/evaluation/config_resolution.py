"""
File: config_resolution.py
Path: src/evaluation/config_resolution.py

Purpose:
  Provides configuration validation and resolution functions for experiment
  evaluation. Handles metric selection, k-value validation, and parallel
  job configuration.

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
  - Configuration must be a valid dictionary with expected keys
  - Metric names must match supported evaluation metrics
  - K-values must not exceed top_k configuration

Related Docs:
  - docs/src/evaluation/evaluation.md
  - docs/data_contracts.md
"""

from typing import Dict, Any


_SUPPORTED_METRICS = {"ndcg_at_k", "precision_at_k", "recall_at_k"}
_DEFAULT_PARALLEL_JOBS = 2


def resolve_parallel_jobs(config: Dict[str, Any]) -> int:
    """
    Normalize configured worker count; defaults to a small safe process pool.

    Inputs:
      - config: Dict[str, Any] — experiment configuration

    Outputs:
      - int — number of parallel jobs to use

    Raises:
      - ValueError: if parallel_jobs is not a positive integer
    """
    configured_jobs = config.get('training', {}).get('parallel_jobs', _DEFAULT_PARALLEL_JOBS)

    if configured_jobs is None:
        return _DEFAULT_PARALLEL_JOBS

    if not isinstance(configured_jobs, int) or configured_jobs <= 0:
        raise ValueError(
            f"training.parallel_jobs must be a positive integer, got {configured_jobs!r}"
        )

    return configured_jobs


def resolve_metric_selection(config: Dict[str, Any]) -> set[str]:
    """
    Normalize configured metric names into the supported evaluator metric set.

    Inputs:
      - config: Dict[str, Any] — experiment configuration

    Outputs:
      - set[str] — set of selected metric names

    Raises:
      - ValueError: if unsupported metrics are configured
    """
    evaluation_cfg = config.get("evaluation", {})
    configured = evaluation_cfg.get("metrics")

    if configured is None:
        return set(_SUPPORTED_METRICS)

    selected = set(configured)
    unsupported = selected - _SUPPORTED_METRICS
    if unsupported:
        unsupported_str = ", ".join(sorted(unsupported))
        supported_str = ", ".join(sorted(_SUPPORTED_METRICS))
        raise ValueError(
            f"Unsupported evaluation metric(s): {unsupported_str}. "
            f"Supported metrics: {supported_str}"
        )

    return selected


def validate_k_values(k_values: Any, top_k: int) -> list[int]:
    """
    Validate and normalize k-value configuration.

    Inputs:
      - k_values: Any — configured k-values
      - top_k: int — maximum k value

    Outputs:
      - list[int] — validated k-values

    Raises:
      - ValueError: if k-values are invalid or exceed top_k
    """
    if not isinstance(top_k, int) or top_k <= 0:
        raise ValueError(f"evaluation.top_k must be a positive integer, got {top_k!r}")

    if not isinstance(k_values, list) or not k_values:
        raise ValueError("evaluation.k_values must be a non-empty list of positive integers")

    normalized: list[int] = []
    for value in k_values:
        if not isinstance(value, int) or value <= 0:
            raise ValueError(
                f"evaluation.k_values must contain only positive integers, got {value!r}"
            )
        if value > top_k:
            raise ValueError(
                f"evaluation.k_values contains {value}, which exceeds evaluation.top_k={top_k}. "
                "Use k_values <= top_k so metrics reflect returned recommendations."
            )
        normalized.append(value)

    return normalized


def resolve_top_k(evaluation_cfg: Dict[str, Any], model: Any) -> int:
    """
    Resolve top_k value from configuration or model defaults.

    Inputs:
      - evaluation_cfg: Dict[str, Any] — evaluation configuration
      - model: Any — model instance

    Outputs:
      - int — resolved top_k value

    Raises:
      - ValueError: if configured top_k is invalid
    """
    configured_top_k = evaluation_cfg.get("top_k")
    if configured_top_k is not None:
        if not isinstance(configured_top_k, int) or configured_top_k <= 0:
            raise ValueError(
                f"evaluation.top_k must be a positive integer, got {configured_top_k!r}"
            )
        return configured_top_k

    model_top_n_jobs = getattr(model, "top_n_jobs", None)
    if isinstance(model_top_n_jobs, int) and model_top_n_jobs > 0:
        return model_top_n_jobs

    return 5


def filter_metric_results(
    metric_results: Dict[str, Any],
    selected_metrics: set[str],
) -> Dict[str, Any]:
    """
    Filter computed metric outputs based on configured metric families.

    Inputs:
      - metric_results: Dict[str, Any] — computed metrics
      - selected_metrics: set[str] — selected metric families

    Outputs:
      - Dict[str, Any] — filtered metrics
    """
    filtered: Dict[str, Any] = {}

    for key, value in metric_results.items():
        if key.startswith("ndcg@") and "ndcg_at_k" in selected_metrics:
            filtered[key] = value
        elif key.startswith("precision@") and "precision_at_k" in selected_metrics:
            filtered[key] = value
        elif key.startswith("recall@") and "recall_at_k" in selected_metrics:
            filtered[key] = value

    return filtered
