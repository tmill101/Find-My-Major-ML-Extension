"""
File: evaluator.py
Path: src/evaluation/evaluator.py

Purpose:
  Orchestrates the evaluation loop across datasets and models. Manages the
  evaluation workflow including training, testing, metric computation, and
  benchmarking. Delegates configuration resolution and result building to
  specialized helper modules for improved modularity.

Original Author(s):
  - AI Assistant (GitHub Copilot)

AI Tools Used:
  - GitHub Copilot - Code generation and documentation

Editors:
  - AI Assistant (2026-03-23) — Initial implementation
  - AI Assistant (2026-03-30) — Added configurable metric selection enforcement
  - AI Assistant (2026-03-30) — Added evaluation contract checks and k-value validation
  - OpenAI Codex (2026-04-06) — Added ProcessPoolExecutor experiment-level parallelism
  - AI Assistant (2026-04-20) — Refactored for modularity by extracting helpers

Last Editor:
  - AI Assistant

Last Edit Date:
  2026-04-20

Assumptions & Constraints:
  - Models implement BaseModel interface (train, test, get_name)
  - Dataset/model jobs come from config dataset/model matrices
  - Each worker process loads its own dataset and O*NET CSV
  - Evaluation results stored in experiments/results/<id>/evaluation.json
  - Configuration resolution and result building delegated to helper modules

Related Docs:
  - docs/src/evaluation/evaluation.md
  - docs/data_contracts.md
"""

import json
import random
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import List, Dict, Any, Tuple

import numpy as np
import pandas as pd

from src.data.loader import load_training_records, split_training_records
from src.data.schemas import TrainingRecord
from src.evaluation.metrics import compute_all_metrics
from src.evaluation.benchmark import benchmark_model_inference
from src.evaluation.config_resolution import (
    resolve_parallel_jobs, resolve_metric_selection, validate_k_values,
    resolve_top_k, filter_metric_results
)
from src.evaluation.result_builders import (
    collect_prediction_contract_violations, merge_violation_counts,
    build_model_result, sanitize_path_component, write_aggregate_results,
    write_single_experiment_result, _REQUIRED_PREDICTION_COLUMNS
)
from src.models import MODEL_REGISTRY


class EvaluationInterrupted(Exception):
    """Raised when evaluation is interrupted but partial model results exist."""

    def __init__(self, partial_result: Dict[str, Any]):
        super().__init__("Evaluation interrupted with partial model results.")
        self.partial_result = partial_result


# Build the dataset/model matrix used for experiment-level process parallelism.
def _build_experiment_jobs(config: Dict[str, Any]) -> List[Tuple[Dict[str, Any], Dict[str, Any]]]:
    dataset_configs = config.get('datasets')
    if not isinstance(dataset_configs, list) or not dataset_configs:
        raise ValueError("Configuration must contain a non-empty 'datasets' list.")

    model_configs = config.get('models')
    if not isinstance(model_configs, list) or not model_configs:
        raise ValueError("Configuration must contain a non-empty 'models' list.")

    jobs: List[Tuple[Dict[str, Any], Dict[str, Any]]] = []
    for dataset_cfg in dataset_configs:
        if not dataset_cfg.get('enabled', True):
            print(f"Skipping dataset '{dataset_cfg.get('name', '?')}' (enabled: false)", flush=True)
            continue
        for model_cfg in model_configs:
            if not model_cfg.get('enabled', True):
                continue
            model_name = model_cfg.get('model')
            if model_name not in MODEL_REGISTRY:
                print(
                    f"Warning: Unknown model '{model_name}' in config — skipping.",
                    file=sys.stderr,
                    flush=True,
                )
                continue
            jobs.append((dict(dataset_cfg), dict(model_cfg)))

    return jobs


# Resolve a stable human-readable dataset identifier for logging and outputs.
def _resolve_dataset_name(dataset_config: Dict[str, Any], fallback_idx: int) -> str:
    configured_name = dataset_config.get('name')
    if isinstance(configured_name, str) and configured_name.strip():
        return configured_name.strip()
    return f"dataset_{fallback_idx}"


# Resolve a stable model identifier from model config metadata.
def _resolve_model_name(model_config: Dict[str, Any], fallback_idx: int) -> str:
    configured_name = model_config.get('model')
    if isinstance(configured_name, str) and configured_name.strip():
        return configured_name.strip()
    return f"model_{fallback_idx}"


# Instantiate a model from one config row and enforce nested-parallelism safety.
def _build_model_from_config(model_config: Dict[str, Any], config: Dict[str, Any]) -> Any:
    model_name = model_config.get('model')
    if model_name not in MODEL_REGISTRY:
        raise KeyError(f"Unknown model '{model_name}' in config.")

    x_features = model_config.get('x_features')
    if not isinstance(x_features, list) or not x_features:
        raise KeyError(f"Model '{model_name}' must define a non-empty x_features list.")

    y_features = model_config.get('y_features')
    if not isinstance(y_features, list) or not y_features:
        raise KeyError(f"Model '{model_name}' must define a non-empty y_features list.")

    parameters = dict(model_config.get('parameters', {}))
    if 'n_jobs' in parameters:
        parameters['n_jobs'] = 1

    top_n_categories = parameters.pop('top_n_categories', 3)
    top_n_jobs = config.get('evaluation', {}).get('top_k', 5)

    return MODEL_REGISTRY[model_name](
        x_features=x_features,
        y_feature=y_features[0],
        parameters=parameters,
        top_n_jobs=top_n_jobs,
        top_n_categories=top_n_categories,
    )


# Build a Dataset object for one experiment worker from dataset/model configs.
def _build_dataset_from_config(
    dataset_config: Dict[str, Any],
    model_config: Dict[str, Any],
) -> 'Dataset':
    train_path = dataset_config.get('train_path')
    if not train_path:
        raise KeyError("Dataset configuration missing 'train_path'.")

    train_records = load_training_records(train_path)

    test_path = dataset_config.get('test_path')
    if test_path:
        test_records = load_training_records(test_path)
    else:
        split_cfg = dataset_config.get('split') or {}
        val_fraction = float(split_cfg.get('validation', 0.15))
        test_fraction = float(split_cfg.get('test', 0.15))

        train_records, _, test_records = split_training_records(
            train_records,
            val_size=val_fraction,
            test_size=test_fraction,
        )

    if dataset_config.get('shuffle', False):
        random.shuffle(train_records)
        random.shuffle(test_records)

    x_features = model_config.get('x_features')
    if not isinstance(x_features, list) or not x_features:
        raise KeyError("Model configuration missing non-empty 'x_features'.")

    y_features = model_config.get('y_features')
    label_column = y_features[0] if isinstance(y_features, list) and y_features else 'Career Category'

    return Dataset(
        train_records=train_records,
        test_records=test_records,
        feature_columns=x_features,
        label_column=label_column,
    )


# ============================================================================
# Wrapper functions maintaining backward-compatible underscore-prefixed names
# while delegating to imported helper modules. This preserves evaluator.py as
# the orchestration layer without duplicating code across modules.
# ============================================================================

def _resolve_parallel_jobs(config: Dict[str, Any]) -> int:
    """Delegate to config_resolution module."""
    return resolve_parallel_jobs(config)


def _resolve_metric_selection(config: Dict[str, Any]) -> set[str]:
    """Delegate to config_resolution module."""
    return resolve_metric_selection(config)


def _filter_metric_results(
    metric_results: Dict[str, Any],
    selected_metrics: set[str],
) -> Dict[str, Any]:
    """Delegate to config_resolution module."""
    return filter_metric_results(metric_results, selected_metrics)


def _validate_k_values(k_values: Any, top_k: int) -> list[int]:
    """Delegate to config_resolution module."""
    return validate_k_values(k_values, top_k)


def _resolve_top_k(evaluation_cfg: Dict[str, Any], model: Any) -> int:
    """Delegate to config_resolution module."""
    return resolve_top_k(evaluation_cfg, model)


def _collect_prediction_contract_violations(
    predictions: pd.DataFrame,
    top_n_jobs: int,
) -> Dict[str, int]:
    """Delegate to result_builders module."""
    return collect_prediction_contract_violations(predictions, top_n_jobs)


def _merge_violation_counts(
    aggregate: Dict[str, int],
    sample_violations: Dict[str, int],
) -> None:
    """Delegate to result_builders module."""
    return merge_violation_counts(aggregate, sample_violations)


def _build_model_result(
    all_metrics: list[Dict[str, Any]],
    latencies: list[float],
    memory_usages: list[float],
    model: Any,
    constraint_violations: Dict[str, int],
    samples_evaluated: int,
) -> Dict[str, Any]:
    """Delegate to result_builders module."""
    return build_model_result(
        all_metrics, latencies, memory_usages, model,
        constraint_violations, samples_evaluated
    )


def _sanitize_path_component(value: str) -> str:
    """Delegate to result_builders module."""
    return sanitize_path_component(value)


def _write_aggregate_results(results: List[Dict[str, Any]], output_path: Path) -> None:
    """Delegate to result_builders module."""
    return write_aggregate_results(results, output_path)


def _write_single_experiment_result(
    result: Dict[str, Any],
    output_path: Path,
    run_id: str,
    job_index: int,
) -> None:
    """Delegate to result_builders module."""
    return write_single_experiment_result(result, output_path, run_id, job_index)


class Dataset:
    """
    Name: Dataset

    Purpose:
      Wraps training and test data for evaluation. Provides split() method
      and conversion to DataFrames for model training/testing.

    Inputs:
      - train_records: List[TrainingRecord] — training data
      - test_records: List[TrainingRecord] — test data
      - feature_columns: List[str] — RIASEC feature column names
      - label_column: str — career category label column name

    Outputs:
      - split() returns (train_df, test_df) DataFrames

    Raises / Errors:
      - ValueError: if records are empty or feature columns invalid

    Notes:
      - Converts TrainingRecord objects to DataFrames for model compatibility
    """

    def __init__(self, train_records: List[TrainingRecord], test_records: List[TrainingRecord],
                 feature_columns: List[str], label_column: str = 'career_category'):
        if not train_records or not test_records:
            raise ValueError("Train and test records cannot be empty")
        if not feature_columns:
            raise ValueError("Feature columns cannot be empty")

        self.train_records = train_records
        self.test_records = test_records
        self.feature_columns = feature_columns
        self.label_column = label_column

    def split(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Name: split

        Purpose:
          Returns train and test DataFrames for model training/testing.

        Inputs:
          - None

        Outputs:
          - Tuple[pd.DataFrame, pd.DataFrame] — (X_train, X_test) DataFrames

        Raises / Errors:
          - None

        Notes:
          - X_train/X_test contain feature columns only
          - Target labels are separate for training
        """
        # Convert records to DataFrames with configured feature column names
        def record_to_row(r: TrainingRecord) -> Dict[str, Any]:
            row: Dict[str, Any] = {}

            for feature_col in self.feature_columns:
                normalized_feature = feature_col.lower().replace(' ', '_')
                if not hasattr(r, normalized_feature):
                    raise KeyError(f"Training record missing feature '{normalized_feature}'")
                row[feature_col] = getattr(r, normalized_feature)

            row[self.label_column] = r.career_category
            return row

        train_df = pd.DataFrame([record_to_row(r) for r in self.train_records])
        test_df = pd.DataFrame([record_to_row(r) for r in self.test_records])

        return train_df, test_df


def run_single_experiment(
    dataset_config: Dict[str, Any],
    model_config: Dict[str, Any],
    config: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Name: run_single_experiment

    Purpose:
      Executes one complete experiment job for a single (dataset, model) pair.
      Loads data and model from config, trains the model, runs evaluation, and
      returns a structured result payload.

    Inputs:
      - dataset_config: Dict[str, Any] — one dataset row from config['datasets']
      - model_config: Dict[str, Any] — one model row from config['models']
      - config: Dict[str, Any] — full experiment config (run metadata + paths/settings)

    Outputs:
      - Dict[str, Any] — evaluation result for this dataset/model job

    Raises / Errors:
      - KeyError: if required config fields are missing
      - FileNotFoundError: if configured data paths do not exist
      - ValueError: if evaluation configuration values are invalid

    Notes:
      - Worker is process-safe and self-contained (no shared mutable state).
      - If model parameters include `n_jobs`, it is forced to 1 to prevent
        nested process parallelism under ProcessPoolExecutor.
    """
    seed_value = config.get('experiment', {}).get('random_seed')
    if seed_value is not None:
        try:
            seed = int(seed_value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"experiment.random_seed must be an integer if provided, got {seed_value!r}"
            ) from exc
        random.seed(seed)
        np.random.seed(seed)

    onet_db_path = config.get('onet_db_path')
    if not onet_db_path:
        raise KeyError("Configuration missing 'onet_db_path'.")
    onet_db = pd.read_csv(onet_db_path)

    dataset = _build_dataset_from_config(dataset_config, model_config)
    model = _build_model_from_config(model_config, config)

    X_train, _ = dataset.split()
    y_col = getattr(model, 'y_feature', dataset.label_column)
    if y_col not in X_train.columns:
        raise KeyError(f"Training DataFrame missing label column '{y_col}'.")
    y_train = X_train[y_col]

    if hasattr(model, 'reset'):
        model.reset()
    model.train(X_train[dataset.feature_columns], y_train)

    result = evaluate_model(model, dataset, onet_db, config)
    result['dataset'] = _resolve_dataset_name(dataset_config, fallback_idx=0)
    result['model'] = _resolve_model_name(model_config, fallback_idx=0)
    return result


def evaluate_experiment(
    datasets: List[Dataset],
    models: List[Any],
    onet_db: pd.DataFrame,
    config: Dict[str, Any],
    output_path: Path | None = None,
) -> List[Dict[str, Any]]:
    """
    Name: evaluate_experiment

    Purpose:
      Runs the full evaluation matrix across datasets and models using
      process-level parallelism (one process per dataset/model job).

    Inputs:
      - datasets: List[Dataset] — retained for interface compatibility (unused)
      - models: List[Any] — retained for interface compatibility (unused)
      - onet_db: pd.DataFrame — retained for interface compatibility (unused)
      - config: Dict[str, Any] — experiment configuration

    Outputs:
      - List[Dict[str, Any]] — evaluation results for each (dataset, model) pair

    Raises / Errors:
      - ValueError: if parallel_jobs or config matrices are invalid

    Notes:
      - Builds jobs from config['datasets'] × config['models'].
      - Uses ProcessPoolExecutor with max_workers from training.parallel_jobs.
      - Worker failures are isolated; one failed job does not abort other jobs.
    """
    _ = datasets, models, onet_db  # preserve the existing public signature

    jobs = _build_experiment_jobs(config)
    if not jobs:
        return []

    run_id = str(config.get('run', {}).get('run_id', 'run'))
    max_workers = _resolve_parallel_jobs(config)
    results_by_job: Dict[int, Dict[str, Any]] = {}

    futures: Dict[Any, Tuple[int, str, str]] = {}
    try:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            for job_index, (dataset_config, model_config) in enumerate(jobs):
                dataset_name = _resolve_dataset_name(dataset_config, fallback_idx=job_index)
                model_name = _resolve_model_name(model_config, fallback_idx=job_index)
                future = executor.submit(
                    run_single_experiment,
                    dataset_config,
                    model_config,
                    config,
                )
                futures[future] = (job_index, dataset_name, model_name)

            for future in as_completed(futures):
                job_index, dataset_name, model_name = futures[future]
                try:
                    result = future.result()
                except Exception as exc:
                    print(
                        f"Experiment failed for dataset='{dataset_name}', model='{model_name}': {exc}",
                        file=sys.stderr,
                        flush=True,
                    )
                    continue

                result['dataset'] = dataset_name
                result['model'] = model_name
                result['run_id'] = run_id
                results_by_job[job_index] = result

                ordered_results = [results_by_job[idx] for idx in sorted(results_by_job)]
                if output_path is not None:
                    _write_aggregate_results(ordered_results, output_path)
                    _write_single_experiment_result(result, output_path, run_id, job_index)
    except KeyboardInterrupt:
        if output_path is not None:
            ordered_results = [results_by_job[idx] for idx in sorted(results_by_job)]
            _write_aggregate_results(ordered_results, output_path)
        raise

    return [results_by_job[idx] for idx in sorted(results_by_job)]


def evaluate_model(model: Any, dataset: Dataset, onet_db: pd.DataFrame,
                  config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Name: evaluate_model

    Purpose:
      Evaluates a single model on a single dataset.

    Inputs:
      - model: Any — model instance with train/test methods
      - dataset: Dataset — evaluation dataset
      - onet_db: pd.DataFrame — O*NET career database
      - config: Dict[str, Any] — experiment configuration

    Outputs:
      - Dict[str, Any] — evaluation results with metrics, latency, memory, etc.

    Raises / Errors:
      - AttributeError: if model doesn't implement test method
      - ValueError: if unsupported metrics are configured
      - ValueError: if zero test samples are evaluated

    Notes:
      - Computes metrics for each test sample and averages
      - Measures performance benchmarks
    """
    _, X_test = dataset.split()
    evaluation_cfg = config.get('evaluation', {})
    selected_metrics = _resolve_metric_selection(config)
    configured_top_k = _resolve_top_k(evaluation_cfg, model)
    k_values = _validate_k_values(evaluation_cfg.get('k_values', [configured_top_k]), configured_top_k)

    all_metrics = []
    latencies = []
    memory_usages = []
    constraint_violations: Dict[str, int] = {}

    benchmark_runs = evaluation_cfg.get('benchmark_runs', 1)
    if not isinstance(benchmark_runs, int) or benchmark_runs <= 0:
        raise ValueError(
            f"evaluation.benchmark_runs must be a positive integer, got {benchmark_runs!r}"
        )
    max_test_samples = evaluation_cfg.get('max_test_samples')
    if max_test_samples is not None:
        if not isinstance(max_test_samples, int) or max_test_samples <= 0:
            raise ValueError(
                "evaluation.max_test_samples must be null or a positive integer"
            )
    progress_log_interval = evaluation_cfg.get('progress_log_interval', 1000)
    if progress_log_interval is not None:
        if not isinstance(progress_log_interval, int) or progress_log_interval <= 0:
            raise ValueError(
                "evaluation.progress_log_interval must be null or a positive integer"
            )

    # Evaluate each test sample
    label_col = getattr(model, 'y_feature', dataset.label_column)

    samples_evaluated = 0
    for sample_idx, (_, test_row) in enumerate(X_test.iterrows()):
        if max_test_samples is not None and sample_idx >= max_test_samples:
            break
        # Create single-row DataFrame for this test sample
        X_sample = test_row[dataset.feature_columns].to_frame().T
        ground_truth = test_row[label_col]

        # Get predictions and benchmark metrics from model inference
        try:
            benchmark_results = benchmark_model_inference(model, X_sample, onet_db, num_runs=benchmark_runs)
        except KeyboardInterrupt as exc:
            if samples_evaluated > 0:
                partial = _build_model_result(
                    all_metrics=all_metrics,
                    latencies=latencies,
                    memory_usages=memory_usages,
                    model=model,
                    constraint_violations=constraint_violations,
                    samples_evaluated=samples_evaluated,
                )
                raise EvaluationInterrupted(partial) from exc
            raise
        predictions = benchmark_results['predictions']

        if not isinstance(predictions, pd.DataFrame):
            raise TypeError(
                f"{model.get_name()}.test() must return a pandas DataFrame, got {type(predictions)!r}"
            )

        missing_columns = [c for c in _REQUIRED_PREDICTION_COLUMNS if c not in predictions.columns]
        if missing_columns:
            raise ValueError(
                f"{model.get_name()} predictions missing required columns: {missing_columns}"
            )

        sample_violations = _collect_prediction_contract_violations(
            predictions=predictions,
            top_n_jobs=configured_top_k,
        )
        _merge_violation_counts(constraint_violations, sample_violations)

        # Compute metrics
        sample_metrics = compute_all_metrics(predictions, ground_truth, k_values)
        sample_metrics = _filter_metric_results(sample_metrics, selected_metrics)
        all_metrics.append(sample_metrics)

        latencies.append(benchmark_results['latency_ms'])
        memory_usages.append(benchmark_results['memory_bytes'])
        samples_evaluated += 1
        if progress_log_interval and samples_evaluated % progress_log_interval == 0:
            print(
                f"[{model.get_name()}] evaluated {samples_evaluated} sample(s)...",
                flush=True,
            )

    return _build_model_result(
        all_metrics=all_metrics,
        latencies=latencies,
        memory_usages=memory_usages,
        model=model,
        constraint_violations=constraint_violations,
        samples_evaluated=samples_evaluated,
    )


def save_evaluation_results(results: List[Dict[str, Any]], output_path: Path) -> None:
    """
    Name: save_evaluation_results

    Purpose:
      Saves evaluation results to JSON file.

    Inputs:
      - results: List[Dict[str, Any]] — evaluation results
      - output_path: Path — output file path

    Outputs:
      - None (writes to file)

    Raises / Errors:
      - IOError: if file cannot be written

    Notes:
      - Creates output directory if it doesn't exist
      - Stores results in experiments/results/<id>/evaluation.json format
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
