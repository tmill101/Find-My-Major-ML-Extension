# Module: src/evaluation

## Responsibility
Owns evaluation orchestration and reporting for trained recommendation models.
This includes:
- Running model evaluation across datasets
- Scheduling dataset/model experiment jobs via multiprocessing
- Computing rank-based recommendation metrics
- Benchmarking latency/memory/model-size proxies
- Saving/loading formatted evaluation artifacts
- Providing fairness utility functions for category-distribution analysis

This module does NOT implement model training algorithms or raw CSV loading.

## Public Interfaces
- `Dataset(train_records, test_records, feature_columns, label_column='career_category')`
  - `split()` returns train/test DataFrames
- `evaluate_experiment(datasets, models, onet_db, config, output_path=None)`
- `run_single_experiment(dataset_config, model_config, config)`
- `evaluate_model(model, dataset, onet_db, config)`
- `save_evaluation_results(results, output_path)`
- `compute_all_metrics(predictions, ground_truth_category, k_values)`
- `ndcg_at_k(predictions, ground_truth_category, k)`
- `precision_at_k(predictions, ground_truth_category, k)`
- `recall_at_k(predictions, ground_truth_category, k)`
- `cosine_similarity(vector1, vector2)`
- `benchmark_model_inference(model, test_data, onet_db, num_runs=5)`
- `measure_latency_ms(func, *args, **kwargs)`
- `estimate_memory_usage(obj)`
- `get_model_size_mb(model)`
- `format_evaluation_results(results)`
- `save_results_to_file(results, output_dir)`
- `save_results_to_csv(results, output_path)`
- `load_results_from_file(input_dir)`
- Configuration resolution helpers (for custom evaluation workflows):
  - `resolve_parallel_jobs(config)` — validates and resolves max_workers count
  - `resolve_metric_selection(config)` — selects metrics from supported set
  - `validate_k_values(k_values, top_k)` — validates k-values don't exceed top_k
  - `resolve_top_k(evaluation_cfg, model)` — resolves top_k from config or model defaults
  - `filter_metric_results(metric_results, selected_metrics)` — filters computed metrics
- Result building helpers (for custom evaluation workflows):
  - `collect_prediction_contract_violations(predictions, top_n_jobs)` — validates prediction format
  - `merge_violation_counts(aggregate, sample_violations)` — accumulates violation counts
  - `build_model_result(all_metrics, latencies, memory_usages, model, ...)` — aggregates results
  - `sanitize_path_component(value)` — sanitizes identifiers for filesystem paths
  - `write_aggregate_results(results, output_path)` — persists aggregate results to JSON
  - `write_single_experiment_result(result, output_path, run_id, job_index)` — persists per-experiment artifacts
- Fairness utilities:
  - `compute_category_distribution(recommendations, category_names)`
  - `compute_fairness_score(distribution, ideal_distribution)`
  - `detect_overrepresented_categories(distribution, ideal_distribution, threshold=10.0)`
  - `detect_underrepresented_categories(distribution, ideal_distribution, threshold=10.0)`

## Internal Structure
- `evaluator.py` — orchestration loop, worker function, and `Dataset` class
  - Coordinates parallel experiment execution via ProcessPoolExecutor
  - Implements `run_single_experiment()` worker function for process isolation
  - Implements `evaluate_model()` orchestration for single model evaluation
  - Manages `Dataset` class for training/test data handling
  - Delegates configuration resolution to `config_resolution.py`
  - Delegates result building to `result_builders.py`
- `config_resolution.py` — configuration validation and resolution helpers
  - `resolve_parallel_jobs()` — validates and resolves max_workers count
  - `resolve_metric_selection()` — selects metrics from supported set
  - `validate_k_values()` — validates k-values don't exceed top_k
  - `resolve_top_k()` — resolves top_k from config or model defaults
  - `filter_metric_results()` — filters computed metrics by metric family
- `result_builders.py` — result aggregation and persistence helpers
  - `collect_prediction_contract_violations()` — validates prediction format
  - `merge_violation_counts()` — accumulates violation counts
  - `build_model_result()` — aggregates metrics/latency/memory into result dict
  - `sanitize_path_component()` — sanitizes identifiers for filesystem paths
  - `write_aggregate_results()` — persists aggregate results to JSON
  - `write_single_experiment_result()` — persists per-experiment JSON artifacts
- `metrics.py` — recommendation metrics (`ndcg@k`, `precision@k`) + vector similarity helper
- `benchmark.py` — inference latency/memory/model-size approximation
- `reporting.py` — formatting and file persistence helpers
- `fairness.py` — category distribution/fairness helper functions
- `__init__.py` — public re-exports

## Refactoring & Design Decisions

### Modularity Strategy
In April 2026, `evaluator.py` was refactored to improve modularity by extracting configuration and result-building logic to dedicated helper modules:

**Configuration Resolution (`config_resolution.py`)**:
- Responsibility: Validate and resolve evaluation configuration parameters
- Why extracted: Configuration handling is a distinct concern from orchestration
- Benefits: Reusable logic for custom evaluation workflows; cleaner separation of concerns
- Key functions: `resolve_parallel_jobs()`, `resolve_metric_selection()`, `validate_k_values()`, etc.

**Result Building (`result_builders.py`)**:
- Responsibility: Aggregate metrics, validate predictions, build and persist result artifacts
- Why extracted: Result aggregation is orthogonal to evaluation orchestration
- Benefits: Enables custom result formatting; cleaner separation of concerns
- Key functions: `build_model_result()`, `write_aggregate_results()`, contract validation helpers

### Line Count Rationale
`evaluator.py` remains at ~523 lines despite refactoring. This is intentional:
- The module serves as the orchestration coordinator for the evaluation layer
- It contains the `Dataset` class, `run_single_experiment()` worker function, and `evaluate_model()` orchestration logic
- These core responsibilities require substantial code and cannot be further decomposed without losing cohesion
- Extracted helpers are imported and delegated to, not duplicated
- The refactoring prioritized **code organization and modularity** over strict line count reduction
- Configuration_rules.md guideline (~400 lines) applies primarily to business logic functions; orchestration layers may exceed this while maintaining quality

### Import Pattern
`evaluator.py` wrapper functions preserve backward compatibility while delegating to helpers:
- `_resolve_parallel_jobs()` delegates to `config_resolution.resolve_parallel_jobs()`
- `_build_model_result()` delegates to `result_builders.build_model_result()`
- Etc. (11 wrapper functions total)
- This pattern allows breaking up large files without breaking existing code that imports from evaluator

## Data Contracts
- Inputs:
  - `evaluate_experiment` consumes dataset/model matrices from config:
    - `config.datasets`: list of dataset rows
    - `config.models`: list of model rows
  - `config.training.parallel_jobs`: max worker process count (defaults to 2 if missing)
  - `datasets`, `models`, `onet_db` parameters remain for interface compatibility
    but are not used for process job execution
  - Worker-level inputs:
    - `dataset_config`: one entry from `config.datasets`
    - `model_config`: one entry from `config.models`
    - `onet_db_path`: CSV path loaded inside each worker process
  - `config.evaluation` keys currently consumed by evaluator:
    - `metrics` (optional; supported: `ndcg_at_k`, `precision_at_k`; defaults to both)
    - `k_values` (optional, defaults to `[top_k]`; each value must be `<= top_k`)
    - `top_k` (optional, defaults to `5`)
    - `benchmark_runs` (optional, defaults to `1`)
    - `max_test_samples` (optional)
    - `progress_log_interval` (optional, defaults to `1000`; logs per-model sample progress)
- Outputs:
  - Per model/dataset result dict containing:
    - `metrics`
    - `latency_ms`
    - `memory_bytes`
    - `model_size_mb`
    - `constraint_violations`
    - `samples_evaluated`
    - `interrupted` (optional; `true` when user interruption occurs after partial sample evaluation)
    - `run_id` (copied from `config.run.run_id`)
  - Aggregate output remains `evaluation.json` under the run directory.
  - Additional per-job outputs are stored as JSON files under `<run_dir>/per_experiment/`.

## Constraints
- Metric implementations are centralized in `metrics.py`.
- Parallelism is applied only at the experiment job level:
  one `(dataset_config, model_config)` pair per process task.
- No threading is used for experiment scheduling.
- Each worker loads dataset rows and O*NET data locally; no global preloaded dataset objects are shared.
- If a model config contains `parameters.n_jobs`, evaluator forces it to `1`
  before model construction to avoid nested parallelism.
- Job failures are isolated: failed worker futures are logged and skipped while other jobs continue.
- Unsupported values in `evaluation.metrics` raise a configuration error.
- Invalid `k_values` (non-positive or greater than `top_k`) raise a configuration error.
- Evaluation loop computes per-sample metrics and aggregates averages.
- Evaluator enforces model prediction contract: output must be a DataFrame with
  `Title`, `Career Category`, and `Match_Score`.
- `constraint_violations` reports detected prediction contract violations.
- Benchmark memory/size values are approximate and computed with deep DataFrame
  sizing plus serialized model-size estimation when possible.
- Fairness helpers are available in-module and exported, but currently invoked explicitly by callers rather than auto-run in `evaluate_experiment`.

## Related Modules
- `src/models` — provides train/test interfaces
- `src/data` — provides typed records and splits
- `src/config` — provides runtime configuration

## Related Documentation
- `docs/evaluation_workflow.md`
- `docs/src/evaluation/evaluation.md`
- `docs/experiments/tracking.md`
