# Module Documentation: main.py

## Responsibility

`main.py` is the top-level command-line entrypoint for the machine learning training pipeline. It is responsible for bootstrapping a single experiment run from the repository root, validating the provided configuration, initializing deterministic runtime state, creating output directories, and delegating the actual evaluation work to reusable `src/` modules.

This file does NOT implement training, feature processing, model inference, or evaluation scoring directly. Those responsibilities are delegated to `src/evaluation`, `src/config`, and downstream model modules.

## Public Interfaces

- `main()` — orchestrates CLI parsing, configuration loading, runtime validation, and experiment execution.

## Internal Structure

- Argument parsing using `argparse`
- Configuration loading via `src.config.config_loader.load_config`
- Runtime seed initialization for reproducible behavior with `random` and `numpy`
- Output directory creation using `pathlib.Path`
- Pre-flight validation of critical resources such as `onet_db_path`
- Delegation of experiment execution to `src.evaluation.evaluate_experiment`
- Result persistence via `src.evaluation.reporting`
- Optional visualization orchestration via `src.evaluation.visualization`

## Data Contracts

- Inputs:
  - Command-line arguments
  - Experiment configuration YAML file
  - O*NET database CSV file referenced by `onet_db_path`

- Outputs:
  - Structured evaluation results written to `experiments/results/<experiment_id>/<run_id>/`
  - Optional CSV export when `--export-csv` is provided or `config.output.save_metrics_csv` is enabled
  - Optional visual output when `config.output.visual_output` is enabled

## Constraints

- Must be executed from the repository root.
- Requires a valid configuration file path as the first positional CLI argument.
- `experiment.random_seed`, when provided, must be coercible to an integer.
- `onet_db_path` must point to an existing file before evaluation begins.
- Model artifact persistence and per-sample prediction persistence are accepted config flags, but warnings are raised because those features are not implemented in this pipeline.

## Related Modules

- `src/config/config_loader`
- `src/evaluation/evaluator`
- `src/evaluation/reporting`
- `src/evaluation/visualization`

## Related Documentation

- `docs/repo_structure.md`
- `docs/src/config/config_loader.md`
- `docs/evaluation.md`
