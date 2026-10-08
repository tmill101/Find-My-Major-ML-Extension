# Module Documentation: Config Loader

## Module Name

`config_loader`

## Location

```
src/config/config_loader.py
```

## Purpose

The Config Loader module is responsible for reading and validating experiment configuration files used by the machine learning training pipeline. It loads the experiment configuration from `exp_config.yaml` (stored in experiments/config/, see `docs/templates/exp_config.yaml`), performs basic validation, and prepares the configuration object for use by downstream modules.

This module also generates a **unique run identifier** for each experiment execution to ensure experiment results are not overwritten and can be traced to specific runs.

---

# Responsibilities

The Config Loader performs the following tasks:

1. Load experiment configuration files
2. Parse YAML configuration into Python objects
3. Generate unique experiment run identifiers
4. Attach runtime metadata to the configuration
5. Provide the finalized configuration to the training pipeline

---

# Input

Configuration file:

```
experiments/config/exp_config.yaml
```

The configuration file defines:

* experiment metadata
* dataset configurations
* model configurations
* training settings
* evaluation settings
* output locations

---

# Configuration Structure

The configuration file contains several major sections.

## Experiment Metadata

Defines the experiment identity and metadata.

Example:

```yaml
experiment:
  id: exp_001
  description: Baseline model comparison
  author: team_member
  created: 2026-03-14
  random_seed: 42
```

---

## Dataset Matrix

Defines datasets used in training.

Each row represents a dataset configuration.

Example:

```yaml
datasets:

  - name: riasec_dataset
    train_path: docs/data/master_careers_riasec_categories.csv
    test_path: null

    split:
      train: 0.7
      validation: 0.15
      test: 0.15
```

Each dataset row can represent either:

* a single dataset that will be split internally
* separate training and testing datasets

---

## Model Matrix

Defines model training configurations.

Each row represents **one model training configuration**.

Example:

```yaml
models:

  - model: logistic_regression

    parameters:
      max_iter: 1000
      solver: lbfgs
      top_n_categories: 3

    x_features:
      - Realistic
      - Investigative
      - Artistic
      - Social
      - Enterprising
      - Conventional

    y_features:
      - Career Category
```

Each model row defines:

* model type
* hyperparameters
* input features
* target features

Multiple rows may reference the same model with different parameters.

---

## Training Configuration

Defines training pipeline behavior.

Example:

```yaml
training:

  cross_validation_folds: 5
  normalize_features: true
  standardize_features: true
  parallel_jobs: 4
```

`training.parallel_jobs` controls `ProcessPoolExecutor` worker count for
experiment-level parallelism (`datasets × models`) in `src/evaluation/evaluator.py`.
This setting does not enable threading and does not parallelize inside individual
model training methods.

---

## Evaluation Configuration

Defines evaluation metrics and settings.

Example:

```yaml
evaluation:

  metrics:
    - ndcg_at_k
    - precision_at_k
  k_values:
    - 5
    # each k must be <= top_k
  benchmark_runs: 1
  max_test_samples: null
  progress_log_interval: 1000
  top_k: 5
```

`metrics` currently supports `ndcg_at_k` and `precision_at_k`.
`k_values` must contain positive integers less than or equal to `top_k`.

---

## Output Configuration

Defines output locations and what to save.

Example:

```yaml
output:

  directory: experiments/results/exp_001

  save_models: true
  save_predictions: true
  save_metrics: true
```

`save_metrics` is implemented.
`save_models` and `save_predictions` are currently accepted config flags but emit runtime warnings because artifact persistence for those outputs is not yet implemented in `main.py`.

---

# Runtime Metadata Injection

The config loader injects runtime metadata into the configuration dictionary under the `run` key.

## Run ID Generation

A unique run identifier is generated for each execution in the format:

```
<experiment_id>_<YYYYMMDD_HHMMSS>
```

Example:

```
exp_001_20260314_154212
```

The timestamp uses Python's `datetime` module.

## Injected Metadata

```python
config["run"] = {
    "run_id": "<generated_run_id>",
    "start_time": "<ISO_timestamp>"
}
```

This metadata exists only in memory and does not modify the configuration file.

# CLI Usage

The training pipeline can be launched from the command line with a configuration file.

## Command Signature

```
python main.py <config_path>
```

## Example

```
python main.py experiments/config/exp_config.yaml
```

## Requirements

- The config file must exist
- The file must be a valid YAML
- Must contain `experiment.id`

The CLI validates the file existence and loads the configuration before training begins.

# Runtime Output Directory (Pipeline)

`load_config()` injects runtime metadata (`run_id`, `start_time`) only.

`experiment.random_seed` is consumed by `main.py` to seed Python `random` and NumPy for deterministic runs.

The output directory is created by `main.py` using:

```
<output_directory>/<run_id>/
```

Example:

```
experiments/results/exp_001/exp_001_20260314_154212/
```

This keeps configuration loading and filesystem side effects separated.

---

# Public Interface

Primary function:

```python
load_config(config_path: str) -> dict
```

### Parameters

`config_path`

Path to the experiment configuration file.

Example:

```
experiments/config/exp_config.yaml
```

---

### Returns

A dictionary containing the parsed configuration and runtime metadata.

Example:

```python
config = load_config("experiments/config/exp_config.yaml")
```

---

# Usage Example

Example usage in the training pipeline:

```python
from src.config.config_loader import load_config

config = load_config("experiments/config/exp_config.yaml")

run_id = config["run"]["run_id"]
datasets = config["datasets"]
models = config["models"]
```

The training pipeline can then iterate over the dataset and model matrices to generate experiment runs.

---

# Dependencies

Required libraries:

```
PyYAML
datetime
```

Install dependency:

```
pip install pyyaml
```

---

# Future Improvements

Potential enhancements to the config loader include:

* configuration schema validation
* type-safe config classes
* automatic dataset path validation
* experiment reproducibility tracking
* CLI integration for overriding configuration values

---

# Related Modules

Primary modules interacting with the config loader:

```
main.py
src/data/loader.py
src/models/__init__.py
src/evaluation/evaluator.py
```

These modules rely on the configuration object produced by the config loader.
