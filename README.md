# Find My Major ML Extension

A Python-based research repository for RIASEC-informed career recommendation experiments. This project is built around a configuration-driven evaluation pipeline, reusable source modules, and export utilities for career matching and model analysis.

## Overview

This repository contains the code and documentation needed to:

- load and validate experiment configuration
- process career and RIASEC-related data
- execute model evaluation workflows
- generate evaluation artifacts and optional frontend export assets
- support reproducible experiment tracking under `experiments/results/`

The project is organized to separate core logic in `src/`, experiment definitions in `experiments/`, and documentation in `docs/`.

## What’s Included

- `main.py` — top-level CLI entrypoint for running experiments
- `src/config/` — configuration loading, validation, and run metadata
- `src/evaluation/` — evaluation orchestration, reporting, and visualization
- `src/export/` — frontend artifact export utilities
- `experiments/config/` — example experiment configuration files
- `environment/requirements.txt` — Python dependencies for development and execution
- `docs/` — repository guidance, architecture rules, and workflow documentation
- `tests/` — unit tests for configuration, data, export, and evaluation logic

## Requirements

- Python 3.11 or later
- Install dependencies:

```bash
pip install -r environment/requirements.txt
```

## Running an Experiment

1. Choose or create a YAML config file under `experiments/config/`. (see `experiments/config/exp_config.yaml` as a template.)
2. Run the pipeline from the repository root:

```bash
python main.py experiments/config/exp_config.yaml
```

The pipeline will:

- validate the experiment configuration
- create a timestamped run directory under `experiments/results/`
- execute the evaluation workflow
- optionally export CSV results, visualizations, and frontend artifacts

### Optional Export

To export frontend inference assets, enable the appropriate export flags in your configuration or use the repository export utilities in `src/export/` and `src/scripts/export_for_frontend.py`.

## Testing

Run the test suite from the repository root:

```bash
pytest
```

## Recommended Workflow

Before modifying code, consult the repository documentation:

- `docs/repo_structure.md`
- `docs/coding_rules.md`
- `docs/ai_usage_policy.md`
- `docs/evaluation_workflow.md`
- `docs/data/data_governance.md`
- `docs/experiments/tracking.md`

These documents describe the intended architecture, development conventions, and experiment tracking expectations.

## Repository Layout

```text
.
├── docs/
├── environment/
├── experiments/
├── main.py
├── models/
├── notebooks/
├── src/
│   ├── config/
│   ├── data/
│   ├── evaluation/
│   ├── export/
│   ├── models/
│   └── scripts/
├── tests/
└── README.md
```

## Status

The repository is structured for research-driven development and reproducible experiment execution. Core runtime entry points, configuration handling, evaluation reporting, and artifact export scaffolding are implemented. Ongoing work focuses on expanding model implementations, data processing,evaluation coverage, and automation of data collection and export integration processes.

## Companion Website

For references, demos, and related repositories, visit:

https://nmeiners.github.io/brand-new-repository/

## AI Usage
This repository encourages responsible AI usage. Please refer to `docs/ai_usage_policy.md` for guidelines on how to use AI tools effectively and ethically in your development process.

All AI-generated code within this repository adheres to the guidelines outlined in the AI usage policy and is documented in code comments for transparency.
