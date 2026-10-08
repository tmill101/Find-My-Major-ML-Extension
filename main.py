"""
File: main.py
Path: main.py

Purpose:
  Entry point for the machine learning training pipeline. Handles command-line argument parsing, configuration loading, and initiates the training process.

Original Author(s):
  - Nathan Meiners
  - AI Assistant

AI Tools Used:
  - GitHub Copilot - Code generation and documentation
  - Claude Sonnet 4.6 - Training loop implementation

Editors:
  - AI Assistant (2026-03-14) — Initial implementation
  - AI Assistant (2026-03-20) — Added config-driven model training loop
  - AI Assistant (2026-03-30) — Added deterministic seed initialization and output flag warnings
  - OpenAI Codex (2026-04-06) — Delegated dataset/model execution to evaluator multiprocessing workers
  - AI Assistant (2026-04-13) — Added visual_output flag support via generate_visualizations

Last Editor:
  - AI Assistant

Last Edit Date:
  2026-04-13

Assumptions & Constraints:
  - Executed from repository root
  - Config file exists and is valid
  - Output directory is writable
  - onet_db_path in config points to a valid CSV file

Related Docs:
  - docs/src/config/config_loader.md
  - docs/src/models.md
"""

import argparse
import os
import sys
import random
from pathlib import Path

import pandas as pd
import numpy as np
from src.config.config_loader import load_config
from src.evaluation import evaluate_experiment
from src.evaluation.reporting import save_results_to_file, save_results_to_csv
from src.evaluation.visualization import generate_visualizations
from src.export import export_frontend_artifacts


def main():
    """
    Name: main

    Purpose:
      Parses command-line arguments, loads the experiment configuration, and
      executes evaluation across the dataset/model matrix defined in config.
      Execution orchestration is handled by src.evaluation.evaluator.

    Inputs:
      - Command-line arguments: config_path (positional)

    Outputs:
      - list[dict] — evaluation result payloads for completed experiment jobs

    Raises / Errors:
      - SystemExit: if config file does not exist or loading fails
      - KeyError: if config is missing required keys (e.g., onet_db_path)

    Notes:
      - Dataset/model loading and model training occur inside evaluator workers.
      - Main only validates critical paths and handles output persistence.
    """
    parser = argparse.ArgumentParser(description="Run ML training pipeline with config file.")
    parser.add_argument('config_path', help='Path to the experiment configuration YAML file')
    parser.add_argument('--export-csv', action='store_true', default=False,
                        help='Export evaluation results to CSV after run')

    args = parser.parse_args()

    config_path = args.config_path

    if not os.path.isfile(config_path):
        print(f"Error: Configuration file '{config_path}' does not exist.", file=sys.stderr)
        sys.exit(1)

    try:
        config = load_config(config_path)
    except Exception as e:
        print(f"Error loading configuration: {e}", file=sys.stderr)
        sys.exit(1)

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

    # Create output directory (per docs: experiments/results/<experiment_id>/<run_id>/)
    base_dir = config['output']['directory']
    run_id = config['run']['run_id']

    # Ensure user output path exists
    output_dir = Path(base_dir) / run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Configuration loaded successfully. Run ID: {run_id}")
    print(f"Output directory created: {output_dir}")

    # Validate O*NET database path once before spawning workers.
    onet_db_path = config.get('onet_db_path')
    if not onet_db_path:
        raise KeyError("Configuration missing 'onet_db_path'.")

    if not os.path.isfile(onet_db_path):
        raise FileNotFoundError(f"O*NET DB file not found: {onet_db_path}")

    dataset_count = sum(1 for d in config.get('datasets', []) if d.get('enabled', True))
    model_count = sum(1 for m in config.get('models', []) if m.get('enabled', True))
    print(f"\n{dataset_count} dataset config(s), {model_count} model config(s) discovered.")

    # Run evaluation across all datasets and models
    evaluation_results_path = output_dir / 'evaluation.json'
    try:
        evaluation_results = evaluate_experiment(
            datasets=[],
            models=[],
            onet_db=pd.DataFrame(),
            config=config,
            output_path=evaluation_results_path,
        )
    except KeyboardInterrupt:
        print("\nEvaluation interrupted by user.", file=sys.stderr)
        if evaluation_results_path.exists():
            print(f"Partial results preserved at {evaluation_results_path}", file=sys.stderr)
        else:
            print("No partial evaluation results were written yet.", file=sys.stderr)
        sys.exit(130)

    if config['output'].get('save_metrics', False):
        output_path = save_results_to_file(
            evaluation_results,
            output_dir=output_dir,
        )
        print(f"Evaluation results saved to {output_path}")

    if config['output'].get('save_metrics_csv', False) or args.export_csv:
        csv_path = save_results_to_csv(
            evaluation_results,
            output_dir=output_dir,
        )
        print(f"Evaluation CSV exported to {csv_path}")

    if config['output'].get('save_models', False):
        print(
            "Warning: output.save_models=true but model artifact persistence "
            "is not implemented in this pipeline."
        )

    if config['output'].get('save_predictions', False):
        print(
            "Warning: output.save_predictions=true but per-sample prediction "
            "persistence is not implemented in this pipeline."
        )

    if config.get('export', {}).get('export_inference_model', False):
        print("Exporting frontend inference artifacts ...")
        export_paths = export_frontend_artifacts(config, output_dir)
        print(
            f"Frontend artifacts written to: {export_paths['onnx_model']} "
            f"and {export_paths['frontend_db']}"
        )

    if config['output'].get('visual_output', False):
        print("Generating visualizations ...")
        generate_visualizations(output_dir)

    print(f"\nExperiment complete. Total results: {len(evaluation_results)}")

    return evaluation_results


if __name__ == "__main__":
    main()
