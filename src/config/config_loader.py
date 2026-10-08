"""
File: config_loader.py
Path: src/config/config_loader.py

Purpose:
  This file implements the configuration loader module responsible for loading YAML experiment configurations, validating them, and injecting runtime metadata for unique run identification.

Original Author(s):
  - Nathan Meiners
  - AI Assistant

AI Tools Used:
  - GitHub Copilot - Code generation and documentation

Editors:
  - AI Assistant (2026-03-14) — Initial implementation

Last Editor:
  - AI Assistant

Last Edit Date:
  2026-03-14

Assumptions & Constraints:
  - YAML files are small and well-formed
  - No external network access required
  - Run IDs are unique per execution

Related Docs:
  - docs/src/config/config_loader.md
"""

import os
import yaml
from datetime import datetime
from pathlib import Path


def load_config(config_path):
    """
    Name: load_config

    Purpose:
      Loads and validates an experiment configuration from a YAML file, generates a unique run ID, and injects runtime metadata.

    Inputs:
      - config_path: str — Absolute or relative path to the YAML configuration file

    Outputs:
      - dict — The loaded configuration dictionary with injected runtime metadata under 'run' key

    Raises / Errors:
      - FileNotFoundError: If the config file does not exist
      - yaml.YAMLError: If the YAML is malformed
      - KeyError: If required config keys are missing (e.g., 'experiment.id')

    Notes:
      - Generates run_id in format <experiment_id>_<YYYYMMDD_HHMMSS>
      - Injects 'run' dict with 'run_id' and 'start_time' (ISO format)
      - Does not modify the original file
    """
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError("Configuration file must contain a YAML mapping at the top level.")

    if 'experiment' not in config or 'id' not in config['experiment']:
        raise KeyError("Configuration must contain 'experiment.id'")

    if 'output' not in config or 'directory' not in config['output']:
        raise KeyError("Configuration must contain 'output.directory'")

    experiment_id = config['experiment']['id']
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"{experiment_id}_{timestamp}"
    start_time = datetime.now().isoformat()

    output_directory = config['output']['directory']
    run_output_dir = str(Path(output_directory) / run_id)

    config.setdefault('output', {})
    config['output'].setdefault('save_models', False)
    config['output'].setdefault('save_predictions', False)
    config['output'].setdefault('save_metrics', False)
    config['output'].setdefault('save_metrics_csv', False)
    config['output'].setdefault('visual_output', False)

    export_config = config.setdefault('export', {})
    export_config.setdefault('export_inference_model', False)
    export_config.setdefault('format', 'onnx')
    export_config.setdefault('save_model_artifact', False)
    export_config.setdefault('package_name', 'riasec_export_package.zip')
    export_config.setdefault('verify_package', True)
    export_config.setdefault('upload_script', None)
    export_config.setdefault('upload_arguments', [])

    config['run'] = {
        'run_id': run_id,
        'start_time': start_time,
        'output_dir': run_output_dir,
    }

    return config