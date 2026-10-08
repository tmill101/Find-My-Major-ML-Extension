"""
File: exporter.py
Path: src/export/exporter.py

Purpose:
  Implements export support for frontend inference artifacts. Trains the configured
  experiment model on the configured training dataset, converts the trained
  sklearn estimator to ONNX, and exports a lightweight JSON career database for
  browser consumption.

Original Author(s):
  - AI Assistant

AI Tools Used:
  - GitHub Copilot - Module implementation

Editors:
  - AI Assistant (2026-04-29) - Added export module and config-driven ONNX export
  - AI Assistant (2026-04-30) - Added package and verification support

Last Editor:
  - AI Assistant

Last Edit Date:
  2026-04-30

Assumptions & Constraints:
  - export_inference_model is enabled in config.export
  - Supported export format is ONNX only
  - Training dataset CSV and onet_db_path CSV exist and include expected columns
  - Model registry classes support x_features/y_features the same way as evaluation

Related Docs:
  - docs/src/export/export.md
  - docs/src/config/config_loader.md
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from src.models import MODEL_REGISTRY
from src.scripts.inference_engine import CareerRecommender

EXPORT_MODEL_FILENAME = "riasec_model.onnx"
EXPORT_DB_FILENAME = "riasec_jobs_db.json"
EXPORT_MANIFEST_FILENAME = "package_manifest.json"
DEFAULT_EXPORT_FORMAT = "onnx"
REQUIRED_FRONTEND_COLUMNS = ["O*NET-SOC Code", "Title"]
TRAINING_COLUMN_ALIAS_MAP = {
    "Realistic": "R normalized",
    "Investigative": "I normalized",
    "Artistic": "A normalized",
    "Social": "S normalized",
    "Enterprising": "E normalized",
    "Conventional": "C normalized",
}


class ExporterError(Exception):
    """Raised when export processing fails."""


def export_frontend_artifacts(config: dict[str, Any], output_dir: Path | str) -> dict[str, Path]:
    """
    Name: export_frontend_artifacts

    Purpose:
      Exports frontend artifacts for an experiment run when export is enabled.

    Inputs:
      - config: dict[str, Any] � loaded experiment configuration
      - output_dir: Path | str � directory where frontend artifacts will be written

    Outputs:
      - dict[str, Path] � paths to the generated ONNX model, frontend JSON database, and package

    Raises / Errors:
      - KeyError: if required configuration fields are missing
      - FileNotFoundError: if configured data files cannot be found
      - RuntimeError: if export verification or upload fails
      - ImportError: if skl2onnx is not installed for ONNX export
    """
    export_cfg = config.get('export', {})
    enabled = export_cfg.get('export_inference_model', False)
    if not enabled:
        return {}

    export_format = str(export_cfg.get('format', DEFAULT_EXPORT_FORMAT)).strip().lower()
    if export_format != DEFAULT_EXPORT_FORMAT:
        raise ValueError(
            f"Unsupported export format '{export_format}'. Only '{DEFAULT_EXPORT_FORMAT}' is supported."
        )

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("Starting export of frontend inference artifacts...", flush=True)

    dataset_config = _get_first_enabled_dataset(config)
    model_config = _get_first_enabled_model(config)
    train_path = dataset_config.get('train_path')
    if not train_path:
        raise KeyError("Enabled dataset configuration is missing 'train_path'.")
    train_path = Path(train_path)
    if not train_path.is_file():
        raise FileNotFoundError(f"Training dataset file not found: {train_path}")

    print(f"Reading training data from: {train_path}", flush=True)
    training_df = pd.read_csv(train_path)
    training_df = _normalize_training_dataframe_columns(training_df, model_config.get('x_features', []))
    print(
        "Training dataset columns after normalization: "
        f"{list(training_df.columns)}",
        flush=True,
    )

    onet_db_path = config.get('onet_db_path')
    if not onet_db_path:
        raise KeyError("Configuration missing 'onet_db_path'.")
    onet_db_path = Path(onet_db_path)
    if not onet_db_path.is_file():
        raise FileNotFoundError(f"O*NET DB file not found: {onet_db_path}")

    frontend_df = pd.read_csv(onet_db_path)

    print(f"Building export model '{model_config.get('model')}'", flush=True)
    model = _build_model_from_config(model_config, config)
    y_feature = model_config['y_features'][0]
    if y_feature not in training_df.columns:
        raise KeyError(
            f"Training dataset does not contain configured label column '{y_feature}'."
        )

    missing_features = [f for f in model.x_features if f not in training_df.columns]
    if missing_features:
        raise KeyError(
            f"Training dataset missing required feature columns: {missing_features}"
        )

    print(f"Training export model with features: {model.x_features}", flush=True)
    model.train(training_df[model.x_features], training_df[y_feature])
    sklearn_estimator = _get_wrapped_sklearn_estimator(model)

    model_artifact_path = None
    save_model_artifact = export_cfg.get(
        'save_model_artifact',
        config.get('output', {}).get('save_models', False),
    )
    if save_model_artifact:
        model_artifact_path = _save_trained_model_artifact(
            sklearn_estimator,
            output_path,
            model.get_name(),
        )

    onnx_path = output_path / EXPORT_MODEL_FILENAME
    print(f"Exporting ONNX model to: {onnx_path}", flush=True)
    _save_model_to_onnx(sklearn_estimator, model.x_features, onnx_path)

    json_path = output_path / EXPORT_DB_FILENAME
    print(f"Writing frontend JSON database to: {json_path}", flush=True)
    _save_frontend_json(frontend_df, model.x_features, y_feature, json_path)

    package_name = export_cfg.get('package_name', 'riasec_export_package.zip')
    package_path = output_path / package_name
    print(f"Packaging export artifacts to: {package_path}", flush=True)
    _package_exported_artifacts(
        onnx_path=onnx_path,
        json_path=json_path,
        model_artifact_path=model_artifact_path,
        package_path=package_path,
        model_name=model.get_name(),
        config=config,
    )

    if export_cfg.get('verify_package', True):
        print("Verifying export package...", flush=True)
        _verify_export_package(package_path)
        print("Export package verification complete.", flush=True)

    upload_script = export_cfg.get('upload_script')
    upload_args = export_cfg.get('upload_arguments', [])
    if upload_script:
        _invoke_upload_script(package_path, upload_script, upload_args)

    results = {
        'onnx_model': onnx_path,
        'frontend_db': json_path,
        'export_package': package_path,
    }
    if model_artifact_path is not None:
        results['trained_model'] = model_artifact_path

    return results


def _get_first_enabled_dataset(config: dict[str, Any]) -> dict[str, Any]:
    for dataset_cfg in config.get('datasets', []):
        if dataset_cfg.get('enabled', True):
            return dataset_cfg
    raise ValueError("Experiment configuration must include at least one enabled dataset.")


def _get_first_enabled_model(config: dict[str, Any]) -> dict[str, Any]:
    for model_cfg in config.get('models', []):
        if model_cfg.get('enabled', True):
            return model_cfg
    raise ValueError("Experiment configuration must include at least one enabled model.")


def _build_model_from_config(model_config: dict[str, Any], config: dict[str, Any]) -> Any:
    model_name = model_config.get('model')
    if model_name not in MODEL_REGISTRY:
        raise KeyError(f"Unknown model '{model_name}' in export config.")

    x_features = model_config.get('x_features')
    if not isinstance(x_features, list) or not x_features:
        raise KeyError(f"Model '{model_name}' must define a non-empty x_features list.")

    y_features = model_config.get('y_features')
    if not isinstance(y_features, list) or not y_features:
        raise KeyError(f"Model '{model_name}' must define a non-empty y_features list.")

    parameters = dict(model_config.get('parameters', {}))
    parameters.pop('n_jobs', None)
    top_n_categories = parameters.pop('top_n_categories', 3)
    top_n_jobs = config.get('evaluation', {}).get('top_k', 5)

    return MODEL_REGISTRY[model_name](
        x_features=x_features,
        y_feature=y_features[0],
        parameters=parameters,
        top_n_jobs=top_n_jobs,
        top_n_categories=top_n_categories,
    )


def _get_wrapped_sklearn_estimator(model: Any) -> Any:
    if hasattr(model, '_model') and model._model is not None:
        return model._model
    if hasattr(model, 'model') and getattr(model, 'model') is not None:
        return getattr(model, 'model')
    raise RuntimeError(
        "Cannot export model to ONNX because no underlying sklearn estimator was found."
    )


def _normalize_training_dataframe_columns(
    training_df: pd.DataFrame,
    x_features: list[str],
) -> pd.DataFrame:
    if set(x_features).issubset(training_df.columns):
        return training_df

    rename_map: dict[str, str] = {}
    for feature in x_features:
        alternate = TRAINING_COLUMN_ALIAS_MAP.get(feature)
        if alternate and alternate in training_df.columns:
            rename_map[alternate] = feature

    if rename_map:
        print(
            "Normalizing training dataset column aliases: "
            f"{rename_map}",
            flush=True,
        )
        training_df = training_df.rename(columns=rename_map)

    return training_df


def _save_trained_model_artifact(sklearn_model: Any, output_path: Path, model_name: str) -> Path:
    from joblib import dump

    artifact_dir = output_path / 'models'
    artifact_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = artifact_dir / f"{model_name}_trained.pkl"
    dump(sklearn_model, artifact_path)
    return artifact_path


def _save_model_to_onnx(sklearn_model: Any, feature_columns: list[str], output_path: Path) -> None:
    python_version_note = ""
    if sys.version_info >= (3, 13):
        python_version_note = (
            "On Python 3.13+ this stack may also require numpy>=2.1,<3. "
            "If your environment pins numpy <2, update it before installing ONNX export dependencies.\n"
        )

    try:
        from skl2onnx import convert_sklearn
        from skl2onnx.common.data_types import FloatTensorType
    except (ImportError, AttributeError) as exc:
        message = (
            "Exporting to ONNX requires compatible skl2onnx, onnx, and ml_dtypes packages. "
            "A mismatch was detected while importing the ONNX export dependencies. "
        )
        if "ml_dtypes" in str(exc) or "float4_e2m1fn" in str(exc):
            message += (
                "This appears to be caused by an incompatible ml_dtypes/onnx combination. "
                "Install a validated set of versions with:\n"
                "  python -m pip install \"onnx>=1.20,<1.22\" \"skl2onnx>=1.20,<1.21\" \"ml_dtypes>=0.5.3,<0.6\"\n"
                f"{python_version_note}"
                "or use the repository environment requirements."
            )
        else:
            message += (
                "Install a validated set of versions with:\n"
                "  python -m pip install \"onnx>=1.20,<1.22\" \"skl2onnx>=1.20,<1.21\" \"ml_dtypes>=0.5.3,<0.6\"\n"
                f"{python_version_note}"
                "or use the repository environment requirements."
            )
        raise ImportError(message) from exc

    initial_type = [('float_input', FloatTensorType([None, len(feature_columns)]))]
    try:
        onx = convert_sklearn(sklearn_model, initial_types=initial_type)
    except (ImportError, AttributeError) as exc:
        message = (
            "Exporting to ONNX requires compatible skl2onnx, onnx, and ml_dtypes packages. "
            "A mismatch was detected while converting the model to ONNX. "
        )
        if "ml_dtypes" in str(exc) or "float4_e2m1fn" in str(exc):
            message += (
                "This appears to be caused by an incompatible ml_dtypes/onnx combination. "
                "Install a validated set of versions with:\n"
                "  python -m pip install \"onnx>=1.20,<1.22\" \"skl2onnx>=1.20,<1.21\" \"ml_dtypes>=0.5.3,<0.6\"\n"
                f"{python_version_note}"
                "or use the repository environment requirements."
            )
        else:
            message += (
                "Install a validated set of versions with:\n"
                "  python -m pip install \"onnx>=1.20,<1.22\" \"skl2onnx>=1.20,<1.21\" \"ml_dtypes>=0.5.3,<0.6\"\n"
                f"{python_version_note}"
                "or use the repository environment requirements."
            )
        raise ImportError(message) from exc

    with open(output_path, 'wb') as model_file:
        model_file.write(onx.SerializeToString())


def _save_frontend_json(
    frontend_df: pd.DataFrame,
    feature_columns: list[str],
    label_column: str,
    output_path: Path,
) -> None:
    required_columns = [*REQUIRED_FRONTEND_COLUMNS, label_column, *feature_columns]
    missing_frontend_cols = [c for c in required_columns if c not in frontend_df.columns]
    if missing_frontend_cols:
        raise KeyError(
            f"Frontend database is missing required columns: {missing_frontend_cols}"
        )

    output_df = frontend_df[[*REQUIRED_FRONTEND_COLUMNS, label_column, *feature_columns]]
    output_df.to_json(output_path, orient='records')


def _package_exported_artifacts(
    onnx_path: Path,
    json_path: Path,
    model_artifact_path: Path | None,
    package_path: Path,
    model_name: str,
    config: dict[str, Any],
) -> None:
    metadata = {
        'package_name': package_path.name,
        'model_name': model_name,
        'run_id': config.get('run', {}).get('run_id'),
        'exported_at': datetime.now().isoformat(),
        'format': config.get('export', {}).get('format', DEFAULT_EXPORT_FORMAT),
    }

    with zipfile.ZipFile(package_path, mode='w', compression=zipfile.ZIP_DEFLATED) as archive:
        archive.write(onnx_path, onnx_path.name)
        archive.write(json_path, json_path.name)
        if model_artifact_path is not None:
            archive.write(model_artifact_path, model_artifact_path.name)
        archive.writestr(EXPORT_MANIFEST_FILENAME, json.dumps(metadata, indent=2))


def _verify_export_package(package_path: Path) -> None:
    if not package_path.is_file():
        raise FileNotFoundError(f"Expected export package not found: {package_path}")

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir_path = Path(temp_dir)
        with zipfile.ZipFile(package_path, mode='r') as archive:
            archive.extractall(temp_dir_path)

        extracted_model = temp_dir_path / EXPORT_MODEL_FILENAME
        extracted_db = temp_dir_path / EXPORT_DB_FILENAME
        if not extracted_model.is_file() or not extracted_db.is_file():
            raise ExporterError(
                "Packaged export is malformed: missing ONNX model or frontend database."
            )

        try:
            recommender = CareerRecommender(
                model_path=str(extracted_model),
                db_path=str(extracted_db),
            )
            recommender.get_top_n_recommendations([0.5, 0.5, 0.5, 0.5, 0.5, 0.5], top_n=1)
        except Exception as exc:
            raise RuntimeError(
                f"Export package verification failed: {exc}"
            ) from exc


def _invoke_upload_script(package_path: Path, upload_script: str, upload_args: list[Any]) -> None:
    script_path = Path(upload_script)
    if not script_path.is_file():
        print(
            f"Upload script configured but not found: {script_path}. "
            "Package generation completed but upload was skipped."
        )
        return

    command = [sys.executable, str(script_path), '--package', str(package_path), *upload_args]
    subprocess.run(command, check=True)
