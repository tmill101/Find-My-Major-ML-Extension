# Module: src/export

## Responsibility
`src/export` implements export pipelines for frontend inference artifacts.
It trains the configured experiment model, exports the trained classifier as an ONNX model,
and writes a lightweight career database JSON file for browser-based ranking.

## Status
The export package now supports config-driven frontend artifact export.
Artifacts are written into the experiment `output.directory` run folder.

## Integration
- `main.py` calls `src.export.export_frontend_artifacts()` when `export.export_inference_model` is enabled.
- `src/scripts/export_for_frontend.py` provides a CLI wrapper for exporting artifacts from a YAML experiment config.
- The module reads the first enabled dataset and model from the experiment config.
- The module relies on `src.models.MODEL_REGISTRY` so the same model contract is used by evaluation and export.

## Configuration
Supported export config keys:
- `export.export_inference_model`: boolean
- `export.format`: string, currently only `onnx`
- `export.save_model_artifact`: boolean, optionally saves the trained sklearn model artifact alongside ONNX
- `export.package_name`: string, ZIP filename for packaged export artifacts
- `export.verify_package`: boolean, performs package validation after export
- `export.upload_script`: string, optional path to a script that receives `--package <path>` after export
- `export.upload_arguments`: list, optional additional CLI arguments forwarded to the upload script

Example:
```yaml
export:
  export_inference_model: true
  format: onnx
  save_model_artifact: false
  package_name: riasec_export_package.zip
  verify_package: true
  upload_script: scripts/upload_export.py
  upload_arguments: ['--target', 'frontend']
```

## Notes
- ONNX export depends on `skl2onnx`, `onnx`, and a compatible `ml_dtypes` package.
- Install the validated export dependency set with:
  `python -m pip install "onnx>=1.20,<1.22" "skl2onnx>=1.20,<1.21" "ml_dtypes>=0.5.3,<0.6"`
- On Python 3.13 and later, this export stack may also require `numpy>=2.1,<3` to satisfy the `ml_dtypes` dependency.
- Runtime errors are raised if required data files or model configuration are missing.
- If `export.save_model_artifact` is enabled, the trained model is also persisted to `models/` inside the run folder.
- The export package includes the ONNX model, career JSON database, and a package manifest.
