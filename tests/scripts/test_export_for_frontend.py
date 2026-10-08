"""
File: test_export_for_frontend.py
Path: tests/scripts/test_export_for_frontend.py

Purpose:
  Unit tests for src/scripts/export_for_frontend.py. Validates the wrapper CLI
  interface and ensures export invocation is wired correctly.

Original Author(s):
  - AI Assistant

AI Tools Used:
  - GitHub Copilot - Test generation

Editors:
  - AI Assistant (2026-04-30) — Added wrapper CLI test for export script

Last Editor:
  - AI Assistant

Last Edit Date:
  2026-04-30

Assumptions & Constraints:
  - CLI test uses monkeypatch to isolate file I/O and config loading
  - No actual model export is executed during the wrapper test

Related Docs:
  - docs/src/export/export.md
"""

import sys
from pathlib import Path

import pytest


def test_export_for_frontend_wrapper_invokes_export_and_creates_output(tmp_path, monkeypatch):
    fake_config = {
        'output': {'directory': str(tmp_path)},
        'run': {'run_id': 'run123'},
    }
    fake_export_paths = {
        'onnx_model': tmp_path / 'riasec_model.onnx',
        'frontend_db': tmp_path / 'riasec_jobs_db.json',
    }

    import src.scripts.export_for_frontend as export_script

    monkeypatch.setattr(export_script, 'load_config', lambda config_path: fake_config)
    monkeypatch.setattr(export_script, 'export_frontend_artifacts', lambda cfg, out: fake_export_paths)
    monkeypatch.setattr(sys, 'argv', ['export_for_frontend.py', 'dummy_config.yaml'])

    output_dir = Path(fake_config['output']['directory']) / fake_config['run']['run_id']
    assert not output_dir.exists()

    export_script.main()

    assert output_dir.exists()
    assert output_dir.is_dir()
