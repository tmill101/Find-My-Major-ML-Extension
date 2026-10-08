"""
File: __init__.py
Path: src/export/__init__.py

Purpose:
  Exposes frontend export helpers from the export package.

Original Author(s):
  - AI Assistant

AI Tools Used:
  - AI Assistant - documentation alignment

Editors:
  - AI Assistant (2026-04-29) — Added export package entry point

Last Editor:
  - AI Assistant

Last Edit Date:
  2026-04-29

Assumptions & Constraints:
  - Export runtime logic is implemented in src/export/exporter.py

Related Docs:
  - docs/src/export/export.md
"""

from src.export.exporter import export_frontend_artifacts

__all__ = [
    "export_frontend_artifacts",
]
