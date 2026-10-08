"""
File: __init__.py
Path: src/data/__init__.py

Purpose:
  Public interface for the data module. Exports schemas, API fetcher, and
  local storage functions for O*NET Interest Profiler questions.

Original Author(s):
  - Claude Code

AI Tools Used:
  - Claude Code - initial implementation

Editors:
  - Claude Code (2026-02-16) — initial implementation

Last Editor:
  - Claude Code

Last Edit Date:
  2026-02-16

Assumptions & Constraints:
  - All public interfaces are documented in docs/src/data.md

Related Docs:
  - docs/src/data.md
"""

from src.data.fetch_questions import fetch_questions
from src.data.schemas import AnswerOption, Question, QuestionSet, TrainingRecord, CareerProfile
from src.data.store import load_questions, save_questions
from src.data.loader import load_training_records, load_career_profiles, split_training_records
from src.data.preprocess import records_to_dataframe, extract_features, extract_labels, build_training_matrix
from src.data.validate import validate_training_dataframe, validate_career_dataframe

__all__ = [
    # Question schemas
    "AnswerOption",
    "Question",
    "QuestionSet",
    "fetch_questions",
    "load_questions",
    "save_questions",
    # Training record schemas and loading
    "TrainingRecord",
    "CareerProfile",
    "load_training_records",
    "load_career_profiles",
    "split_training_records",
    # Preprocessing
    "records_to_dataframe",
    "extract_features",
    "extract_labels",
    "build_training_matrix",
    # Validation
    "validate_training_dataframe",
    "validate_career_dataframe",
]
