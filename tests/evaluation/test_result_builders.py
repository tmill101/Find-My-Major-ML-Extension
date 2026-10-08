"""
File: test_result_builders.py
Path: tests/evaluation/test_result_builders.py

Purpose:
  Unit tests for result aggregation and persistence helpers extracted from evaluator.py.
  Tests validation, aggregation, and file writing functionality.

Original Author(s):
  - AI Assistant (GitHub Copilot)

AI Tools Used:
  - GitHub Copilot - Code generation and documentation

Editors:
  - AI Assistant (2026-04-20) — Initial implementation

Last Editor:
  - AI Assistant

Last Edit Date:
  2026-04-20

Assumptions & Constraints:
  - Tests use standard unittest and mock fixtures
  - No external dependencies required
  - Tests verify contract of helper functions

Related Docs:
  - docs/src/evaluation/evaluation.md
"""

import unittest
import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import pandas as pd

from src.evaluation.result_builders import (
    collect_prediction_contract_violations,
    merge_violation_counts,
    build_model_result,
    sanitize_path_component,
    write_aggregate_results,
    write_single_experiment_result,
    _REQUIRED_PREDICTION_COLUMNS,
)


class TestCollectPredictionContractViolations(unittest.TestCase):
    """Test prediction contract validation."""

    def test_valid_predictions(self):
        """Test valid predictions have zero violations."""
        predictions = pd.DataFrame({
            'Title': ['Job1', 'Job2'],
            'Career Category': ['Arts', 'Engineering'],
            'Match_Score': [0.9, 0.8],
        })
        result = collect_prediction_contract_violations(predictions, top_n_jobs=2)
        self.assertEqual(result, {})

    def test_more_rows_than_top_n_jobs(self):
        """Test violation when predictions have more rows than top_n_jobs."""
        predictions = pd.DataFrame({
            'Title': ['Job1', 'Job2', 'Job3'],
            'Career Category': ['Arts', 'Engineering', 'Math'],
            'Match_Score': [0.9, 0.8, 0.7],
        })
        result = collect_prediction_contract_violations(predictions, top_n_jobs=2)
        self.assertIn('row_count_exceeds_top_n_jobs', result)
        self.assertEqual(result['row_count_exceeds_top_n_jobs'], 1)

    def test_missing_required_column(self):
        """Test violation when required column is missing (should raise KeyError)."""
        predictions = pd.DataFrame({
            'Title': ['Job1', 'Job2'],
            'Career Category': ['Arts', 'Engineering'],
            # Missing 'Match_Score'
        })
        with self.assertRaises(KeyError):
            collect_prediction_contract_violations(predictions, top_n_jobs=2)

    def test_null_values_in_required_column(self):
        """Test violation when required column has null values."""
        predictions = pd.DataFrame({
            'Title': ['Job1', None],
            'Career Category': ['Arts', 'Engineering'],
            'Match_Score': [0.9, 0.8],
        })
        result = collect_prediction_contract_violations(predictions, top_n_jobs=2)
        self.assertIn('null_in_required_columns', result)

    def test_empty_predictions_dataframe(self):
        """Test empty predictions DataFrame returns no violations (no rows, no contract violation)."""
        predictions = pd.DataFrame({
            'Title': [],
            'Career Category': [],
            'Match_Score': [],
        })
        result = collect_prediction_contract_violations(predictions, top_n_jobs=5)
        self.assertEqual(result, {})

    def test_non_numeric_match_score(self):
        """Test violation when Match_Score is non-numeric."""
        predictions = pd.DataFrame({
            'Title': ['Job1', 'Job2'],
            'Career Category': ['Arts', 'Engineering'],
            'Match_Score': ['invalid', 0.8],
        })
        result = collect_prediction_contract_violations(predictions, top_n_jobs=2)
        self.assertIn('non_numeric_match_score', result)


class TestMergeViolationCounts(unittest.TestCase):
    """Test violation count merging."""

    def test_merge_empty_aggregate(self):
        """Test merge into empty aggregate."""
        aggregate = {}
        sample_violations = {'missing_columns': 1}
        merge_violation_counts(aggregate, sample_violations)
        self.assertEqual(aggregate, {'missing_columns': 1})

    def test_merge_new_violation_type(self):
        """Test merge with new violation type."""
        aggregate = {'missing_columns': 1}
        sample_violations = {'null_values': 2}
        merge_violation_counts(aggregate, sample_violations)
        self.assertEqual(aggregate, {'missing_columns': 1, 'null_values': 2})

    def test_merge_existing_violation_type(self):
        """Test merge increments existing violation type."""
        aggregate = {'missing_columns': 1}
        sample_violations = {'missing_columns': 2}
        merge_violation_counts(aggregate, sample_violations)
        self.assertEqual(aggregate, {'missing_columns': 3})

    def test_merge_multiple_violations(self):
        """Test merge multiple violation types."""
        aggregate = {'missing_columns': 1, 'null_values': 1}
        sample_violations = {'missing_columns': 1, 'null_values': 2, 'non_numeric_scores': 1}
        merge_violation_counts(aggregate, sample_violations)
        self.assertEqual(aggregate, {
            'missing_columns': 2,
            'null_values': 3,
            'non_numeric_scores': 1,
        })


class TestBuildModelResult(unittest.TestCase):
    """Test model result aggregation."""

    def setUp(self):
        """Set up test data."""
        self.mock_model = Mock()
        self.mock_model.get_name.return_value = 'test_model'
        self.mock_model.top_n_jobs = 5

    def test_build_result_basic(self):
        """Test building basic model result."""
        result = build_model_result(
            all_metrics=[{'ndcg@5': 0.8}],
            latencies=[10.0],
            memory_usages=[1024],
            model=self.mock_model,
            constraint_violations={},
            samples_evaluated=1,
        )

        self.assertIn('metrics', result)
        self.assertIn('latency_ms', result)
        self.assertIn('memory_bytes', result)
        self.assertIn('model_size_mb', result)
        self.assertIn('constraint_violations', result)
        self.assertIn('samples_evaluated', result)
        self.assertEqual(result['samples_evaluated'], 1)

    def test_build_result_aggregates_metrics(self):
        """Test building result aggregates metrics."""
        result = build_model_result(
            all_metrics=[{'ndcg@5': 0.8}, {'ndcg@5': 0.9}],
            latencies=[10.0, 12.0],
            memory_usages=[1024, 2048],
            model=self.mock_model,
            constraint_violations={},
            samples_evaluated=2,
        )
        self.assertAlmostEqual(result['metrics']['ndcg@5'], 0.85, places=6)  # Average
        self.assertEqual(result['samples_evaluated'], 2)

    def test_build_result_includes_violations(self):
        """Test building result includes constraint violations."""
        violations = {'missing_columns': 1, 'null_values': 2}
        result = build_model_result(
            all_metrics=[{'ndcg@5': 0.8}],
            latencies=[10.0],
            memory_usages=[1024],
            model=self.mock_model,
            constraint_violations=violations,
            samples_evaluated=1,
        )

        self.assertEqual(result['constraint_violations'], violations)

    def test_build_result_no_samples(self):
        """Test building result with no samples evaluated."""
        with self.assertRaises(ValueError):
            build_model_result(
                all_metrics=[],
                latencies=[],
                memory_usages=[],
                model=self.mock_model,
                constraint_violations={},
                samples_evaluated=0,
            )

    @patch('src.evaluation.result_builders.get_model_size_mb')
    def test_build_result_includes_model_size(self, mock_size):
        """Test building result includes model size."""
        mock_size.return_value = 5.2
        result = build_model_result(
            all_metrics=[{'ndcg@5': 0.8}],
            latencies=[10.0],
            memory_usages=[1024],
            model=self.mock_model,
            constraint_violations={},
            samples_evaluated=1,
        )

        self.assertEqual(result['model_size_mb'], 5.2)


class TestSanitizePathComponent(unittest.TestCase):
    """Test path component sanitization."""

    def test_alphanumeric_unchanged(self):
        """Test alphanumeric strings are unchanged."""
        result = sanitize_path_component('model_123')
        self.assertEqual(result, 'model_123')

    def test_spaces_replaced_with_underscore(self):
        """Test spaces are replaced with underscore."""
        result = sanitize_path_component('test model')
        self.assertEqual(result, 'test_model')

    def test_special_characters_removed(self):
        """Test special characters are replaced with underscores."""
        result = sanitize_path_component('test@model#2')
        self.assertEqual(result, 'test_model_2')

    def test_slashes_removed(self):
        """Test forward/backward slashes are replaced with underscores."""
        result = sanitize_path_component('path/to/model')
        self.assertEqual(result, 'path_to_model')

    def test_empty_string_returns_default(self):
        """Test empty string returns default."""
        result = sanitize_path_component('')
        self.assertEqual(result, 'unknown')

    def test_only_special_chars_returns_default(self):
        """Test string with only special chars returns default."""
        result = sanitize_path_component('!@#$%')
        self.assertEqual(result, 'unknown')


class TestWriteAggregateResults(unittest.TestCase):
    """Test aggregate result writing."""

    def test_write_aggregate_results(self):
        """Test writing aggregate results to JSON."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / 'results.json'
            results = [
                {'model': 'model1', 'metrics': {'ndcg@5': 0.8}},
                {'model': 'model2', 'metrics': {'ndcg@5': 0.9}},
            ]
            write_aggregate_results(results, output_path)

            # Verify file exists
            self.assertTrue(output_path.exists())

            # Verify content
            with open(output_path, 'r') as f:
                loaded = json.load(f)
            self.assertEqual(loaded, results)

    def test_write_aggregate_creates_directory(self):
        """Test write creates output directory if missing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / 'subdir' / 'results.json'
            results = [{'model': 'model1', 'metrics': {}}]
            write_aggregate_results(results, output_path)

            self.assertTrue(output_path.exists())
            self.assertTrue(output_path.parent.exists())

    def test_write_empty_results(self):
        """Test writing empty results."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / 'results.json'
            results = []
            write_aggregate_results(results, output_path)

            with open(output_path, 'r') as f:
                loaded = json.load(f)
            self.assertEqual(loaded, [])


class TestWriteSingleExperimentResult(unittest.TestCase):
    """Test single experiment result writing."""

    def test_write_single_experiment_result(self):
        """Test writing single experiment result."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / 'results'
            result = {'model': 'model1', 'dataset': 'data1', 'metrics': {'ndcg@5': 0.8}}
            run_id = 'run_001'
            job_index = 0

            write_single_experiment_result(result, output_path, run_id, job_index)

            # Verify file in per_experiment dir
            per_experiment_dir = output_path.parent / 'per_experiment'
            file_name = f"{job_index:03d}_data1_model1.json"
            per_experiment_path = per_experiment_dir / file_name
            self.assertTrue(per_experiment_dir.exists())
            self.assertTrue(per_experiment_path.exists())
            with open(per_experiment_path, 'r') as f:
                loaded = json.load(f)
            self.assertEqual(loaded, result)

    def test_write_single_experiment_creates_directories(self):
        """Test write creates nested directories."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / 'nested' / 'results'
            result = {'model': 'model1', 'metrics': {}}
            run_id = 'run_001'
            job_index = 0

            write_single_experiment_result(result, output_path, run_id, job_index)

            per_experiment_dir = output_path.parent / 'per_experiment'
            file_name = f"{job_index:03d}_dataset_model1.json"
            per_experiment_path = per_experiment_dir / file_name
            self.assertTrue(per_experiment_dir.exists())
            self.assertTrue(per_experiment_path.exists())

    def test_write_multiple_experiment_results(self):
        """Test writing multiple experiment results with different indices."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / 'results'
            run_id = 'run_001'

            for job_idx in range(3):
                result = {'model': f'model{job_idx}', 'metrics': {}}
                write_single_experiment_result(result, output_path, run_id, job_idx)

            per_experiment_dir = output_path.parent / 'per_experiment'
            for job_idx in range(3):
                file_name = f"{job_idx:03d}_dataset_model{job_idx}.json"
                per_experiment_path = per_experiment_dir / file_name
                self.assertTrue(per_experiment_path.exists())


if __name__ == '__main__':
    unittest.main()
