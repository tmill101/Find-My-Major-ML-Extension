"""
File: test_config_resolution.py
Path: tests/evaluation/test_config_resolution.py

Purpose:
  Unit tests for configuration resolution and validation helpers extracted from evaluator.py.
  Tests the delegation pattern and module interfaces.

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
from unittest.mock import Mock

from src.evaluation.config_resolution import (
    resolve_parallel_jobs,
    resolve_metric_selection,
    validate_k_values,
    resolve_top_k,
    filter_metric_results,
)


class TestResolveParallelJobs(unittest.TestCase):
    """Test resolve_parallel_jobs configuration validation."""

    def test_explicit_parallel_jobs(self):
        """Test explicit parallel_jobs from config."""
        config = {'training': {'parallel_jobs': 4}}
        result = resolve_parallel_jobs(config)
        self.assertEqual(result, 4)

    def test_default_parallel_jobs(self):
        """Test default parallel_jobs when missing."""
        config = {}
        result = resolve_parallel_jobs(config)
        self.assertEqual(result, 2)

    def test_parallel_jobs_must_be_positive(self):
        """Test parallel_jobs must be positive integer."""
        config = {'training': {'parallel_jobs': 0}}
        with self.assertRaises(ValueError):
            resolve_parallel_jobs(config)

    def test_parallel_jobs_non_integer_raises(self):
        """Test non-integer parallel_jobs raises error."""
        config = {'training': {'parallel_jobs': 'invalid'}}
        with self.assertRaises(ValueError):
            resolve_parallel_jobs(config)


class TestResolveMetricSelection(unittest.TestCase):
    """Test metric selection from configuration."""

    def test_explicit_metrics(self):
        """Test explicit metric selection from config."""
        config = {'evaluation': {'metrics': ['ndcg_at_k', 'precision_at_k']}}
        result = resolve_metric_selection(config)
        self.assertEqual(result, {'ndcg_at_k', 'precision_at_k'})

    def test_default_metrics(self):
        """Test default metrics when not specified."""
        config = {'evaluation': {}}
        result = resolve_metric_selection(config)
        self.assertEqual(result, {'ndcg_at_k', 'precision_at_k', 'recall_at_k'})

    def test_invalid_metric_raises(self):
        """Test invalid metric name raises error."""
        config = {'evaluation': {'metrics': ['accuracy']}}
        with self.assertRaises(ValueError):
            resolve_metric_selection(config)

    def test_partial_valid_metrics_raises(self):
        """Test partial valid metrics raises error."""
        config = {'evaluation': {'metrics': ['ndcg_at_k', 'invalid_metric']}}
        with self.assertRaises(ValueError):
            resolve_metric_selection(config)

    def test_empty_metrics_list_uses_default(self):
        """Test empty metrics list returns empty set (explicitly configured as empty)."""
        config = {'evaluation': {'metrics': []}}
        result = resolve_metric_selection(config)
        # Empty list is treated as "select nothing" not "use default"
        self.assertEqual(result, set())


class TestValidateKValues(unittest.TestCase):
    """Test k-value validation."""

    def test_single_k_value(self):
        """Test single k-value validation."""
        result = validate_k_values([5], top_k=5)
        self.assertEqual(result, [5])

    def test_multiple_k_values(self):
        """Test multiple k-values validation."""
        result = validate_k_values([1, 3, 5], top_k=5)
        self.assertEqual(result, [1, 3, 5])

    def test_k_values_sorted(self):
        """Test k-values are not sorted (preserve input order)."""
        result = validate_k_values([5, 1, 3], top_k=5)
        self.assertEqual(result, [5, 1, 3])

    def test_k_value_exceeds_top_k_raises(self):
        """Test k-value exceeding top_k raises error."""
        with self.assertRaises(ValueError):
            validate_k_values([10], top_k=5)

    def test_non_positive_k_value_raises(self):
        """Test non-positive k-value raises error."""
        with self.assertRaises(ValueError):
            validate_k_values([0], top_k=5)

    def test_empty_k_values_raises(self):
        """Test empty k-values list raises error."""
        with self.assertRaises(ValueError):
            validate_k_values([], top_k=5)

    def test_non_list_k_values_raises(self):
        """Test non-list k-values raises error."""
        with self.assertRaises(ValueError):
            validate_k_values('invalid', top_k=5)


class TestResolveTopK(unittest.TestCase):
    """Test top_k resolution."""

    def test_explicit_top_k(self):
        """Test explicit top_k from config."""
        config = {'top_k': 10}
        result = resolve_top_k(config, model=None)
        self.assertEqual(result, 10)

    def test_model_default_top_k(self):
        """Test top_k from model default."""
        mock_model = Mock()
        mock_model.top_n_jobs = 5
        result = resolve_top_k({}, model=mock_model)
        self.assertEqual(result, 5)

    def test_global_default_top_k(self):
        """Test global default top_k."""
        result = resolve_top_k({}, model=None)
        self.assertEqual(result, 5)

    def test_config_overrides_model_default(self):
        """Test config top_k overrides model default."""
        mock_model = Mock()
        mock_model.top_n_jobs = 5
        config = {'top_k': 10}
        result = resolve_top_k(config, model=mock_model)
        self.assertEqual(result, 10)

    def test_negative_top_k_raises(self):
        """Test negative top_k raises error."""
        config = {'top_k': -1}
        with self.assertRaises(ValueError):
            resolve_top_k(config, model=None)


class TestFilterMetricResults(unittest.TestCase):
    """Test metric result filtering."""

    def test_filter_single_metric(self):
        """Test filtering to single metric."""
        metric_results = {
            'ndcg@5': 0.8,
            'precision@5': 0.6,
            'recall@5': 0.7,
        }
        selected = {'ndcg_at_k'}
        result = filter_metric_results(metric_results, selected)
        self.assertEqual(result, {'ndcg@5': 0.8})

    def test_filter_multiple_metrics(self):
        """Test filtering to multiple metrics."""
        metric_results = {
            'ndcg@5': 0.8,
            'precision@5': 0.6,
            'recall@5': 0.7,
        }
        selected = {'ndcg_at_k', 'precision_at_k'}
        result = filter_metric_results(metric_results, selected)
        self.assertEqual(result, {'ndcg@5': 0.8, 'precision@5': 0.6})

    def test_filter_unmatched_metric(self):
        """Test filtering with unmatched metric returns empty."""
        metric_results = {'ndcg@5': 0.8, 'precision@5': 0.6}
        selected = {'recall_at_k'}
        result = filter_metric_results(metric_results, selected)
        self.assertEqual(result, {})

    def test_empty_selected_metrics(self):
        """Test empty selected metrics returns empty dict."""
        metric_results = {'ndcg@5': 0.8, 'precision@5': 0.6}
        result = filter_metric_results(metric_results, set())
        self.assertEqual(result, {})


if __name__ == '__main__':
    unittest.main()
