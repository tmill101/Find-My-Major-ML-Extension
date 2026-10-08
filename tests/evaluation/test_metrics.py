"""
File: test_metrics.py
Path: tests/evaluation/test_metrics.py

Purpose:
  Unit tests for evaluation metrics functions.

Original Author(s):
  - AI Assistant (GitHub Copilot)

AI Tools Used:
  - GitHub Copilot - Code generation and documentation

Editors:
  - AI Assistant (2026-03-23) — Initial implementation

Last Editor:
  - AI Assistant

Last Edit Date:
  2026-03-23

Assumptions & Constraints:
  - Tests are deterministic
  - No external dependencies
  - RIASEC vectors are length 6

Related Docs:
  - docs/src/evaluation/evaluation.md
"""

import unittest
import pandas as pd
import numpy as np
from src.evaluation.metrics import cosine_similarity, ndcg_at_k, precision_at_k, compute_all_metrics


class TestCosineSimilarity(unittest.TestCase):
    """Test cosine similarity calculations."""

    def test_identical_vectors(self):
        """Test cosine similarity of identical vectors."""
        v = [0.5, 0.3, 0.2, 0.1, 0.4, 0.6]
        self.assertAlmostEqual(cosine_similarity(v, v), 1.0, places=6)

    def test_opposite_vectors(self):
        """Test cosine similarity of opposite vectors."""
        v1 = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        v2 = [-1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self.assertAlmostEqual(cosine_similarity(v1, v2), -1.0, places=6)

    def test_orthogonal_vectors(self):
        """Test cosine similarity of orthogonal vectors."""
        v1 = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        v2 = [0.0, 1.0, 0.0, 0.0, 0.0, 0.0]
        self.assertAlmostEqual(cosine_similarity(v1, v2), 0.0, places=6)

    def test_zero_vector(self):
        """Test cosine similarity with zero vector."""
        v1 = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        v2 = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self.assertEqual(cosine_similarity(v1, v2), 0.0)

    def test_wrong_length(self):
        """Test error handling for wrong vector length."""
        v1 = [1.0, 0.0, 0.0, 0.0, 0.0]  # length 5
        v2 = [0.0, 1.0, 0.0, 0.0, 0.0, 0.0]  # length 6
        with self.assertRaises(ValueError):
            cosine_similarity(v1, v2)


class TestNDCG(unittest.TestCase):
    """Test NDCG@K calculations."""

    def setUp(self):
        """Set up test data."""
        self.predictions = pd.DataFrame({
            'Title': ['Job1', 'Job2', 'Job3', 'Job4', 'Job5'],
            'Career Category': ['Engineering', 'Science', 'Engineering', 'Arts', 'Science'],
            'Match_Score': [0.9, 0.8, 0.7, 0.6, 0.5]
        })

    def test_perfect_ranking(self):
        """Test NDCG when all relevant items are at top."""
        # For Engineering: relevant items are Job1 (pos 1) and Job3 (pos 3)
        # DCG = 1/log2(2) + 1/log2(4) = 1 + 0.5 = 1.5
        # IDCG = 1/log2(2) + 1/log2(3) = 1 + 0.6309 = 1.6309
        # NDCG = 1.5/1.6309 ≈ 0.9197
        ground_truth = 'Engineering'
        score = ndcg_at_k(self.predictions, ground_truth, k=3)
        expected = (1/np.log2(2) + 1/np.log2(4)) / (1/np.log2(2) + 1/np.log2(3))
        self.assertAlmostEqual(score, expected, places=4)

    def test_no_relevant_items(self):
        """Test NDCG when no relevant items in top K."""
        ground_truth = 'Business'
        score = ndcg_at_k(self.predictions, ground_truth, k=3)
        self.assertEqual(score, 0.0)

    def test_partial_relevance(self):
        """Test NDCG with partial relevance."""
        # For Science: relevant items are Job2 (pos 2) and Job5 (pos 5)
        # DCG = 1/log2(3) + 1/log2(6) = 0.6309 + 0.3869 = 1.0178
        # IDCG = 1/log2(2) + 1/log2(3) = 1 + 0.6309 = 1.6309
        # NDCG = 1.0178/1.6309 ≈ 0.624
        ground_truth = 'Science'
        score = ndcg_at_k(self.predictions, ground_truth, k=5)
        expected = (1/np.log2(3) + 1/np.log2(6)) / (1/np.log2(2) + 1/np.log2(3))
        self.assertAlmostEqual(score, expected, places=3)

    def test_missing_columns(self):
        """Test error handling for missing DataFrame columns."""
        bad_df = pd.DataFrame({'Title': ['Job1'], 'Score': [0.9]})
        with self.assertRaises(ValueError):
            ndcg_at_k(bad_df, 'Engineering', k=1)


class TestPrecision(unittest.TestCase):
    """Test Precision@K calculations."""

    def setUp(self):
        """Set up test data."""
        self.predictions = pd.DataFrame({
            'Title': ['Job1', 'Job2', 'Job3', 'Job4', 'Job5'],
            'Career Category': ['Engineering', 'Science', 'Engineering', 'Arts', 'Science'],
            'Match_Score': [0.9, 0.8, 0.7, 0.6, 0.5]
        })

    def test_all_relevant(self):
        """Test precision when all top K are relevant."""
        # For Engineering with k=2: top 2 are Job1 (Engineering) and Job2 (Science)
        # Only 1 is relevant, so precision = 1/2 = 0.5
        ground_truth = 'Engineering'
        score = precision_at_k(self.predictions, ground_truth, k=2)
        self.assertEqual(score, 0.5)

    def test_no_relevant(self):
        """Test precision when no top K are relevant."""
        ground_truth = 'Business'
        score = precision_at_k(self.predictions, ground_truth, k=2)
        self.assertEqual(score, 0.0)

    def test_partial_relevance(self):
        """Test precision with partial relevance."""
        # For Science with k=3: top 3 are Job1 (Engineering), Job2 (Science), Job3 (Engineering)
        # Only 1 is relevant, so precision = 1/3
        ground_truth = 'Science'
        score = precision_at_k(self.predictions, ground_truth, k=3)
        self.assertAlmostEqual(score, 1.0/3.0, places=6)

    def test_missing_columns(self):
        """Test error handling for missing DataFrame columns."""
        bad_df = pd.DataFrame({'Title': ['Job1'], 'Score': [0.9]})
        with self.assertRaises(ValueError):
            precision_at_k(bad_df, 'Engineering', k=1)


class TestComputeAllMetrics(unittest.TestCase):
    """Test compute_all_metrics function."""

    def setUp(self):
        """Set up test data."""
        self.predictions = pd.DataFrame({
            'Title': ['Job1', 'Job2', 'Job3', 'Job4', 'Job5'],
            'Career Category': ['Engineering', 'Science', 'Engineering', 'Arts', 'Science'],
            'Match_Score': [0.9, 0.8, 0.7, 0.6, 0.5]
        })
        self.ground_truth = 'Engineering'
        self.k_values = [3, 5]

    def test_returns_correct_keys(self):
        """Test that function returns metrics for all K values."""
        results = compute_all_metrics(self.predictions, self.ground_truth, self.k_values)
        expected_keys = ['ndcg@3', 'precision@3', 'recall@3', 'ndcg@5', 'precision@5', 'recall@5']
        self.assertEqual(set(results.keys()), set(expected_keys))

    def test_values_are_floats(self):
        """Test that all returned values are floats."""
        results = compute_all_metrics(self.predictions, self.ground_truth, self.k_values)
        for value in results.values():
            self.assertIsInstance(value, float)

    def test_values_in_range(self):
        """Test that all metric values are in [0, 1]."""
        results = compute_all_metrics(self.predictions, self.ground_truth, self.k_values)
        for value in results.values():
            self.assertGreaterEqual(value, 0.0)
            self.assertLessEqual(value, 1.0)


if __name__ == '__main__':
    unittest.main()