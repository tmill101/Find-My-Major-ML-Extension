"""
File: test_evaluator.py
Path: tests/evaluation/test_evaluator.py

Purpose:
  Unit tests for evaluation orchestration functions.

Original Author(s):
  - AI Assistant (GitHub Copilot)

AI Tools Used:
  - GitHub Copilot - Code generation and documentation

Editors:
  - AI Assistant (2026-03-23) — Initial implementation
  - AI Assistant (2026-03-30) — Added validation and contract-check coverage
  - OpenAI Codex (2026-04-06) — Added multiprocessing execution coverage

Last Editor:
  - OpenAI Codex

Last Edit Date:
  2026-04-06

Assumptions & Constraints:
  - Uses mock models and datasets
  - Tests evaluation loop behavior
  - No real model training/inference

Related Docs:
  - docs/src/evaluation/evaluation.md
"""

import unittest
import tempfile
import json
import shutil
from pathlib import Path
from unittest.mock import Mock, patch
import pandas as pd
from src.evaluation.evaluator import (
    _build_experiment_jobs,
    Dataset,
    EvaluationInterrupted,
    evaluate_experiment,
    evaluate_model,
    run_single_experiment,
)
from src.data.schemas import TrainingRecord


class TestDataset(unittest.TestCase):
    """Test Dataset class functionality."""

    def setUp(self):
        """Set up test data."""
        self.train_records = [
            TrainingRecord(realistic=0.8, investigative=0.6, artistic=0.4,
                          social=0.7, enterprising=0.5, conventional=0.3,
                          career_category='Engineering, Manufacturing & Construction'),
            TrainingRecord(realistic=0.5, investigative=0.8, artistic=0.6,
                          social=0.4, enterprising=0.7, conventional=0.3,
                          career_category='Science & Mathematics')
        ]
        self.test_records = [
            TrainingRecord(realistic=0.7, investigative=0.5, artistic=0.6,
                          social=0.8, enterprising=0.4, conventional=0.3,
                          career_category='Arts, Communications & Humanities')
        ]
        self.feature_columns = ['realistic', 'investigative', 'artistic', 'social', 'enterprising', 'conventional']

    def test_dataset_creation(self):
        """Test Dataset object creation."""
        dataset = Dataset(self.train_records, self.test_records, self.feature_columns)

        self.assertEqual(len(dataset.train_records), 2)
        self.assertEqual(len(dataset.test_records), 1)
        self.assertEqual(dataset.feature_columns, self.feature_columns)

    def test_dataset_split(self):
        """Test Dataset.split() method."""
        dataset = Dataset(self.train_records, self.test_records, self.feature_columns)
        train_df, test_df = dataset.split()

        # Check DataFrame shapes
        self.assertEqual(train_df.shape, (2, 7))  # 2 rows, 6 features + category
        self.assertEqual(test_df.shape, (1, 7))   # 1 row, 6 features + category

        # Check column names
        expected_cols = self.feature_columns + ['career_category']
        self.assertEqual(list(train_df.columns), expected_cols)
        self.assertEqual(list(test_df.columns), expected_cols)

        # Check data values
        self.assertEqual(train_df.iloc[0]['realistic'], 0.8)
        self.assertEqual(train_df.iloc[0]['career_category'], 'Engineering, Manufacturing & Construction')
        self.assertEqual(test_df.iloc[0]['career_category'], 'Arts, Communications & Humanities')

    def test_dataset_empty_train(self):
        """Test error handling for empty train records."""
        with self.assertRaises(ValueError):
            Dataset([], self.test_records, self.feature_columns)

    def test_dataset_empty_test(self):
        """Test error handling for empty test records."""
        with self.assertRaises(ValueError):
            Dataset(self.train_records, [], self.feature_columns)

    def test_dataset_empty_features(self):
        """Test error handling for empty feature columns."""
        with self.assertRaises(ValueError):
            Dataset(self.train_records, self.test_records, [])


class TestEvaluateModel(unittest.TestCase):
    """Test evaluate_model function."""

    def setUp(self):
        """Set up mock objects."""
        # Create mock dataset
        train_records = [
            TrainingRecord(realistic=0.8, investigative=0.6, artistic=0.4,
                          social=0.7, enterprising=0.5, conventional=0.3,
                          career_category='Engineering, Manufacturing & Construction')
        ]
        test_records = [
            TrainingRecord(realistic=0.7, investigative=0.5, artistic=0.6,
                          social=0.8, enterprising=0.4, conventional=0.3,
                          career_category='Arts, Communications & Humanities')
        ]
        self.dataset = Dataset(train_records, test_records,
                              ['realistic', 'investigative', 'artistic', 'social', 'enterprising', 'conventional'])

        # Create mock model
        self.mock_model = Mock()
        self.mock_model.get_name.return_value = 'test_model'
        self.mock_model.y_feature = 'career_category'
        self.mock_model.test.return_value = pd.DataFrame({
            'Title': ['Job1', 'Job2'],
            'Career Category': ['Arts', 'Engineering'],
            'Match_Score': [0.9, 0.8]
        })

        # Mock O*NET database
        self.onet_db = pd.DataFrame({
            'Title': ['Job1', 'Job2'],
            'Career Category': ['Arts', 'Engineering']
        })

        self.config = {'evaluation': {'k_values': [5]}}

    @patch('src.evaluation.evaluator.benchmark_model_inference')
    @patch('src.evaluation.evaluator.compute_all_metrics')
    def test_evaluate_model_basic(self, mock_compute_metrics, mock_benchmark):
        """Test basic evaluate_model functionality."""
        mock_benchmark.return_value = {
            'latency_ms': 10.0,
            'memory_bytes': 1024,
            'predictions': self.mock_model.test.return_value
        }
        mock_compute_metrics.return_value = {'ndcg@5': 0.8, 'precision@5': 0.6}

        result = evaluate_model(self.mock_model, self.dataset, self.onet_db, self.config)

        # Check result structure
        self.assertIn('metrics', result)
        self.assertIn('latency_ms', result)
        self.assertIn('memory_bytes', result)
        self.assertIn('model_size_mb', result)
        self.assertIn('constraint_violations', result)
        self.assertIn('samples_evaluated', result)

        # Check values
        self.assertEqual(result['metrics'], {'ndcg@5': 0.8, 'precision@5': 0.6})
        self.assertEqual(result['latency_ms'], 10.0)
        self.assertEqual(result['memory_bytes'], 1024)
        self.assertEqual(result['samples_evaluated'], 1)

    @patch('src.evaluation.evaluator.benchmark_model_inference')
    @patch('src.evaluation.evaluator.compute_all_metrics')
    def test_evaluate_model_respects_metric_selection(self, mock_compute_metrics, mock_benchmark):
        """Test evaluate_model filters metrics based on config.evaluation.metrics."""
        mock_benchmark.return_value = {
            'latency_ms': 10.0,
            'memory_bytes': 1024,
            'predictions': self.mock_model.test.return_value
        }
        mock_compute_metrics.return_value = {'ndcg@5': 0.8, 'precision@5': 0.6}

        config = {'evaluation': {'k_values': [5], 'metrics': ['ndcg_at_k']}}
        result = evaluate_model(self.mock_model, self.dataset, self.onet_db, config)

        self.assertEqual(result['metrics'], {'ndcg@5': 0.8})

    def test_evaluate_model_invalid_metric_raises(self):
        """Test evaluate_model fails on unsupported metric names."""
        bad_config = {'evaluation': {'k_values': [5], 'metrics': ['accuracy']}}

        with self.assertRaises(ValueError):
            evaluate_model(self.mock_model, self.dataset, self.onet_db, bad_config)

    def test_evaluate_model_k_values_exceed_top_k_raises(self):
        """Test evaluate_model fails when evaluation.k_values exceeds evaluation.top_k."""
        bad_config = {'evaluation': {'k_values': [10], 'top_k': 5}}

        with self.assertRaises(ValueError):
            evaluate_model(self.mock_model, self.dataset, self.onet_db, bad_config)

    def test_evaluate_model_invalid_progress_log_interval_raises(self):
        """Test evaluate_model validates evaluation.progress_log_interval."""
        bad_config = {'evaluation': {'k_values': [5], 'progress_log_interval': 0}}

        with self.assertRaises(ValueError):
            evaluate_model(self.mock_model, self.dataset, self.onet_db, bad_config)

    def test_evaluate_model_raises_when_no_samples_evaluated(self):
        """Test evaluate_model fails fast when max_test_samples excludes all rows."""
        config = {'evaluation': {'k_values': [5], 'max_test_samples': 0}}

        with patch('src.evaluation.evaluator.benchmark_model_inference') as mock_benchmark, \
             patch('src.evaluation.evaluator.compute_all_metrics') as mock_metrics:
            mock_benchmark.return_value = {
                'latency_ms': 10.0,
                'memory_bytes': 1024,
                'predictions': self.mock_model.test.return_value
            }
            mock_metrics.return_value = {'ndcg@5': 0.8, 'precision@5': 0.6}

            with self.assertRaises(ValueError):
                evaluate_model(self.mock_model, self.dataset, self.onet_db, config)

    def test_evaluate_model_max_test_samples_uses_sample_count_not_row_index(self):
        """Test max_test_samples is based on sample count, not DataFrame row index values."""
        config = {'evaluation': {'k_values': [5], 'max_test_samples': 1}}

        train_df, test_df = self.dataset.split()
        test_df.index = [99]

        with patch.object(self.dataset, 'split', return_value=(train_df, test_df)), \
             patch('src.evaluation.evaluator.benchmark_model_inference') as mock_benchmark, \
             patch('src.evaluation.evaluator.compute_all_metrics') as mock_metrics:
            mock_benchmark.return_value = {
                'latency_ms': 10.0,
                'memory_bytes': 1024,
                'predictions': self.mock_model.test.return_value
            }
            mock_metrics.return_value = {'ndcg@5': 0.8, 'precision@5': 0.6}

            result = evaluate_model(self.mock_model, self.dataset, self.onet_db, config)

            self.assertEqual(mock_benchmark.call_count, 1)
            self.assertEqual(result['samples_evaluated'], 1)

    def test_evaluate_model_calls_benchmark(self):
        """Test that evaluate_model calls benchmark_model_inference for each test sample."""
        with patch('src.evaluation.evaluator.benchmark_model_inference') as mock_benchmark, \
             patch('src.evaluation.evaluator.compute_all_metrics'):

            mock_benchmark.return_value = {
                'latency_ms': 10.0,
                'memory_bytes': 1024,
                'predictions': self.mock_model.test.return_value
            }
            evaluate_model(self.mock_model, self.dataset, self.onet_db, self.config)

            # Should call benchmark once for one test sample
            self.assertEqual(mock_benchmark.call_count, 1)

    def test_evaluate_model_interrupt_returns_partial_result(self):
        """Test evaluate_model raises EvaluationInterrupted with partial metrics."""
        train_records = [
            TrainingRecord(realistic=0.8, investigative=0.6, artistic=0.4,
                          social=0.7, enterprising=0.5, conventional=0.3,
                          career_category='Engineering, Manufacturing & Construction')
        ]
        test_records = [
            TrainingRecord(realistic=0.7, investigative=0.5, artistic=0.6,
                          social=0.8, enterprising=0.4, conventional=0.3,
                          career_category='Arts, Communications & Humanities'),
            TrainingRecord(realistic=0.6, investigative=0.4, artistic=0.5,
                          social=0.7, enterprising=0.3, conventional=0.2,
                          career_category='Arts, Communications & Humanities'),
        ]
        dataset = Dataset(
            train_records,
            test_records,
            ['realistic', 'investigative', 'artistic', 'social', 'enterprising', 'conventional']
        )

        with patch('src.evaluation.evaluator.benchmark_model_inference') as mock_benchmark, \
             patch('src.evaluation.evaluator.compute_all_metrics') as mock_metrics:
            mock_benchmark.side_effect = [
                {
                    'latency_ms': 10.0,
                    'memory_bytes': 1024,
                    'predictions': self.mock_model.test.return_value,
                },
                KeyboardInterrupt(),
            ]
            mock_metrics.return_value = {'ndcg@5': 0.8, 'precision@5': 0.6}

            with self.assertRaises(EvaluationInterrupted) as ctx:
                evaluate_model(self.mock_model, dataset, self.onet_db, {'evaluation': {'k_values': [5]}})

            partial = ctx.exception.partial_result
            self.assertEqual(partial['samples_evaluated'], 1)
            self.assertEqual(partial['metrics']['ndcg@5'], 0.8)
            self.assertEqual(partial['metrics']['precision@5'], 0.6)

    def test_evaluate_model_no_test_method(self):
        """Test error handling when model has no test method."""
        bad_model = Mock(spec=['get_name'])
        bad_model.get_name.return_value = 'bad_model'

        with self.assertRaises(AttributeError):
            evaluate_model(bad_model, self.dataset, self.onet_db, self.config)


class TestExperimentParallelExecution(unittest.TestCase):
    """Tests multiprocessing experiment orchestration."""

    def setUp(self):
        self.temp_dir = Path(tempfile.mkdtemp())
        self.feature_columns = [
            'Realistic',
            'Investigative',
            'Artistic',
            'Social',
            'Enterprising',
            'Conventional',
        ]

        self.train_path = self.temp_dir / 'train.csv'
        self.test_path = self.temp_dir / 'test.csv'
        self.onet_path = self.temp_dir / 'onet.csv'

        self._write_training_csv(
            self.train_path,
            [
                self._training_row(
                    category='Engineering, Manufacturing & Construction',
                    realistic=0.9,
                    investigative=0.7,
                    artistic=0.2,
                    social=0.3,
                    enterprising=0.6,
                    conventional=0.8,
                ),
                self._training_row(
                    category='Science & Mathematics',
                    realistic=0.5,
                    investigative=0.9,
                    artistic=0.4,
                    social=0.5,
                    enterprising=0.3,
                    conventional=0.4,
                ),
            ],
        )

        self._write_training_csv(
            self.test_path,
            [
                self._training_row(
                    category='Science & Mathematics',
                    realistic=0.6,
                    investigative=0.8,
                    artistic=0.5,
                    social=0.4,
                    enterprising=0.3,
                    conventional=0.4,
                )
            ],
        )

        pd.DataFrame(
            [
                {
                    'O*NET-SOC Code': '11-9041.00',
                    'Title': 'Engineering Manager',
                    'Realistic': 0.9,
                    'Investigative': 0.6,
                    'Artistic': 0.2,
                    'Social': 0.4,
                    'Enterprising': 0.8,
                    'Conventional': 0.7,
                    'Career Category': 'Engineering, Manufacturing & Construction',
                },
                {
                    'O*NET-SOC Code': '19-2031.00',
                    'Title': 'Chemist',
                    'Realistic': 0.5,
                    'Investigative': 0.9,
                    'Artistic': 0.3,
                    'Social': 0.4,
                    'Enterprising': 0.2,
                    'Conventional': 0.5,
                    'Career Category': 'Science & Mathematics',
                },
            ]
        ).to_csv(self.onet_path, index=False)

        dataset_a = {
            'name': 'dataset_a',
            'train_path': str(self.train_path),
            'test_path': str(self.test_path),
            'split': None,
            'shuffle': False,
        }
        dataset_b = {
            'name': 'dataset_b',
            'train_path': str(self.train_path),
            'test_path': str(self.test_path),
            'split': None,
            'shuffle': False,
        }
        model_cfg = {
            'model': 'heuristic',
            'parameters': {'top_n_categories': 2},
            'x_features': self.feature_columns,
            'y_features': ['Career Category'],
        }

        self.base_config = {
            'experiment': {'id': 'test_exp', 'random_seed': 42},
            'run': {'run_id': 'test_exp_20260406_000000'},
            'onet_db_path': str(self.onet_path),
            'datasets': [dataset_a, dataset_b],
            'models': [model_cfg],
            'training': {'parallel_jobs': 2},
            'evaluation': {
                'metrics': ['ndcg_at_k', 'precision_at_k'],
                'k_values': [1],
                'top_k': 1,
                'benchmark_runs': 1,
                'max_test_samples': 1,
                'progress_log_interval': 1000,
            },
        }

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    @staticmethod
    def _training_row(
        category: str,
        realistic: float,
        investigative: float,
        artistic: float,
        social: float,
        enterprising: float,
        conventional: float,
    ) -> dict:
        return {
            'R normalized': realistic,
            'I normalized': investigative,
            'A normalized': artistic,
            'S normalized': social,
            'E normalized': enterprising,
            'C normalized': conventional,
            'Career Category': category,
        }

    @staticmethod
    def _write_training_csv(path: Path, rows: list[dict]) -> None:
        pd.DataFrame(rows).to_csv(path, index=False)

    def test_jobs_are_generated_from_dataset_model_matrix(self):
        config = {
            'datasets': [{'name': 'd1'}, {'name': 'd2'}],
            'models': [{'model': 'heuristic'}, {'model': 'knn'}],
        }

        jobs = _build_experiment_jobs(config)

        self.assertEqual(len(jobs), 4)
        self.assertEqual(jobs[0][0]['name'], 'd1')
        self.assertEqual(jobs[0][1]['model'], 'heuristic')
        self.assertEqual(jobs[3][0]['name'], 'd2')
        self.assertEqual(jobs[3][1]['model'], 'knn')

    def test_worker_returns_expected_result_structure(self):
        result = run_single_experiment(
            self.base_config['datasets'][0],
            self.base_config['models'][0],
            self.base_config,
        )

        self.assertIn('metrics', result)
        self.assertIn('latency_ms', result)
        self.assertIn('memory_bytes', result)
        self.assertIn('model_size_mb', result)
        self.assertIn('constraint_violations', result)
        self.assertIn('samples_evaluated', result)
        self.assertEqual(result['dataset'], 'dataset_a')
        self.assertEqual(result['model'], 'heuristic')

    def test_parallel_execution_completes_successfully(self):
        output_path = self.temp_dir / 'evaluation.json'
        results = evaluate_experiment(
            datasets=[],
            models=[],
            onet_db=pd.DataFrame(),
            config=self.base_config,
            output_path=output_path,
        )

        self.assertEqual(len(results), 2)
        self.assertTrue(output_path.exists())
        with open(output_path, 'r', encoding='utf-8') as f:
            saved = json.load(f)
        self.assertEqual(len(saved), 2)

        per_experiment_files = list((self.temp_dir / 'per_experiment').glob('*.json'))
        self.assertEqual(len(per_experiment_files), 2)

    def test_output_path_parent_is_created_if_missing(self):
        output_path = self.temp_dir / 'nested' / 'evaluation.json'
        results = evaluate_experiment(
            datasets=[],
            models=[],
            onet_db=pd.DataFrame(),
            config=self.base_config,
            output_path=output_path,
        )

        self.assertEqual(len(results), 2)
        self.assertTrue(output_path.exists())
        self.assertTrue((self.temp_dir / 'nested' / 'per_experiment').exists())

    def test_failed_job_does_not_stop_other_jobs(self):
        bad_dataset = {
            'name': 'broken_dataset',
            'train_path': str(self.temp_dir / 'missing_train.csv'),
            'test_path': str(self.temp_dir / 'missing_test.csv'),
            'split': None,
            'shuffle': False,
        }
        config = dict(self.base_config)
        config['datasets'] = [self.base_config['datasets'][0], bad_dataset]

        results = evaluate_experiment(
            datasets=[],
            models=[],
            onet_db=pd.DataFrame(),
            config=config,
            output_path=self.temp_dir / 'evaluation_failed.json',
        )

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]['dataset'], 'dataset_a')
        self.assertEqual(results[0]['model'], 'heuristic')


if __name__ == '__main__':
    unittest.main()
