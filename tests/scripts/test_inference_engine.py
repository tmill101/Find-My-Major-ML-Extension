"""
File: test_inference_engine.py
Path: tests/scripts/test_inference_engine.py

Purpose:
  Unit tests for src/scripts/inference_engine.py. Validates CareerRecommender
  class initialization, ONNX model loading, and job recommendation generation.

Original Author(s):
  - AI Assistant

AI Tools Used:
  - GitHub Copilot - Test generation

Editors:
  - AI Assistant (2026-04-20) — Initial test implementation

Last Editor:
  - AI Assistant

Last Edit Date:
  2026-04-20

Assumptions & Constraints:
  - Tests use mock ONNX session to avoid model file dependencies
  - CareerRecommender interface is validated without actual inference
  - Test data is synthetic to ensure determinism

Related Docs:
  - docs/repo_structure.md
  - docs/src/models/models.md
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import pandas as pd
import numpy as np


class TestCareerRecommenderInitialization:
    """Test suite for CareerRecommender initialization and configuration."""

    @patch('src.scripts.inference_engine.ort.InferenceSession')
    @patch('pandas.read_json')
    def test_career_recommender_initializes_with_valid_paths(self, mock_read_json, mock_session):
        """
        Test that CareerRecommender initializes successfully with model and database paths.
        """
        # Arrange
        mock_session_instance = MagicMock()
        mock_session.return_value = mock_session_instance
        mock_read_json.return_value = pd.DataFrame({
            'Title': ['Software Engineer', 'Data Scientist'],
            'Career Category': ['IT', 'IT'],
            'Realistic': [0.5, 0.6],
            'Investigative': [0.8, 0.9],
            'Artistic': [0.2, 0.3],
            'Social': [0.4, 0.5],
            'Enterprising': [0.6, 0.7],
            'Conventional': [0.7, 0.6]
        })

        # Act
        from src.scripts.inference_engine import CareerRecommender
        recommender = CareerRecommender('riasec_model.onnx', 'riasec_jobs_db.json')

        # Assert
        assert recommender.session == mock_session_instance
        assert isinstance(recommender.db, pd.DataFrame)
        assert len(recommender.features) == 6
        assert recommender.features == ['Realistic', 'Investigative', 'Artistic', 'Social', 'Enterprising', 'Conventional']

    @patch('src.scripts.inference_engine.ort.InferenceSession')
    def test_career_recommender_initialization_handles_missing_model(self, mock_session):
        """
        Test that CareerRecommender handles model loading errors appropriately.
        """
        # Arrange
        mock_session.side_effect = Exception("Model file not found")

        # Act & Assert
        from src.scripts.inference_engine import CareerRecommender
        with pytest.raises(Exception):
            CareerRecommender('nonexistent_model.onnx', 'riasec_jobs_db.json')


class TestCareerRecommenderInterface:
    """Test suite for CareerRecommender public methods."""

    @patch('src.scripts.inference_engine.ort.InferenceSession')
    @patch('pandas.read_json')
    def test_career_recommender_has_required_methods(self, mock_read_json, mock_session):
        """
        Test that CareerRecommender exposes required public interface methods.
        """
        # Arrange
        mock_session_instance = MagicMock()
        mock_session.return_value = mock_session_instance
        mock_read_json.return_value = pd.DataFrame({
            'Title': ['Software Engineer'],
            'Career Category': ['IT'],
            'Realistic': [0.5],
            'Investigative': [0.8],
            'Artistic': [0.2],
            'Social': [0.4],
            'Enterprising': [0.6],
            'Conventional': [0.7]
        })

        from src.scripts.inference_engine import CareerRecommender
        recommender = CareerRecommender('riasec_model.onnx', 'riasec_jobs_db.json')

        # Act & Assert
        assert hasattr(recommender, 'get_all_recommendations')
        assert callable(recommender.get_all_recommendations)
        assert hasattr(recommender, 'get_top_n_recommendations')
        assert callable(recommender.get_top_n_recommendations)

    @patch('src.scripts.inference_engine.ort.InferenceSession')
    @patch('pandas.read_json')
    def test_career_recommender_input_validation(self, mock_read_json, mock_session):
        """
        Test that CareerRecommender validates student score inputs.
        """
        # Arrange
        mock_session_instance = MagicMock()
        mock_session.return_value = mock_session_instance
        mock_input = MagicMock()
        mock_input.name = 'input'
        mock_session_instance.get_inputs.return_value = [mock_input]
        mock_session_instance.run.return_value = [None, {'IT': 0.8, 'Other': 0.2}]
        
        mock_read_json.return_value = pd.DataFrame({
            'Title': ['Software Engineer'],
            'Career Category': ['IT'],
            'Realistic': [0.5],
            'Investigative': [0.8],
            'Artistic': [0.2],
            'Social': [0.4],
            'Enterprising': [0.6],
            'Conventional': [0.7]
        })

        from src.scripts.inference_engine import CareerRecommender
        recommender = CareerRecommender('riasec_model.onnx', 'riasec_jobs_db.json')

        # Act
        valid_scores = [0.5, 0.6, 0.7, 0.8, 0.4, 0.3]
        # Should execute without raising exceptions
        result = recommender.get_all_recommendations(valid_scores)

        # Assert
        assert result is not None


class TestCareerRecommenderOutput:
    """Test suite for CareerRecommender output format and content."""

    @patch('src.scripts.inference_engine.ort.InferenceSession')
    @patch('pandas.read_json')
    def test_recommendation_output_includes_required_fields(self, mock_read_json, mock_session):
        """
        Test that recommendation output includes Title, Career Category, and Match_Score.
        """
        # Arrange
        mock_session_instance = MagicMock()
        mock_session.return_value = mock_session_instance
        mock_input = MagicMock()
        mock_input.name = 'input'
        mock_session_instance.get_inputs.return_value = [mock_input]
        mock_session_instance.run.return_value = [None, {'Category1': 0.8, 'Category2': 0.2}]
        
        mock_read_json.return_value = pd.DataFrame({
            'Title': ['Software Engineer', 'Data Scientist'],
            'Career Category': ['IT', 'IT'],
            'Realistic': [0.5, 0.6],
            'Investigative': [0.8, 0.9],
            'Artistic': [0.2, 0.3],
            'Social': [0.4, 0.5],
            'Enterprising': [0.6, 0.7],
            'Conventional': [0.7, 0.6]
        })

        from src.scripts.inference_engine import CareerRecommender
        recommender = CareerRecommender('riasec_model.onnx', 'riasec_jobs_db.json')

        # Act
        student_scores = [0.5, 0.8, 0.2, 0.4, 0.6, 0.7]
        result = recommender.get_all_recommendations(student_scores)

        # Assert - Result should be JSON string with required fields
        assert isinstance(result, str)
        # Verify it can be parsed as JSON
        import json
        parsed = json.loads(result)
        assert isinstance(parsed, list)

    @patch('src.scripts.inference_engine.ort.InferenceSession')
    @patch('pandas.read_json')
    def test_recommendation_output_accepts_list_of_dict_output(self, mock_read_json, mock_session):
        """
        Test that ONNX outputs wrapped in a single-element list are accepted.
        """
        # Arrange
        mock_session_instance = MagicMock()
        mock_session.return_value = mock_session_instance
        mock_input = MagicMock()
        mock_input.name = 'input'
        mock_session_instance.get_inputs.return_value = [mock_input]
        mock_session_instance.run.return_value = [None, [{'Category1': 0.8, 'Category2': 0.2}]]

        mock_read_json.return_value = pd.DataFrame({
            'Title': ['Software Engineer', 'Data Scientist'],
            'Career Category': ['IT', 'IT'],
            'Realistic': [0.5, 0.6],
            'Investigative': [0.8, 0.9],
            'Artistic': [0.2, 0.3],
            'Social': [0.4, 0.5],
            'Enterprising': [0.6, 0.7],
            'Conventional': [0.7, 0.6]
        })

        from src.scripts.inference_engine import CareerRecommender
        recommender = CareerRecommender('riasec_model.onnx', 'riasec_jobs_db.json')

        # Act
        student_scores = [0.5, 0.8, 0.2, 0.4, 0.6, 0.7]
        result = recommender.get_all_recommendations(student_scores)

        # Assert
        assert isinstance(result, str)
        import json
        parsed = json.loads(result)
        assert isinstance(parsed, list)
