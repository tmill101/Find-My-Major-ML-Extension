"""
File: analyze_batch_recommendations.py
Path: src/scripts/analyze_batch_recommendations.py

Purpose:
  CLI utility to analyze batch recommendations across a set of student profiles.
  Uses the champion Gradient Boosting model to generate and validate recommendations.

Original Author(s):
  - Nathan Meiners
  - AI Assistant

AI Tools Used:
  - GitHub Copilot - Initial implementation
  - Claude Sonnet 4.6 - Documentation

Editors:
  - AI Assistant (2026-04-20) — Added file header and relocated from scripts/ to src/scripts/

Last Editor:
  - AI Assistant

Last Edit Date:
  2026-04-20

Assumptions & Constraints:
  - Requires master training data and test data CSV files
  - Uses pre-configured Gradient Boosting model with fixed hyperparameters
  - Results compared against actual student major labels in test data
  - top_n_jobs controls number of recommendations (supports "Reject & Replace")

Related Docs:
  - docs/src/models/models.md
  - docs/evaluation_workflow.md
"""

import pandas as pd
from src.scripts.inference_engine import CareerRecommender


def analyze_batch(batch_size: int = 10) -> None:
    """
    Name: analyze_batch

    Purpose:
      Analyzes batch of student profiles using champion model and compares
      recommendations against ground truth majors.

    Inputs:
      - batch_size: int — number of students to analyze (default 10)

    Outputs:
      - Prints detailed recommendation analysis per student to console

    Raises / Errors:
      - FileNotFoundError: if data CSV files not found

    Notes:
      - Uses production-tuned model parameters
      - Validates recommendations against test data labels
    """
    # 1. Load the 6-feature master data and test dataset
    # These paths are defined in your project architecture
    master_df = pd.read_csv("docs/data/master_careers_riasec_categories.csv")
    test_df = pd.read_csv("docs/data/Kaggle_Cleaned_Mapped_Categories.csv")

    # 2. Define the 6 RIASEC features
    features = ["Realistic", "Investigative", "Artistic", "Social", "Enterprising", "Conventional"]

    # 3. Initialize the career recommender
    recommender = CareerRecommender()

    # 4. Select a batch of users to analyze (e.g., first 10 users)
    actual_features = [feature for feature in features if feature in test_df.columns]
    if len(actual_features) < len(features):
        numeric_features = [
            col for col in test_df.select_dtypes(include='number').columns
            if col not in {'major'}
        ]
        actual_features = numeric_features[:6]

    for i in range(batch_size):
        student = test_df.iloc[[i]]
        actual_major = student['major'].values[0]
        student_scores = student[actual_features].iloc[0].tolist()

        print(f"\n=== Student {i+1} ===")
        print(f"Actual Major: {actual_major}")

        # Run inference
        recommendations = recommender.get_top_n_recommendations(student_scores, top_n=3)

        print("Top 3 Recommendations:")
        print(recommendations)


if __name__ == "__main__":
    analyze_batch(batch_size=10)
