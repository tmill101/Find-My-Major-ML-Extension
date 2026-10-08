"""
File: metrics.py
Path: src/evaluation/metrics.py

Purpose:
  Implements evaluation metrics for RIASEC career recommendation models.
  Provides cosine similarity for RIASEC vectors and rank-based metrics (NDCG@K, Precision@K).
  All functions are pure and deterministic.

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
  - RIASEC vectors are length 6 with values in [0,1]
  - Ground truth is career category string
  - Predictions are DataFrame with columns [Title, Career Category, Match_Score]
  - No external dependencies beyond numpy and pandas

Related Docs:
  - docs/src/evaluation/evaluation.md
  - docs/data_contracts.md
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any


def cosine_similarity(vector1: List[float], vector2: List[float]) -> float:
    """
    Name: cosine_similarity

    Purpose:
      Computes cosine similarity between two RIASEC vectors.

    Inputs:
      - vector1: List[float] — first RIASEC vector (length 6)
      - vector2: List[float] — second RIASEC vector (length 6)

    Outputs:
      - float — cosine similarity in range [-1, 1]

    Raises / Errors:
      - ValueError: if vectors have different lengths or are not length 6

    Notes:
      - Returns 1.0 for identical vectors, -1.0 for opposite vectors
      - Handles zero vectors by returning 0.0
    """
    if len(vector1) != len(vector2) or len(vector1) != 6:
        raise ValueError("RIASEC vectors must both be length 6")

    v1 = np.array(vector1)
    v2 = np.array(vector2)

    dot_product = np.dot(v1, v2)
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)


def ndcg_at_k(predictions: pd.DataFrame, ground_truth_category: str, k: int) -> float:
    """
    Name: ndcg_at_k

    Purpose:
      Computes Normalized Discounted Cumulative Gain at K for career recommendations.

    Inputs:
      - predictions: pd.DataFrame — model predictions with columns [Title, Career Category, Match_Score]
      - ground_truth_category: str — correct career category
      - k: int — number of top recommendations to consider

    Outputs:
      - float — NDCG@K score in range [0, 1]

    Raises / Errors:
      - ValueError: if predictions DataFrame missing required columns

    Notes:
      - Relevance is binary: 1 if career category matches ground truth, 0 otherwise
      - Returns 0.0 if no relevant items in top K
    """
    required_cols = ['Title', 'Career Category', 'Match_Score']
    if not all(col in predictions.columns for col in required_cols):
        raise ValueError(f"Predictions DataFrame must contain columns: {required_cols}")

    # Get top K predictions sorted by Match_Score descending
    top_k = predictions.nlargest(k, 'Match_Score')

    # Compute DCG
    dcg = 0.0
    for i, (_, row) in enumerate(top_k.iterrows()):
        relevance = 1.0 if row['Career Category'] == ground_truth_category else 0.0
        dcg += relevance / np.log2(i + 2)  # i+2 because positions start at 1

    # Compute IDCG (ideal DCG)
    num_relevant = (predictions['Career Category'] == ground_truth_category).sum()
    idcg = 0.0
    for i in range(min(k, num_relevant)):
        idcg += 1.0 / np.log2(i + 2)

    return dcg / idcg if idcg > 0 else 0.0


def precision_at_k(predictions: pd.DataFrame, ground_truth_category: str, k: int) -> float:
    """
    Name: precision_at_k

    Purpose:
      Computes Precision at K for career recommendations.

    Inputs:
      - predictions: pd.DataFrame — model predictions with columns [Title, Career Category, Match_Score]
      - ground_truth_category: str — correct career category
      - k: int — number of top recommendations to consider

    Outputs:
      - float — Precision@K score in range [0, 1]

    Raises / Errors:
      - ValueError: if predictions DataFrame missing required columns

    Notes:
      - Returns fraction of top K recommendations that match ground truth category
    """
    required_cols = ['Title', 'Career Category', 'Match_Score']
    if not all(col in predictions.columns for col in required_cols):
        raise ValueError(f"Predictions DataFrame must contain columns: {required_cols}")

    # Get top K predictions
    top_k = predictions.nlargest(k, 'Match_Score')

    # Count relevant items in top K
    relevant_count = (top_k['Career Category'] == ground_truth_category).sum()

    return relevant_count / k

def recall_at_k(predictions: pd.DataFrame, ground_truth_category: str, k: int) -> float:
    """
    Name: recall_at_k

    Purpose:
      Computes Recall at K for career recommendations.

    Inputs:
      - predictions: pd.DataFrame — model predictions with columns [Title, Career Category, Match_Score]
      - ground_truth_category: str — correct career category
      - k: int — number of top recommendations to consider

    Outputs:
      - float — Recall@K score in range [0, 1]
    """
    required_cols = ['Title', 'Career Category', 'Match_Score']
    if not all(col in predictions.columns for col in required_cols):
        raise ValueError(f"Predictions DataFrame must contain columns: {required_cols}")

    # Total relevant items the model returned overall
    total_relevant = (predictions['Career Category'] == ground_truth_category).sum()

    if total_relevant == 0:
        return 0.0

    # Get top K predictions
    top_k = predictions.nlargest(k, 'Match_Score')

    # Count relevant items specifically in the top K
    relevant_count = (top_k['Career Category'] == ground_truth_category).sum()

    return relevant_count / total_relevant

def compute_all_metrics(predictions: pd.DataFrame, ground_truth_category: str, k_values: List[int]) -> Dict[str, Any]:
    """
    Name: compute_all_metrics

    Purpose:
      Computes all evaluation metrics for a single prediction set.

    Inputs:
      - predictions: pd.DataFrame — model predictions with columns [Title, Career Category, Match_Score]
      - ground_truth_category: str — correct career category
      - k_values: List[int] — K values for rank-based metrics

    Outputs:
      - Dict[str, Any] — metric results with keys like 'ndcg@5', 'precision@5', etc.

    Raises / Errors:
      - ValueError: if predictions DataFrame missing required columns

    Notes:
      - Returns metrics for each K value in k_values
    """
    results = {}

    for k in k_values:
        results[f'ndcg@{k}'] = ndcg_at_k(predictions, ground_truth_category, k)
        results[f'precision@{k}'] = precision_at_k(predictions, ground_truth_category, k)
        results[f'recall@{k}'] = recall_at_k(predictions, ground_truth_category, k)  # added recall metric

    return results