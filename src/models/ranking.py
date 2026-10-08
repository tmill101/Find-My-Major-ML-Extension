"""
File: ranking.py
Path: src/models/ranking.py

Purpose:
  Provides the shared cosine similarity ranking function used by all model
  implementations to score and select top job recommendations. Centralizing
  this logic enforces consistent ranking behavior across all models and
  prevents silent metric redefinition (coding rule 3.6).

Original Author(s):
  - Angela Fleenor
  - AI Assistant (Claude Sonnet 4.6)

AI Tools Used:
  - Claude Sonnet 4.6 - Code generation and documentation

Editors:
  - AI Assistant (2026-03-20) — Initial implementation; logic adapted from baseline.py:82-95

Last Editor:
  - AI Assistant

Last Edit Date:
  2026-03-20

Assumptions & Constraints:
  - onet_db must contain all columns listed in feature_cols
  - onet_db must contain the column named by category_col
  - student_vector must be a single-row DataFrame with columns matching feature_cols
  - predicted_categories must be a non-empty list of category strings

Related Docs:
  - docs/src/models.md
"""

import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


def rank_jobs(
    student_vector: pd.DataFrame,
    onet_db: pd.DataFrame,
    feature_cols: list,
    top_n_categories: int,
    top_n_jobs: int,
    category_col: str,
    predicted_categories: list,
) -> pd.DataFrame:
    """
    Name: rank_jobs

    Purpose:
      Filters the O*NET career database to a candidate set of jobs matching
      predicted career categories, then ranks them by cosine similarity to
      the student's feature vector. Returns the top-N highest-scoring jobs.

    Inputs:
      - student_vector: pd.DataFrame — single-row DataFrame with feature columns matching feature_cols
      - onet_db: pd.DataFrame — career database; must contain feature_cols and category_col
      - feature_cols: list[str] — feature column names to use for similarity; from self.x_features
      - top_n_categories: int — number of categories already selected; used for documentation only
      - top_n_jobs: int — number of job rows to return; from config evaluation.top_k
      - category_col: str — name of the category column in onet_db (e.g. "Career Category")
      - predicted_categories: list[str] — career category labels predicted by the model

    Outputs:
      - pd.DataFrame with columns [Title, category_col, Match_Score] and at most top_n_jobs rows.
        May return fewer rows if the filtered candidate set is smaller than top_n_jobs.

    Raises / Errors:
      - KeyError: if onet_db is missing any column in feature_cols or the category_col
      - KeyError: if student_vector is missing any column in feature_cols

    Notes:
      - All column names and counts are passed in — nothing is hardcoded in this function.
      - Match_Score is cosine similarity in [0, 1].
      - Adapted from baseline.py get_job_recommendations() lines 82-95.
    """
    # Filter career database to candidate categories
    candidates = onet_db[onet_db[category_col].isin(predicted_categories)].copy()

    output_cols = ["Title", category_col, "Match_Score"]
    if "O*NET-SOC Code" in onet_db.columns:
        output_cols = ["O*NET-SOC Code", *output_cols]

    if candidates.empty:
        return pd.DataFrame(columns=output_cols)

    # Compute cosine similarity between student vector and each candidate job
    job_vectors = candidates[feature_cols].values
    student_array = student_vector[feature_cols].values

    similarities = cosine_similarity(student_array, job_vectors)[0]
    candidates["Match_Score"] = similarities

    # Return top-N by similarity score
    best = candidates.sort_values(by="Match_Score", ascending=False).head(top_n_jobs)
    
    return best[output_cols].reset_index(drop=True)
