"""
File: test_engine.py
Path: src/scripts/test_engine.py

Purpose:
  CLI utility to test the inference engine with sample student profiles.
  Validates ONNX model loading, database connectivity, and recommendation ranking.

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
  - Requires riasec_model.onnx and riasec_jobs_db.json in repo root
  - Test student profile must have 6 RIASEC values
  - Run from repository root directory

Related Docs:
  - docs/src/models/models.md
"""

import json
from src.scripts.inference_engine import CareerRecommender


def test_inference():
    """
    Name: test_inference

    Purpose:
      Validates inference engine by running recommendations on a test student profile.

    Inputs:
      - None (uses hardcoded test student)

    Outputs:
      - Prints formatted top 10 recommendations to console

    Raises / Errors:
      - FileNotFoundError: if model or database files missing
      - RuntimeError: if ONNX inference fails

    Notes:
      - Test student is high Realistic / high Investigative
      - For debugging and integration testing only
    """
    print("--- Initializing Inference Engine ---")
    engine = CareerRecommender(
        model_path="riasec_model.onnx", 
        db_path="riasec_jobs_db.json"
    )

    # High Realistic / High Investigative test student
    test_student = [0.9, 0.8, 0.1, 0.2, 0.3, 0.4]

    print(f"\n--- Running Inference for Realistic-Investigative Student ---")
    # 1. Get the JSON string from the engine
    json_results = engine.get_all_recommendations(test_student)

    # 2. Parse the JSON string back into a Python list of dictionaries
    results_list = json.loads(json_results)

    # 3. Now we can correctly count the jobs
    print(f"Total Careers Ranked: {len(results_list)}")
    print("\nTop 10 Recommendations:")
    
    # 4. Loop through the first 10 items in the list and print them cleanly
    print(f"{'O*NET-SOC Code':<15} | {'Title':<45} | {'Match_Score':<10}")
    print("-" * 75)
    for job in results_list[:10]:
        code = job.get("O*NET-SOC Code", "N/A")
        title = job.get("Title", "N/A")[:43]  # Truncate long titles for display
        score = job.get("Match_Score", 0)
        print(f"{code:<15} | {title:<45} | {score:>10.4f}")


if __name__ == "__main__":
    test_inference()
