"""
File: build_riasec_json.py
Path: src/scripts/build_riasec_json.py

Purpose:
  CLI script to train a final production Gradient Boosting classifier and export it
  as ONNX format for browser-compatible inference. Also exports lightweight JSON
  database of careers for frontend ranking.

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
  - Master training data must exist at docs/data/master_careers_riasec_categories.csv
  - Requires sklearn, skl2onnx, and onnxruntime to be installed
  - Outputs are written to repo root: riasec_model.onnx and riasec_jobs_db.json
  - Production model uses tuned hyperparameters: depth 5, learning_rate 0.1

Related Docs:
  - docs/src/models/models.md
  - docs/data/data_governance.md
"""

import joblib
import onnxruntime as ort
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier


def build_riasec_model():
    """
    Name: build_riasec_model

    Purpose:
      Trains final production Gradient Boosting classifier and exports to ONNX
      and lightweight JSON database format.

    Inputs:
      - Reads from docs/data/master_careers_riasec_categories.csv

    Outputs:
      - Writes riasec_model.onnx (ONNX format model)
      - Writes riasec_jobs_db.json (JSON database of careers)

    Raises / Errors:
      - FileNotFoundError: if master data CSV not found

    Notes:
      - Uses production-tuned hyperparameters
      - ONNX model compatible with browser-based inference
      - JSON database includes O*NET-SOC Code for major mapping
    """
    # 1. Load the pristine 6-Feature Master Data
    print("Loading Master Data...")
    try:
        df = pd.read_csv("docs/data/master_careers_riasec_categories.csv")
    except (FileNotFoundError, pd.errors.EmptyDataError) as exc:
        raise FileNotFoundError(
            "Master training data file not found or empty: docs/data/master_careers_riasec_categories.csv"
        ) from exc
    features = ["Realistic", "Investigative", "Artistic", "Social", "Enterprising", "Conventional"]

    # 2. Train the Final Production Model 
    print("Training Final Production Model...")
    # Using tuned parameters: depth 5, learning rate 0.1
    model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=5)
    model.fit(df[features], df["Career Category"])

    # 3. Convert and Save the Model to ONNX (For browser compatibility)
    print("Exporting to ONNX...")
    try:
        from skl2onnx import convert_sklearn
        from skl2onnx.common.data_types import FloatTensorType
    except (ImportError, AttributeError) as exc:
        raise ImportError(
            "Exporting to ONNX requires compatible skl2onnx and onnx packages. "
            "Install or upgrade them with: "
            "python -m pip install 'onnx>=1.15,<2' 'skl2onnx>=1.16,<2'"
        ) from exc

    initial_type = [('float_input', FloatTensorType([None, 6]))]
    onx = convert_sklearn(model, initial_types=initial_type)

    with open("riasec_model.onnx", "wb") as f:
        f.write(onx.SerializeToString())

    # 4. Export the lightweight JSON database for the F1 Team
    print("Exporting Jobs Database for Frontend...")
    # Includes the O*NET-SOC Code for their MSU Major mapping
    frontend_db = df[["O*NET-SOC Code", "Title", "Career Category"] + features]
    frontend_db.to_json("riasec_jobs_db.json", orient="records")


if __name__ == "__main__":
    build_riasec_model()
