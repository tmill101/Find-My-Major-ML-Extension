# Model Card: Baseline Career Recommender (Random Forest)

## 1. Model Details

* **Model Type:** Hybrid Random Forest Classifier + Cosine Similarity Ranking
* **Version:** 1.0.0 (Baseline)
* **Date:** 2024-03-15
* **Author:** Nate G.
* **Architecture:** * **Step 1 (Macro):** Random Forest Classifier (100 estimators) used for "Candidate Generation" (predicting top 3 broad career categories).
  * **Step 2 (Micro):** Cosine Similarity used for "Ranking" (finding specific job matches within predicted categories).

## 2. Intended Use

* **Primary Use:** Recommending career paths to students based on 6-point normalized RIASEC interest profiles.
* **Target Audience:** Undergraduate students exploring degree-pathway alignment.
* **Out-of-Scope:** This model is a baseline and should not be used for clinical psychological assessment.

## 3. Training Data

* **Kaggle Student Profiles:** ~80,000 anonymized student interest profiles.
* **O*NET Career Database:** ~700+ occupations mapped to 10 custom career categories (including Aviation and Trades).
* **Preprocessing:** Standard Min-Max normalization of RIASEC scores (0.0 to 1.0).

## 4. Performance & Metrics

* **Single-Category Accuracy:** ~28%
* **Top-3 Category Accuracy:** ~74% (estimated via validation)
* **Design Rationale:** Due to the inherent overlap in RIASEC traits between complex fields (e.g., Psychology vs. Education), the model is optimized for **Top-3 Category Recall** rather than single-label precision.

## 5. Technical Specifications & Storage

* **Model Size:** 1.3 GB (`baseline_rf_model.joblib`)
* **Storage Status:** **Locally Managed / Git-Ignored.** * Due to GitHub’s 100MB file size limit, the serialized model weights are maintained locally by the lead ML developer.
  * The model can be fully regenerated using the `notebooks/03_random_forest_baseline.ipynb` file.
* **Dependencies:** `scikit-learn`, `pandas`, `numpy`, `joblib`.

## 6. Limitations & Biases

* **Input Constraints:** This baseline model only utilizes the 6 RIASEC traits. Performance is expected to improve significantly with the integration of Big Five (OCEAN) personality traits.
* **Label Noise:** Career categories were mapped semi-heuristically; further refinement of the "Social Sciences" vs. "Psychology" labels may be required.
