# Contributor Onboarding Guide

Welcome to the project.

This guide explains how to get set up and how to contribute without violating repository standards.

The goal is to enable fast iteration while maintaining structural integrity and reproducibility.

---

# 1. Initial Setup

## 1.1 Clone the Repository

```
git clone <repo-url>
cd <repo-name>
```

## 1.2 Create a Virtual Environment

```
python -m venv .venv
source .venv/bin/activate  # macOS/Linux
.venv\Scripts\activate     # Windows
```

## 1.3 Install Dependencies

```
pip install -r environment/requirements.txt
```

## 1.4 Verify Setup

Run:

```
pytest
```

All tests must pass before you begin development.

---

# 2. Required Reading (Before Writing Code)

You must review the following documents:

1. `docs/repo_structure.md`
2. `docs/coding_rules.md`
3. `docs/ai_usage_policy.md`
4. `docs/evaluation_workflow.md`
5. `docs/data/data_governance.md`
6. `docs/experiments/stracking.md`

These define how the repository operates.

---

# 3. Development Workflow

## 3.1 Create a Feature Branch

```
git checkout -b feature/<short-description>
```

Do not commit directly to `main`.

---

## 3.2 Writing Code

Before coding:

* Confirm correct directory placement (`repo_structure.md`).
* Copy the appropriate template from `docs/templates/`.
* If using AI, start with a prompt header from `ai_usage_policy.md`.

While coding:

* Keep modules focused and small.
* Avoid embedding evaluation logic in models.
* Keep preprocessing in `src/`.

After coding:

* Add or update tests.
* Run `pytest`.
* Update documentation if behavior changed.

---

# 4. Using Notebooks

Notebooks are for:

* Exploration
* Visualization
* Demonstration

Notebooks are not for:

* Permanent business logic
* Final evaluation claims

Reusable logic must be migrated to `src/`.

---

# 5. Running Experiments

Every experiment must:

* Use an experiment ID (EXP-YYYYMMDD-XX)
* Generate structured metadata
* Be reproducible from commit + seed

Experiment artifacts are not committed.

---

# 6. Pull Request Process

Before opening a PR:

* Ensure tests pass.
* Update relevant documentation.
* Review the PR checklist (`docs/pr_review_checklist.md`).

PRs must pass CI before merging.

---

# 7. AI Usage Expectations

If you use AI:

* Disclose usage in file headers.
* Review output line-by-line.
* Do not accept code you do not understand.
* Ensure compliance with `coding_rules.md`.

AI is a productivity tool, not an architectural authority.

---

# 8. When Unsure

If a change feels architectural or high-impact:

* Pause.
* Document the reasoning.
* Consider creating an ADR.

Clarity is more important than speed.

---

This repository prioritizes structure, reproducibility, and maintainability. Contribute thoughtfully and intentionally.
