# CI Design

This document defines the Continuous Integration (CI) structure for this repository.

The CI system must be:

* Simple
* Deterministic
* Easy to maintain
* Strict enough to prevent structural decay

The CI workflow will be implemented using GitHub Actions (`.github/workflows/ci.yml`).

---

# 1. Design Principles

1. There must be a single entry point for all tests.
2. The `tests/` directory must mirror the structure of `src/`.
3. CI should enforce structure and discipline, not complexity.
4. CI must remain readable and maintainable.
5. CI should fail fast on structural violations.

---

# 2. Test Directory Structure

The `tests/` directory must directly reflect `src/`.

Example:

```
src/
    config/
        config_loader.py
    data/
        loader.py
    evaluation/
        metrics.py
    models/
        baseline.py
    scripts/
        inference_engine.py

tests/
    test_config_loader.py
    data/
        test_loader.py
    evaluation/
        test_metrics.py
    models/
        test_baseline.py
    export/
        test_export.py
```

Rules:

* Every production module must have a corresponding test file.
* Test file names must begin with `test_`.
* Tests must not depend on notebooks.
* Tests must not rely on external network calls.

---

# 3. Single Test Entry Point

All tests must be runnable through a single command:

```
pytest
```

CI must only execute this command.

No custom per-module test scripts.
No multiple entry points.

This ensures CI configuration remains simple and predictable.

---

# 4. Required CI Checks

CI must perform the following steps:

1. Install dependencies
2. Run tests (`pytest`)
3. Fail if tests fail

Optional but recommended additions:

* Linting (e.g., flake8 or ruff)
* Static type checking (if type hints are used)
* Formatting validation (e.g., black --check)

These may be introduced incrementally.

---

# 5. Structural Enforcement Rules

CI should fail if:

* Tests import from notebooks
* Evaluation logic is defined inside model modules
* Code cannot be imported cleanly
* There are syntax errors

Advanced enforcement (optional future step):

* Verify that new `src/` modules include corresponding test files
* Verify documentation parity

---

# 6. Determinism Requirements

Tests must:

* Use fixed random seeds
* Avoid flaky timing-based assertions
* Avoid reliance on GPU availability

CI must produce consistent results across runs.

---

# 7. Branch Strategy

Recommended workflow:

* All changes occur in feature branches
* CI must pass before merging into main
* Direct commits to main should be restricted

Pull requests must include:

* Passing CI
* Updated documentation if behavior changed
* Tests for new logic

---

# 8. AI-Generated Code in CI

If AI generates code:

* CI acts as a structural filter
* Tests must pass before merge
* Human review remains mandatory

CI is not a substitute for review.

---

# 9. Future Extensions

Possible later additions:

* Coverage thresholds
* Dataset integrity checks
* Evaluation reproducibility checks
* Performance regression checks against baseline

These should only be added when architecture stabilizes.

---

This CI design prioritizes clarity and structural enforcement over automation complexity. It is intentionally minimal but strict.
