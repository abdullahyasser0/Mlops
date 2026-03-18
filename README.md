# MLOps Assignment 4 — ML Model CI Pipeline

This repository contains a GitHub Actions CI pipeline for automated validation and testing of an ML model.

## Pipeline Overview

The pipeline runs automatically on every push to any branch **except** `main`, and on all pull requests.

### Steps

| Step | Description |
|------|-------------|
| Checkout | Fetches the repository source code |
| Set up Python | Installs Python 3.10 |
| Install Dependencies | Installs packages from `requirements.txt` |
| Linter Check | Runs `flake8` on `train.py` to enforce code style |
| Model Dry Test | Verifies the PyTorch environment is correctly installed |
| Upload project-doc | Uploads `README.md` as a GitHub artifact named `project-doc` |

## CI Trigger Strategy

The pipeline is configured to **exclude** the `main` branch from push triggers.
This enforces a branch-based workflow: developers push feature/fix branches, the pipeline validates them, and only reviewed code is merged into `main`.

```yaml
on:
  push:
    branches-ignore:
      - main
  pull_request:
```

## Green CI Run

After fixing all YAML bugs (see `assignment4-report.md`), the pipeline produces a successful (green) run as shown in the GitHub Actions tab.
