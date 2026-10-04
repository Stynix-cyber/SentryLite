# SentryLite

SentryLite is a small experimental static analysis tool for Python code.

The project combines basic AST and data-flow analysis with a lightweight
machine learning model to identify code patterns that may be worth reviewing
from a security perspective.

It's still an early project and is mainly being built to experiment with
static analysis, data-flow tracking and small local ML models.

## Current features

- Python AST analysis
- basic taint tracking
- data-flow tracking across simple function calls
- detection of potentially unsafe uses of:
  - `eval`
  - `exec`
  - `pickle`
  - `subprocess`
- lightweight logistic regression classifier
- local inference
- evaluation suite for safe and risky samples

## Example

```powershell
python -m src.inference example.py
