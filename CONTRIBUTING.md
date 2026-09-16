# Contributing to MLForensics

Thanks for helping make ML debugging more reliable. Small focused changes with
clear evidence are easiest to review.

## Development setup

Use Python 3.10 or newer and create an isolated environment:

```console
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'
```

Optional framework behavior should be tested with the relevant extra, for
example `python -m pip install -e '.[dev,onnx]'`. Do not make PyTorch, NumPy,
ONNX Runtime, fsspec, MLflow, Weights & Biases, or PyYAML an import-time
requirement of the base package.

## Checks

Run the checks that cover your change, followed by the full release suite before
requesting review:

```console
python -m pytest
ruff check mlforensics tests
ruff format --check mlforensics tests
python -m build --sdist --wheel
```

Tests should be deterministic, avoid network access, and use temporary paths for
capsules, caches, Git repositories, and artifact stores. Optional integrations
should be tested with small fakes where practical so the normal suite does not
require external accounts or heavyweight runtimes.

## Design expectations

- Keep the run capsule and core records framework-neutral.
- Preserve local, serverless operation as the default.
- Validate evidence at boundaries and report missing or inconclusive evidence
  explicitly. A missing observation is not a passing observation.
- Keep capture overhead bounded. Traces, samples, and process output need clear
  limits.
- Preserve capsule compatibility within a schema version. A schema change needs
  migration/compatibility handling and release notes.
- Import optional dependencies lazily and raise an actionable error that names
  the required extra.
- Never add unsafe deserialization merely for convenience. In particular, do not
  accept arbitrary pickle-based model files.
- Keep generated reports machine-readable as well as useful to a person.

## Changes and tests

Bug fixes should include a regression test that fails without the fix. New CLI
behavior should test both the exit code and structured output. New capsule fields
or files should test round trips, integrity failures, and directory/ZIP parity.
Statistical changes should cover deterministic seeds, insufficient evidence, and
decision boundaries.

Update README examples and the changelog when public behavior changes. Keep
commits free of generated build products, local capsules, caches, credentials,
and datasets.

## Reporting security issues

Do not submit public patches or issues containing an exploitable vulnerability,
secret, or sensitive capsule. Follow [SECURITY.md](SECURITY.md) instead.

By contributing, you agree that your contribution is licensed under the
project's MIT License.
