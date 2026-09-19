# Changelog

All notable changes to MLForensics are documented here. The project follows
[Semantic Versioning](https://semver.org/), with the usual allowance that APIs
may change between minor releases while the package is below 1.0.

## [Unreleased]

## [0.1.1] - 2026-09-16

### Fixed

- JSON capture reports are now kept machine-readable when the child prints to
  stdout, and all subcommands accept the same config option placement.
- Parity validates dtype contracts, rejects malformed multi-input calls,
  validates the reference before judging the candidate, and distinguishes
  failed parity from inconclusive reference evidence in its exit status.
- Impact analysis retains aliased and relative imports, dynamic-import
  uncertainty, and conservative dependencies for deleted, renamed, notebook,
  configuration, data, and generated files. Edges include confidence and an
  explanation.
- Optional integration exports are resolved lazily, so importing
  `mlforensics.integrations` does not load provider modules.
- Source distributions now contain only release documentation, metadata, and
  package sources; audit artifacts and repository tests are excluded.

- `compare`/`ci` no longer treat successive training steps within one run as
  independent repetitions. A metric series from a single run is one repetition,
  so a one-run-per-side comparison now reports the observed delta and an
  inconclusive verdict instead of a confident claim built on within-run noise.
- Stochastic bisection no longer measures the wrong code. CPython validates a
  cached `.pyc` against its source size and a whole-second mtime, so revisions
  checked out within the same second were silently evaluated with earlier
  bytecode; each revision now gets its own bytecode cache prefix.
- `trace` no longer discards recorded tensor provenance. It previously read
  `evidence["tensor_trace"]` only when a run had logged no events at all, which
  is never true of a realistic run; run events and tensor evidence are now
  merged.
- Resource comparisons no longer report a regression from a single measurement
  per side. A bootstrap over one observation produces a zero-width interval that
  excludes any non-zero delta; at least two paired observations are now required,
  and a point comparison is used only against an explicitly configured threshold.
- `mlforensics.capture()` sessions expose the canonical capsule. `session.capsule`
  previously returned an internal working record whose `save()` wrote a format
  that no other command could load.
- Failure signatures now retain the originating frame, which was dropped when the
  session was converted to a portable capsule.
- `bisect` rejects too few seeds up front instead of spending the full run budget
  only to return inconclusive for every revision.
- The documented `data/fingerprints.json` and `data/schema.json` capsule views are
  emitted, derived from the datasets a run registered.

### Added

- `compare --baseline-run/--candidate-run` and repeatable `ci --baseline/--candidate`
  to compare several seeded runs per side, which is what a run-level claim requires.
- `bisect --min-observations` to set the evidence required before a revision is
  classified.
- The packaged `mlcap-1` JSON Schema and protocol document, capsule migrations,
  selective artifact/file retention, bounded loading, provenance attestations,
  and caller-provided signature verification.
- Fresh-process replay workers, resumable and domain-aware shrinkers, PyTorch
  training-state capture, and structured timeout/exception outcomes.
- Eager PyTorch export, stateful and structured parity, optional TensorRT and
  OpenVINO adapters, feature parity for Python/Pandas/SQL/DuckDB, and
  intermediate-output hooks.
- SQL, notebook, shell, and pipeline impact readers; DVC/OpenLineage-shaped
  lineage import/export; and CI reports in JUnit, SARIF, and GitHub formats.
- Lazy entry-point plugins with capability negotiation, sensitive-data scanning,
  redaction, append-only audit logs, legal holds, recoverable retention,
  offline remote queues, and safe remote-operation facades.

### Changed

- `trace` output reports the recorded tensor statistics (shape, dtype, range,
  mean, finite fraction, source) rather than only operation names.
- `parity` and slice reports show the worst cases and a remainder count instead of
  every failing case and every row index.
- Slice discovery decodes predictions and calibration terms once instead of per
  resample, roughly a 9x speedup on a 5,700-row behavioural diff.
- `TensorTracer.record` accepts `step`, and `TensorTracer.events()` is a method,
  matching `TraceBuffer.events()`.
- `behavioral_diff`/`discover_slices` accept `min_slice_support` as an alias for
  `min_support`, matching `compare_runs`/`ci_gate`.
- Capture requires the core records instead of silently degrading to a shadow data
  model, which could only hide a broken install behind weaker evidence.

## [0.1.0] - 2026-09-16

Initial alpha release.

### Added

- Versioned, integrity-checked directory and deterministic ZIP run capsules with
  canonical JSON records, structured evidence views, and embedded
  content-addressed artifacts.
- External command capture for Git, Python and package inventory, allowlisted
  environment variables, hardware, dataset fingerprints, resources, process
  output, and failures.
- In-process capture APIs for metrics, resources, events, datasets, models,
  lineage, replay inputs, random seeds, named state snapshots, and artifacts,
  including parent/child capsule handoff.
- Paired bootstrap comparison, practical regression thresholds,
  non-inferiority helpers, behavioral and slice differences, performance
  evidence, and conservative CI gating.
- Uncertainty-aware Git bisection with matched seeds, persistent caching,
  run budgets, explicit inconclusive results, and checkout restoration.
- Evidence-backed replay, hierarchical counterexample shrinking, bounded tensor
  tracing and ancestry, and failure-signature verification.
- Python/Git change-impact analysis with changed-line symbol mapping and
  configured dataset, feature, and model relationships.
- Callable, ONNX Runtime, and TorchScript parity backends, deterministic
  representative/edge input generation, output localization, and divergent-input
  shrinking.
- Local and optional fsspec-backed content-addressed artifact stores, including
  provider extras for S3, Google Cloud Storage, and Azure Blob.
- Optional MLflow, Weights & Biases, DVC-shaped, and OpenLineage-shaped
  integration helpers.
- Typed package marker, Python 3.10–3.13 support, CLI JSON output, and source/wheel
  build configuration.

### Security

- Environment capture defaults to an allowlist and run configuration metadata is
  redacted for common secret-bearing key names.
- Capsule files and embedded artifacts are verified by SHA-256 on load.
- Torch model loading is restricted to TorchScript; general pickle-based
  `torch.load()` is not used by the parity backend loader.
