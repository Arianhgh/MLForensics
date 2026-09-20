# MLForensics

MLForensics is a local-first Python toolkit for reliability, regression testing,
and debugging of machine-learning systems. It connects capture, statistical
comparison, replay, shrinking, tensor tracing, change-impact analysis, backend
parity, and CI through one portable evidence format: the run capsule.

It is not an experiment tracker, monitoring dashboard, or workflow orchestrator.
There is no required server. Version 0.1 is an alpha release: the capsule schema is
versioned, but public APIs may still change before 1.0.

## Installation

MLForensics requires Python 3.10 or newer.

```console
python -m pip install mlforensics
```

When working from a source checkout:

```console
python -m pip install -e '.[dev]'
```

The core package has no mandatory ML-framework dependency. Install only the
runtime adapters you need:

```console
python -m pip install 'mlforensics[numpy]'
python -m pip install 'mlforensics[torch]'
python -m pip install 'mlforensics[onnx]'
python -m pip install 'mlforensics[mlflow]'
python -m pip install 'mlforensics[wandb]'
python -m pip install 'mlforensics[dvc]'
python -m pip install 'mlforensics[s3]'    # or gcs / azure / fsspec
```

## Quick start

Capture any external command:

```console
mlforensics run \
  --output runs/baseline.mlcap \
  --data train=data/train.csv \
  python train.py
```

This always records the command result, bounded stdout/stderr, Git metadata when
available, interpreter and dependency information, allowlisted environment
variables, hardware information, dataset fingerprints, elapsed/CPU time, peak
RSS, and exceptions. The command exits with the child process's exit status and
saves a capsule on both success and failure.

To capture ML-specific evidence such as metrics from the child process, add
lightweight instrumentation to the training program:

```python
import mlforensics

with mlforensics.capture(name="train") as run:
    for step in range(3):
        loss = 1.0 / (step + 1)
        run.record_metric("loss", loss, step=step)
    run.record_event({"name": "checkpoint", "payload": {"step": 2}})
```

When this program runs under `mlforensics run`, the child automatically inherits
the parent run ID, writes a handoff capsule, and its metrics, events, artifacts,
and failure evidence are merged into the parent capsule. When it runs by itself,
`mlforensics.capture()` writes `.mlforensics/<run-id>.mlcap` by default.

Compare two captured runs, then use the stricter CI command when the result should
control a process exit code:

```console
mlforensics compare runs/baseline.mlcap runs/candidate.mlcap \
  --threshold accuracy=0.002 \
  --threshold loss=0.01

mlforensics ci \
  --baseline runs/baseline.mlcap \
  --candidate runs/candidate.mlcap \
  --threshold accuracy=0.002
```

`compare` reports evidence and always exits successfully when the comparison can
be computed. `ci` exits with status 1 when a configured regression or a built-in
data-quality/run-health check fails. With `--json`, `run` emits one JSON report
on stdout; child output is retained in that report instead of being mixed into
the machine-readable stream.

### One run per side is not a result

A training run is stochastic, so a single run tells you what happened once, not
what the change does. The steps inside one run are successive measurements of one
evolving model, not independent repeats, so `compare` reports the observed delta
and stops short of a verdict when each side has one run:

```console
accuracy: delta=0.0205 from 1 paired repetition(s); at least 2 are required
          before run-to-run variation can be estimated
Overall: INCONCLUSIVE
```

Pass one capsule per seed to get a real verdict. Seeds are paired across sides,
which removes between-seed variance from the comparison:

```console
mlforensics compare runs/base-11.mlcap runs/cand-11.mlcap \
  --baseline-run runs/base-29.mlcap --baseline-run runs/base-37.mlcap \
  --candidate-run runs/cand-29.mlcap --candidate-run runs/cand-37.mlcap

mlforensics ci \
  --baseline runs/base-11.mlcap --baseline runs/base-29.mlcap \
  --candidate runs/cand-11.mlcap --candidate runs/cand-29.mlcap
```

Set `metadata={"seed": seed}` on capture so runs pair by seed. Runs without a
declared seed, sample, trial, fold, or configuration identity remain unpaired;
list order is never treated as experimental identity. A producer that records
genuinely independent observations inside one run (per-fold or per-sample
scores) can instead declare their identities, and those are used as repetitions
directly.

## The run capsule

A `.mlcap` path is a human-inspectable directory. A path ending in `.zip` is a
deterministic archive of the same format. Every capsule has canonical records and
a SHA-256 manifest; embedded artifacts are content-addressed and checked on load.
Structured views are emitted only when the corresponding evidence exists:

```text
run.mlcap/
├── manifest.json
├── capsule.json
├── run.json
├── code.json
├── environment.json
├── hardware.json
├── dependencies.json
├── data/
│   ├── fingerprints.json
│   └── schema.json
├── randomness/
│   ├── python.rng
│   ├── numpy.rng
│   └── <framework>.rng
├── training/
│   ├── metrics.json
│   └── events.jsonl
├── model/signature.json
├── system/resource_trace.json
├── failure/
│   ├── exception.json
│   └── tensor_trace.json
├── replay/
│   ├── inputs.json
│   └── state.json
└── artifacts/<sha256>
```

The protocol definition and JSON Schema ship with the package. Applications can
inspect them through `RunCapsule.protocol_document()` and
`RunCapsule.protocol_schema()`. `RunCapsule.load()` verifies every member while
optionally retaining only selected files or omitting artifact payloads; callers
can also enforce per-file/total-size limits, migrations, and detached signature
verification.

The immutable core capture API can attach replay inputs, named state providers,
models, lineage, and embedded artifacts directly:

```python
from mlforensics import CaptureContext, RunCapsule

optimizer_state = {"step": 7, "learning_rate": 1e-3}

with CaptureContext(
    name="candidate",
    replay_input={"features": [0.1, 0.3]},
    replay_seed=29,
    state_providers={"optimizer": lambda: optimizer_state},
) as capture:
    capture.record_metric("accuracy", 0.913, step=1)
    capture.record_resource("latency_ms", 17.2, units="ms")
    capture.artifact(b"model bytes", name="model.bin")

capture.capsule.save("candidate.mlcap")
loaded = RunCapsule.load("candidate.mlcap")
assert loaded.run_id == capture.run_id
```

Providers may be callables or expose `snapshot()` or `state_dict()`. Nested
state is encoded as a typed tree (`mlforensics.container`): integer mapping
keys, tuples, bytes, NumPy arrays, and framework tensors keep dtype, shape,
and device instead of being coerced into JSON. Snapshot failures are recorded
as evidence instead of silently pretending the state is replayable.

For single-process eager PyTorch training, `capture_training(model, optimizer)`
takes a pre-step checkpoint on `training.step(...)` so replay can restore the
last known-good model/optimizer state and the offending batch.

## CLI workflows

All commands that produce reports support `--json`. A capsule argument may be a
path, a name below the configured storage root, a recorded run ID, or an alias
from `<storage-root>/index.json`. The default root is `.mlforensics/`.

### Capture

```console
mlforensics run [--output PATH] [--repo PATH] \
  [--data NAME=PATH ...] [--env NAME ...] [--no-output] COMMAND...
```

`--data` accepts files or directories and may be repeated. `--env` opts an
additional variable into capture; all other environment capture is allowlisted.
Options for `run` must appear before the child command. The default output is
configured by `[storage].root`, otherwise `.mlforensics/`.

External process observation cannot infer framework-internal model architecture,
optimizer/scheduler state, metrics, random state inside an arbitrary child
process, or its last batches. Record that evidence in-process with
`mlforensics.capture()`, `CaptureContext`, or a framework-specific hook.
Without child instrumentation, interpreter and dependency evidence describes the
MLForensics launcher environment; it may differ from a child that selects another
Python installation or runs inside a separate container.

### Compare

```console
mlforensics compare BASELINE CANDIDATE \
  [--baseline-run CAPSULE ...] [--candidate-run CAPSULE ...] \
  [--confidence 0.95] [--resamples 2000] \
  [--threshold METRIC=VALUE ...]
```

Repeat `--baseline-run`/`--candidate-run` to add seeded repetitions to each side.

Metric observations are paired by their declared stable identities for
deterministic bootstrap resampling. The report includes deltas, confidence
intervals, practical threshold decisions, failed/missing evidence, resource
regressions, and—when prediction evidence is present—behavior, calibration, and
slice differences.
Common names such as `loss`, `latency`, and `memory` are treated as
lower-is-better; other metrics default to higher-is-better. The Python API accepts
an explicit direction mapping when name-based inference is not appropriate.

### Stochastic Git bisect

```console
mlforensics bisect \
  --good v1.4 \
  --bad HEAD \
  --command 'python evaluate.py' \
  --metric val_f1 \
  --higher-is-better \
  --regression 0.01 \
  --seed 11 --seed 29 --seed 37 --seed 53 --seed 71 \
  --max-runs 60
```

A revision is only classified once enough seeds agree, so at least
`--min-observations` seeds (5 by default) must be supplied. Lowering
`--min-observations` trades evidence for wall-clock time.

The command receives `MLFORENSICS_SEED` and should print a numeric value, or a
JSON object containing the requested `--metric`. A JSON object with `score`,
`metric`, `value`, `passed`, or `good` is also understood when `--metric` is not
set. Runs are cached in `.mlforensics/bisect-cache.json` by default. Unusable or
tied evidence is reported as inconclusive, not silently classified as good.

Bisect checks out revisions in the supplied repository and restores the original
branch or detached revision afterward. Use a clean checkout or dedicated Git
worktree; MLForensics does not stash or discard local changes. The default metric
direction is lower-is-better, so use `--higher-is-better` for accuracy-like
metrics.

### Replay

```console
mlforensics replay failed.mlcap
mlforensics replay failed.mlcap --command 'python reproduce.py' --step 42
```

Without a command, replay displays the recorded incident. With a command, it sets
`MLFORENSICS_REPLAY_CAPSULE` and, when requested, `MLFORENSICS_REPLAY_STEP`. The
replay program is responsible for reading the capsule and restoring application
state. The CLI counts a non-zero run as reproduced only when the normalized
failure signature matches, or—when the incident came from `mlforensics run`—both
the original exit code and normalized captured process output match. Use
`--allow-any-failure` only when any non-zero failure is an acceptable predicate.
A successful command is never reported as a reproduced failure.

For verified in-process restoration, use `mlforensics.replay_incident()` or
`ReplayEngine` with named state restorers. Missing inputs, missing restorers, and
signature mismatches produce explicit non-reproduction results.

### Shrink

```console
mlforensics shrink failing-input.json \
  --command 'python predicate.py' \
  --kind auto
```

The predicate receives each candidate as JSON on standard input. By convention,
exit status **non-zero means the original failure is still present**. The initial
input must fail this predicate or shrinking stops with an error. Supported CLI
strategies are `auto`, `rows`, `columns`, `tokens`, `sequence`, and `tensor`.
`--contains TEXT` is a convenient deterministic predicate for simple JSON/text
cases. The input may also be a capsule containing a JSON failure input or batch.

### Trace

```console
mlforensics trace failed.mlcap --radius 12
```

Trace inspects recorded events, finds the first event marked `abnormal` (or with a
finite fraction below 1), prints a bounded surrounding window, and follows
retained tensor parent IDs. Recording is opt-in: use `TensorTracer`, `TraceBuffer`,
or emit compatible events from an integration. Tracing does not monkey-patch an
arbitrary external process.

### Change impact

```console
mlforensics impact HEAD~1..HEAD --repo .
mlforensics impact --base origin/main --head HEAD --repo .
mlforensics impact HEAD~1..HEAD --config relationships.toml
```

Impact parses the Git diff and Python AST/import graph, maps changed line ranges
to functions and classes, propagates to dependents, and recommends focused
validation. Dataset, feature, and model relationships can be supplied in the
`[impact]` configuration shown below. This is conservative static analysis: it
cannot prove that dynamic imports, generated code, or runtime data dependencies
are unaffected.

### Backend parity

```console
mlforensics parity examples.models:reference examples.models:candidate \
  --inputs inputs.json --atol 1e-6 --rtol 1e-5 --shrink

mlforensics parity pytorch:model.pt onnx:model.onnx \
  --shape 1,3,224,224 --count 100 --edge-inputs
```

Supported backend specifications are:

- `package.module:object` or `python:package.module:object` for an importable
  callable;
- `onnx:PATH` or a `.onnx` path, using ONNX Runtime;
- `pytorch:PATH`, `torch:PATH`, or `torchscript:PATH`, using
  `torch.jit.load()`.

General pickle-based PyTorch loading is intentionally unsupported. When
`--inputs` is omitted, deterministic representative scalar or shaped inputs are
generated from `--shape`, `--count`, and `--seed`; `--edge-inputs` adds boundary
and adversarial cases. Parity exits with status 1 on a candidate mismatch or
backend error, and status 2 when the reference cannot execute or no cases are
available. A JSON input may be a list of cases, a single named-input mapping,
or an object with a `cases`/`inputs` list. Multi-input model calls use mappings
keyed by the model's input names.

### CI gate

```console
mlforensics ci \
  --baseline baseline.mlcap \
  --candidate candidate.mlcap \
  --confidence 0.95 \
  --resamples 2000 \
  --threshold accuracy=0.002
```

The gate fails on statistical/resource regressions, failed candidate runs,
non-finite observations, missing candidate evidence, behavioral regressions, or
captured parity failures. Optional evidence is checked only when present. Use
`--json` for CI annotations or downstream tooling. CI also supports
`--format text|json|junit|sarif|github` for native test-reporting integrations;
`--format json` is equivalent to `--json`.

## Configuration

The CLI loads `mlforensics.toml` in the current directory, or a file passed as a
global option before the subcommand. For consistency, `--config PATH` is also
accepted after any subcommand's options:

```console
mlforensics --config config/mlforensics.toml run python train.py
```

Example:

```toml
[storage]
root = "runs"

[ci]
confidence = 0.95
min_slice_support = 20
fail_on_missing_evidence = true
fail_on_parity_failure = true

[ci.thresholds]
accuracy = 0.002
loss = 0.01
latency_ms = 0.05

[ci.higher_is_better]
accuracy = true
loss = false
latency_ms = false

[ci.noninferiority_margins]
accuracy = 0.001

[impact.datasets.training]
path = "data/train.parquet"

[impact.features.user_profile]
implemented_by = "src/features/user_profile.py"
datasets = ["training"]

[impact.models.ranker]
implemented_by = "src/models/ranker.py"
features = ["user_profile"]
```

Relative storage paths are resolved from the configuration file's directory.
Command-line confidence and thresholds override configured values. The dedicated
`impact --config` option reads only the `[impact]` relationships from that file.
Metric practical thresholds are expressed in metric units; resource thresholds
are fractional changes. In addition to the keys above, `[ci]` accepts
`fail_on_regression`, `fail_on_failed_run`, `fail_on_nonfinite`,
`fail_on_behavior_regression`, and `min_slice_support` policy settings.

Core artifact persistence is content-addressed. `LocalArtifactStore` writes to a
local directory; `FsspecArtifactStore` supports URI-backed stores such as S3,
Google Cloud Storage, and Azure Blob when the corresponding extra and credentials
are available. Credentials belong in the provider credential chain or explicit
runtime `storage_options`, never in a capsule URI.

```python
from mlforensics import CaptureContext, FsspecArtifactStore

store = FsspecArtifactStore("s3://my-bucket/mlforensics")
with CaptureContext(name="remote-artifact", artifact_store=store) as capture:
    capture.artifact(b"immutable evidence", name="evidence.bin")
```

The package also contains optional bridges for exporting capsule
metrics/metadata to MLflow, publishing and retrieving complete capsules through
Weights & Biases artifacts, reading and exporting DVC-shaped dependency data,
and emitting OpenLineage-shaped events. These stores and bridges are explicit
Python API calls; the CLI does not upload capsules automatically.

Operational helpers are available without cloud dependencies. Use
`scan_sensitive_data()` before sharing a capsule, `garbage_collect()` for
policy-based retention (dry-run by default, with recoverable trash and legal
hold support), `AuditLogger` for append-only redacted JSONL events, and
`RemoteOperations`/`OfflineQueue` when a remote transport or an offline upload
queue is explicitly supplied. Plugins are discovered lazily through the
`mlforensics.plugins` entry-point group and negotiate protocol versions and
capabilities before provider code is imported.

## Privacy and security

Capsules are evidence bundles and may contain sensitive material. Review them
before sharing.

- Environment variables use a small default allowlist. `--env` deliberately adds
  names, and their values are stored in clear text.
- Dataset fingerprinting embeds schema and a bounded sample for CSV, JSON, and
  JSONL inputs. A cryptographic fingerprint is not anonymization.
- Child stdout and stderr are retained up to 100,000 characters each and may
  contain secrets or personal data.
- Configuration fields whose names contain `password`, `secret`, `token`, or
  `api_key` are redacted by `mlforensics run`, but this is defense in depth, not
  a substitute for reviewing the capsule.
- Replay, shrink predicates, bisect commands, and Python backend specifications
  execute user-supplied code. Use trusted commands, modules, capsules, and model
  files. ONNX and TorchScript loaders avoid Python pickle, but model parsing still
  belongs in an appropriately isolated environment.
- Loading verifies the capsule manifest and embedded artifact hashes. Integrity
  verification does not establish who created a capsule.

See [SECURITY.md](SECURITY.md) for vulnerability reporting and supported-release
policy.

## Python API and integrations

The top-level package exports versioned record types (`Run`, `RunCapsule`,
`MetricSeries`, `ResourceSeries`, `ArtifactRef`, `DatasetRef`, `ModelRef`),
capture primitives, comparison and CI functions, diagnosis tools, impact
planning, and parity adapters. Framework and service integrations remain
optional: the base installation requires no ML framework, cloud SDK, tracking
client, or YAML parser.

Useful entry points include:

- `compare_runs()` and `ci_gate()` for programmatic policy;
- `replay_incident()` for evidence-backed replay with named restorers;
- `shrink()` for custom failure predicates;
- `TensorTracer` and `TraceBuffer` for bounded tensor provenance;
- `impact_from_git()` and `ImpactPlanner` for validation planning;
- `compare_models()` and `load_backend()` for parity testing;
- `compare_features()` and `FeatureParityComparator` for Python/Pandas/SQL
  feature parity;
- `RunCapsule.protocol_schema()` and `RunCapsule.protocol_document()` for
  interchange metadata;
- `scan_sensitive_data()`, `garbage_collect()`, `AuditLogger`, and
  `discover_plugins()` for safe operational workflows;
- `FsspecArtifactStore` for optional URI-backed artifact persistence;
- `MLflowAdapter`, `WandBAdapter`, `OpenLineageAdapter`, and DVC helpers from
  `mlforensics.integrations`.

## Development and release checks

```console
python -m pytest
ruff check mlforensics tests
ruff format --check mlforensics tests
python -m build --sdist --wheel
```

For an installation smoke test, install the generated wheel into a clean virtual
environment and run `python -c "import mlforensics; print(mlforensics.__version__)"`
plus `mlforensics --help`.

Release artifacts are built from `pyproject.toml` with Hatchling. Maintainers
should update the version in `mlforensics/__about__.py`, add a
dated entry to [CHANGELOG.md](CHANGELOG.md), run the checks above, inspect both
the wheel and source distribution, and publish with the repository's configured
package-index tooling.

Contributions are welcome; see [CONTRIBUTING.md](CONTRIBUTING.md). MLForensics is
released under the MIT License.
