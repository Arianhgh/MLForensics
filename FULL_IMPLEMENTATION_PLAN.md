# MLForensics Full Implementation Plan

## Status

This is a status-bearing plan. Each phase records its dependency, acceptance
bar, and the evidence that currently supports it. **Done** means in-tree
regression tests cover the acceptance bar. **Partial** means code exists but
the documented workflow is incomplete. **Open** means not yet implemented.

| Phase | Status | Dependency | Acceptance evidence |
| --- | --- | --- | --- |
| 0 Correctness recovery | Done | none | 296 tests cover group-wide CI aggregation, explicit pairing identities, sparse-replay gap rejection plus intervening-step execution, restoration status, codec envelopes, reduced-input replay, typed predicates, RNG restore order, index locking, strict JSON serialization, and operational safety. |
| 1 Capsules, capture, execution | Partial | Phase 0 | `mlcap-1` schema/protocol, migrations, selective loading, integrity limits, signing hooks, factory workers, and shared `ExecutionService` are covered; distributed recovery and provider-specific remote workflows remain. |
| 2 Statistical and behavioral comparison | Partial | Phase 0–1 | Paired/independent comparisons, missing-data accounting, behavioral and performance evidence, and policy checks are covered; sequential inference, multiplicity correction, and distributed measurement remain. |
| 3 Stochastic Git bisection | Partial | Phase 2 | Matched-seed, predicate-aware, cached, budgeted isolated-worktree bisection is implemented; adaptive allocation and non-monotonic-history optimization remain. |
| 4 Deterministic incident replay | Partial | Phase 1 | Structured replay envelopes, factory workers, RNG/state restoration, checkpoint gaps, timeouts, and explicit outcomes are implemented; deeper distributed/container replay remains. |
| 5 Hierarchical failure shrinking | Partial | Phase 4 | Bounded, resumable, signature-preserving row/column/token/sequence/tensor/structured reducers are implemented; exhaustive minimality proofs remain. |
| 6 Bounded tensor and data tracing | Partial | Phase 4 | Ring buffers, tensor statistics/provenance, hooks, and divergence localization are implemented; distributed tracing and benchmark corpus remain. |
| 7 Model and feature parity | Partial | Phase 5–6 | Callable, TorchScript, ONNX, eager-export, stateful, structured, optional TensorRT/OpenVINO, feature-table, hooks, and shrink integrations are implemented; deeper provider instrumentation remains. |
| 8 Impact analysis and selective validation | Partial | Phase 2 | Python plus SQL/notebook/shell/pipeline readers, conservative lineage imports/exports, confidence/explanations, and validation planning are implemented; incremental graph caching remains. |
| 9 ML-aware CI | Partial | Phase 2, 8 | Required-evidence policy and terminal/JSON/JUnit/SARIF/GitHub reports are implemented; resumable evidence reuse and signed policy artifacts remain. |
| 10 Integrations, operations, and enterprise readiness | Partial | prior phases | Lazy plugins, capability negotiation, redaction/scanning, audit logs, legal holds, recoverable retention, offline queues, and remote facades are implemented; distributed coordination and production-provider hardening remain. |

The first release milestone remains: a new user can reproduce a documented
attention failure, identify its introducing commit, reduce its input, and
inspect the abnormal operation using only the installed package and generated
capsules.

## Objective

Build MLForensics into a local-first forensic reliability toolkit whose commands share one
portable evidence model. The complete workflow must support:

```text
capture -> compare -> identify regression -> bisect -> replay -> shrink -> trace
        -> parity -> impact-aware validation -> CI decision
```

The project is complete at 1.0 when each command works as a real end-to-end workflow, does
not silently claim success with missing evidence, and can exchange evidence through a stable
run-capsule protocol without requiring a server.

## Product boundaries

MLForensics will own forensic capture, evidence normalization, statistical comparison,
reproduction, reduction, provenance, parity, impact analysis, and CI policy. It will integrate
with experiment trackers, artifact stores, lineage systems, and profilers instead of replacing
them.

The core package must remain framework-neutral and usable with only the Python standard
library. Framework, cloud, and service dependencies belong behind adapters and plugins.

## Definition of done

Every public workflow must meet these rules:

1. A successful result is supported by explicit evidence. Missing or insufficient evidence is
   `inconclusive`, `warn`, or `fail` according to policy; it is never an implicit pass.
2. Paired experiments are paired by stable identity such as seed, sample ID, or case ID, never
   by the position left after filtering failed observations.
3. Replay, shrink, and bisect use the same structured `FailureSignature` predicate.
4. A capsule contains enough information to explain which state was captured, which state was
   omitted, and whether replay can be exact.
5. A command can be interrupted and resumed when it performs expensive work.
6. Optional integrations do not import or initialize unless selected.
7. Reports have stable JSON output in addition to readable terminal output.
8. Every high-risk algorithm has adversarial, flaky, missing-data, and end-to-end tests.
9. Each feature has a measured overhead or cost envelope and exposes its limits.
10. The README demo works from a clean install without a service, account, or database.

## Architecture

### Dependency direction

```text
capture       analysis       diagnose       impact       parity       integrations
    \             |             |              |            |              /
     \------------+-------------+--------------+------------+-------------/
                                      |
                                     core
```

`core` must never import PyTorch, ONNX Runtime, MLflow, W&B, DVC, OpenLineage, cloud SDKs,
GitHub clients, or provider-specific code.

### Stable analysis model

The canonical model should include these versioned records:

- `Run`, `RunCapsule`, `ArtifactRef`, and `EvidenceStatus`
- `DatasetRef`, `ModelRef`, `MetricSeries`, `ResourceSeries`, and `Observation`
- `RNGState`, `StateSnapshot`, `CheckpointRef`, and `ReplayPlan`
- `FailureSignature`, `Incident`, `TraceEvent`, and `Counterexample`
- `LineageNode`, `LineageEdge`, and `ImpactPlan`
- `Comparison`, `StatisticalDecision`, `ParityResult`, and `CIResult`

Each record needs strict validation, an explicit schema version, deterministic serialization,
forward-compatible metadata, and migrations for older capsule versions.

### Evidence status

All analyzers should use the same four-state result:

```text
pass | fail | warn | inconclusive
```

The result must include a reason, required evidence, observed evidence, policy applied, and a
machine-readable remediation. This avoids the current ambiguity where an unavailable metric or
an unpaired sample can look like a pass.

## Phase 0: make the current implementation trustworthy

**Status:** done for the 0.1.1 correctness-recovery scope. The original
in-tree correctness blockers (P0.1–P0.9 capture/replay/shrink/bisect/CI bugs)
and the adversarial evidence cases below have regression tests.

Complete this phase before adding broad features. These issues can currently produce false
results or prevent the core workflow.

### P0 evidence bugs covered by 0.1.1

| ID | Problem | Acceptance |
| --- | --- | --- |
| P0-group | Group comparison ignored failures outside the first run | A failure anywhere in a group triggers the configured policy; reordering capsules does not change the decision |
| P0-pairing | Missing identities paired by list position as `stable_identity`; duplicate seeds counted as independent repetitions | Missing identities cannot silently establish paired inference; duplicate capsules cannot increase effective sample size |
| P0-replay-gaps | Sparse replay restored checkpoint 0, executed only the failing input from step 2, and reported verified reproduction | Checkpoint 0 → step 2 executes intervening steps; missing history produces incomplete replay before the target executes |
| P0-restore | `strict_state=False` could omit model restoration while reporting `state_restoration_verified=True` | A matching exception with omitted model state may count as failure reproduction, but cannot count as verified state restoration |

Broader product scope from the original P0.1–P0.9 list (full parity input
contracts, CUDA RNG, distributed replay) stays with later phases.

### P0.1 Preserve failures and non-finite observations

- Allow capture of NaN and infinity as forensic observations without placing invalid floats in
  strict JSON records. Encode them as typed observation states (`finite`, `nan`, `pos_inf`,
  `neg_inf`, `missing`, `failed`).
- Ensure `capture()` always saves the original incident even if metric or artifact conversion
  fails. Record secondary capture failures separately.
- Add an emergency minimal-capsule writer for exceptions raised while normal capsule assembly is
  failing.

Acceptance: a training run that records `loss=NaN` and then raises still produces a valid capsule
containing both the non-finite metric and original failure.

### P0.2 Repair stochastic comparison and bisection

- Store observations as `seed -> outcome`; intersect matching seeds before paired inference.
- Keep failed and missing seeds in place instead of filtering each side independently.
- Treat insufficient shared seeds as inconclusive.
- Include repository identity, revision, command, metric, direction, environment fingerprint,
  configuration, and schema version in cache keys.
- Refuse a stale cache entry when any execution input differs.
- Validate that the good endpoint is healthy and the bad endpoint reproduces the target failure.
- Return a non-zero CLI status when no conclusive first-bad revision is found.

Acceptance: disjoint successful seeds cannot create a paired regression, and changing the command
cannot reuse results from an older bisection.

### P0.3 Make replay restore real state

- Introduce a state-codec registry. JSON is for simple state; NumPy arrays and framework tensors
  preserve dtype, shape, device, and bytes in embedded artifacts.
- Add first-party PyTorch codecs for model, optimizer, scheduler, GradScaler, CPU RNG, and CUDA RNG.
- Restore an exact RNG snapshot in preference to a logical seed. Use the seed only when a snapshot
  is unavailable or the replay plan explicitly requests reseeding.
- Capture replay checkpoints before a step executes. Keep a bounded checkpoint window so an
  exception can be replayed from the last known good state.
- Capture DataLoader sampler position, epoch, batch/sample IDs, and the offending batch.
- Report deterministic limitations such as unsupported CUDA kernels or state providers.

Acceptance: a PyTorch training step can be restored from a capsule and reproduce the same failure
signature and offending batch in a clean process.

### P0.4 Use the same failure predicate everywhere

- Expand `FailureSignature` beyond exceptions to support non-finite tensors, metric regressions,
  parity mismatches, timeouts, OOMs, hangs, and user-defined predicates.
- Match normalized exception type, message, phase, module/operation, and stable top frame. Do not
  depend on a fully qualified exception name appearing verbatim in ordinary stderr.
- Make the replay child emit a structured result envelope rather than inferring identity from
  unstructured logs.
- Make shrink compare the candidate failure signature with the source incident. A different
  non-zero exit is not preservation.
- Make bisect optionally classify revisions with any `FailureSignature` predicate, not only a
  numeric metric.

Acceptance: replay, shrink, and bisect agree on whether two incidents represent the same failure.

### P0.5 Repair capsule-to-shrink handoff

- Read the canonical `capsule.evidence.replay.input` field and its artifact codec.
- Prefer stable sample IDs over copied raw data when the dataset is resolvable.
- Persist shrink history, signature checks, and the final counterexample as a new capsule linked
  to its parent incident.

Acceptance: `mlforensics shrink failed.mlcap --command ...` starts from the captured replay input
without extra extraction by the user.

### P0.6 Make CI policies explicit

- Add `required_metrics`, `required_resources`, `required_evidence`, and minimum sample counts.
- A configured threshold or non-inferiority margin automatically makes that metric required.
- Distinguish absent evidence from insufficient statistical evidence.
- Remove double counting between generic regression checks and dedicated non-inferiority,
  behavior, parity, failure, and missing-evidence checks.
- Emit stable check IDs for GitHub annotations and policy exemptions.

Acceptance: configuring an accuracy threshold while both capsules omit accuracy cannot pass CI.

### P0.7 Repair parity input contracts

- Add input specifications with name, dtype, shape, dynamic dimensions, range, distribution, and
  semantic constraints.
- Infer specifications from ONNX metadata and TorchScript examples when possible.
- Add CLI `--dtype`, named multi-input support, and JSON input-spec files.
- Default generated floating-point arrays to the model's dtype rather than NumPy float64.
- Validate reference execution before judging candidate parity.

Acceptance: comparing a float32 TorchScript model with itself using generated inputs passes.

### P0.8 Repair static impact propagation

- Resolve absolute and relative imports using package context.
- Resolve aliased imports, imported symbols, methods, inheritance, decorators, and common dynamic
  import patterns conservatively.
- Preserve edges for syntax errors, deleted files, renamed files, notebooks, configuration, SQL,
  and generated artifacts as unknown or conservative dependencies.
- Add confidence and explanation to each impact edge.

Acceptance: a change to `pkg.features` marks a model using `from .features import feature` as
affected.

### P0.9 Correct behavioral accounting

- Define prediction changes as class flips plus confidence-only changes.
- Report both class-flip rate and total behavioral-change rate.
- Add multiple-testing control for automatically discovered slices.
- Add numeric range and intersection slices, not only categorical equality slices.
- Separate exploratory slices from confirmatory CI gates; require held-out confirmation or a
  corrected significance threshold before failure.

Acceptance: a confidence change with the same class contributes to the reported total behavior
change, and slice search does not create an uncorrected false CI failure.

## Phase 1: capsule protocol and capture foundation

**Status:** partial. Directory/zip saves are staged and published only when
complete; `include_artifacts=False` hashes payloads without retaining them;
`mlforensics.toml` rejects unknown keys; every CLI capsule argument honors
`[storage].root` and the local run index. The `mlcap-1` JSON Schema and protocol
document are packaged, and capsule loading supports integrity-checked streaming,
limits, selective retention, migrations, signatures, and provenance. Unified
capture-session routing, distributed recovery, and provider-specific remote
transfers remain open.

### Capsule format

- Publish a JSON Schema and protocol specification for `.mlcap`.
- Support directory and deterministic archive representations.
- Add canonical file roles for code, environment, hardware, dependencies, data, randomness,
  training, model, system, failure, replay, parity, lineage, and reports.
- Keep all artifacts content-addressed and verify every manifest entry on load.
- Add schema migration, producer version, feature flags, required-reader capabilities, and a
  minimum compatible reader version.
- Add optional compression per artifact, size limits, streaming reads, partial loading, and remote
  references with explicit mutability and integrity status.
- Add optional signing and provenance attestations without requiring them for local use.

### Capture runtime

- Consolidate the immutable and append-style capture implementations into one canonical API.
- Capture Git commit, branch, dirty diff, submodules, and repository remote fingerprint.
- Capture Python, OS, containers, packages, compiler/runtime libraries, CUDA, cuDNN, GPU, CPU,
  RAM, and selected environment variables.
- Capture redacted configuration with a configurable secret policy.
- Capture data fingerprints, schema, partitions, sample IDs, and bounded sample batches.
- Capture model signature, architecture digest, parameters, optimizer, scheduler, and checkpoints.
- Capture metrics, events, stdout/stderr, CPU/RAM/GPU utilization, I/O, and process-tree resource
  use.
- Add configurable ring buffers for batches, trace metadata, logs, and checkpoints.
- Support normal exit, Python exception, signal, timeout, OOM, and parent-process crash recovery.
- Keep child-process handoff atomic and merge evidence with documented precedence.

### Storage

- Keep local disk as the zero-configuration default.
- Complete URI-backed storage for S3, GCS, Azure Blob, and generic fsspec.
- Add atomic upload, deduplication, retries, resumable transfers, cache eviction, encryption hooks,
  and offline queues.
- Provide MLflow and W&B import/export adapters while keeping capsules authoritative.

### Plugin system

- Define versioned plugin protocols for capture hooks, state codecs, artifact stores, framework
  adapters, parity backends, lineage readers/writers, reporters, and failure predicates.
- Discover plugins through Python entry points.
- Add capability negotiation and clear errors for incompatible plugin versions.
- Test that the core package imports successfully without optional dependencies installed.

## Phase 2: statistical and behavioral comparison

**Status:** paired comparison, missing-evidence CI, behavioral accounting,
performance evidence, and the packaged attention-git fixture have regression
coverage. Sequential confidence methods, multiple-comparison correction,
distributed measurement, and the full performance benchmark corpus remain open.

### Experimental design

- Pair runs by explicit seed, sample ID, fold, dataset version, and configuration keys.
- Support paired and independent designs, repeated cross-validation, multiple seeds, and grouped
  or hierarchical experiments.
- Record the selected design and reject incompatible evidence rather than guessing.

### Statistical engine

- Provide percentile and BCa bootstrap intervals, permutation tests, effect sizes, and robust
  summaries.
- Support superiority, equivalence, and non-inferiority decisions.
- Support absolute and relative practical thresholds.
- Add sequential confidence methods for evidence collected adaptively.
- Add family-wise or false-discovery correction for multiple metrics and discovered slices.
- Track failed, censored, timed-out, and missing runs explicitly.
- Make statistical methods deterministic under a report seed.

### Behavioral diff

- Compare class labels, probabilities, logits, calibration, ranking, embeddings, structured
  outputs, and user-defined prediction objects.
- Report class flips, confidence-only changes, flips to correct/incorrect, distribution shift,
  confusion changes, ECE/Brier/log-loss changes, and ranking measures.
- Discover categorical, numeric range, intersection, and user-supplied slices.
- Enforce support, correction, validation, and stability requirements for slice claims.
- Add feature perturbation and sensitivity comparisons with explicit budgets.
- Save divergent example IDs and make them direct inputs to shrink and trace.

### Performance diff

- Compare latency distributions, throughput, CPU, RAM, VRAM, GPU utilization, I/O, data wait,
  compile time, and energy when available.
- Support p50/p95/p99, maxima, steady-state windows, warmup removal, and repeated trials.
- Normalize only when hardware and environment evidence makes normalization defensible.
- Report configuration changes as evidence and ranked hypotheses, never proven causes.

## Phase 3: stochastic Git bisection

- Run each revision in an isolated worktree with a clean-state preflight.
- Use matched seeds and dataset versions across revisions.
- Implement sequential decisions that stop early when confidence is sufficient.
- Add adaptive allocation across revisions and seeds.
- Support run-count, wall-time, GPU-hour, and monetary budgets.
- Cache evidence by the complete execution identity and allow safe resume.
- Detect non-monotonic histories and switch from binary search to bounded scan or report the
  ambiguity.
- Handle build failures, dependency incompatibilities, flaky runs, merge commits, and skipped
  revisions.
- Accept numeric regressions, parity results, failure signatures, and arbitrary capsule-based
  predicates.
- Produce an evidence capsule and report for every evaluated revision.

## Phase 4: deterministic incident replay

- Define a replay protocol describing inputs, state components, restore order, entry point,
  environment, expected failure, and determinism level.
- Add PyTorch adapters for model/optimizer/scheduler/GradScaler, sampler, distributed rank state,
  CPU/CUDA RNG, and autocast configuration.
- Add periodic and trigger-based checkpoints with bounded retention.
- Restore the last checkpoint before the incident and advance to the requested step.
- Validate each restored component and list omitted or incompatible state.
- Support subprocess replay with structured IPC and in-process replay for library users.
- Compare reproduced traces and outputs against the incident, not only the final exception.
- Add best-effort environment/container reproduction metadata without making containers required.

## Phase 5: hierarchical failure shrinking

- Make all strategies operate through a resettable replay fixture and a `FailureSignature`.
- Add hierarchical delta debugging: datasets -> rows -> columns/features -> values.
- Add tabular simplification for numbers, nulls, categories, strings, and schema-preserving edits.
- Add NLP reduction for examples, documents, spans, tokens, vocabulary simplification, and length.
- Add vision reduction for examples, crops, masks, regions, channels, and resolution.
- Add tensor reduction for elements, slices, dimensions where legal, magnitudes, dtype, and shape.
- Add sequence and graph-aware reducers through plugins.
- Cache predicate results by candidate digest, detect flakiness, support quorum trials, enforce
  budgets, and resume interrupted sessions.
- Verify 1-minimality for the enabled operations or state precisely which guarantee was reached.
- Emit the minimal input, reproduction command, signature, and history as a child capsule.

## Phase 6: bounded tensor and data tracing

- Capture operation, module, shape, dtype, device, min/max/mean/std, finite percentage, gradient
  norm, source location, parent IDs, step, and rank.
- Implement PyTorch module hooks plus opt-in operator-level tracing using supported dispatch or
  graph mechanisms.
- Trace forward and backward paths and integrate anomaly detection.
- Keep metadata in a bounded ring buffer during normal execution.
- Retain full tensors only for explicit triggers, size limits, and redaction policies.
- Freeze the evidence window when a non-finite value, threshold breach, exception, or user trigger
  occurs.
- Build and render the causal path to the first abnormal transition, including missing-parent
  explanations when the ring buffer truncated history.
- Benchmark overhead by model size and tracing mode, and expose an overhead budget.
- Add distributed rank correlation and collective-event adapters later in this phase.

## Phase 7: model and feature parity

### Runtime parity

- Keep Python callable, TorchScript, and ONNX Runtime backends.
- Add eager PyTorch export paths without unsafe implicit pickle loading.
- Add FP32/FP16/BF16 tolerance profiles and output-specific tolerances.
- Support named, structured, dynamic-shape, multi-input, and stateful model inputs.
- Generate representative, boundary, adversarial, schema-derived, and user-supplied cases.
- Report per-output and per-example errors and retain the largest divergences.
- Add backend-specific intermediate-output instrumentation for real per-layer localization.
- Feed divergent cases into the common shrink engine automatically.
- Add TensorRT and OpenVINO as later optional plugins based on demand.

### Feature parity

- Define adapters for Python/Pandas, SQL/DuckDB, and arbitrary batch/row feature functions.
- Compare null behavior, types, categories, timestamps, ordering, precision, and transformations.
- Generate edge cases from schemas and observed data profiles.
- Shrink mismatches and identify the first divergent feature operation where instrumentation is
  available.

## Phase 8: impact analysis and selective validation

- Complete Python package, import, symbol, call, inheritance, and decorator analysis.
- Add configuration, SQL, notebook, shell, and pipeline declaration readers through plugins.
- Map code -> features -> datasets -> models -> evaluation suites -> deployment/export targets.
- Import DVC graphs and OpenLineage relationships; export MLForensics evidence to OpenLineage.
- Track confidence, source, and explanation for inferred and declared edges.
- Add conservative fallbacks for dynamic code and unresolved dependencies.
- Map tests and validation commands to affected nodes.
- Produce required, recommended, and likely unnecessary validation lists.
- Cache graphs by code digest and update them incrementally.

## Phase 9: ML-aware CI

- Run impact analysis first and produce a validation plan.
- Execute configured unit, data, smoke-training, comparison, parity, performance, and failure
  checks for affected models.
- Support required evidence, minimum sample sizes, confidence, practical thresholds,
  non-inferiority, budgets, and allowed exceptions.
- Select baselines by explicit capsule, Git merge base, branch policy, or approved registry tag.
- Never treat a missing baseline or unavailable check as a pass.
- Emit terminal, JSON, JUnit, SARIF, and GitHub Checks-compatible reports.
- Link each failed CI check to capsules, divergent examples, affected nodes, and remediation.
- Support resumable CI and reuse valid cached evidence.
- Record the final policy, inputs, skipped checks, cost, and decision as a signed optional report
  artifact.

## Phase 10: integrations, operations, and enterprise readiness

- Finish MLflow, W&B, DVC, OpenLineage, S3, GCS, Azure, and GitHub adapters.
- Add credential-chain use, redaction, least-privilege documentation, and no secret persistence.
- Add capsule retention, garbage collection, legal hold hooks, and configurable sensitive-data
  scanners.
- Add audit logs for remote writes and policy decisions.
- Add distributed capture coordination, rank-aware artifact merging, and partial-run recovery.
- Keep every enterprise feature optional; local capture and analysis must remain fully usable.

## Test and validation strategy

### Unit and property tests

- Property-test serialization round trips, schema migrations, digest stability, and unsafe archives.
- Property-test statistical pairing, missingness, directionality, thresholds, and cache identity.
- Property-test shrink invariants and signature preservation.
- Fuzz diff parsing, structured outputs, malformed capsules, and adapter boundaries.

### End-to-end forensic corpus

Create small deterministic fixtures for:

- NaN activation and gradient failures
- optimizer/scheduler/GradScaler replay
- data-order and sampler-state failures
- stochastic accuracy regression across matched seeds
- a non-monotonic Git history
- tabular, NLP, vision, and tensor shrink targets
- PyTorch-to-ONNX numerical divergence
- float32/float16 tolerance behavior
- relative-import and configured-lineage impact
- latency and input-pipeline performance regressions
- missing, corrupt, partial, and remotely referenced capsules

Each fixture should exercise the CLI from capture through the expected diagnosis.

### Optional dependency matrix

- Test the base package with no NumPy or ML framework.
- Test supported NumPy and PyTorch versions on CPU.
- Test ONNX Runtime separately with real exported models.
- Add a smaller CUDA matrix for RNG, checkpoint, tracing, and mixed precision.
- Smoke-test storage and service adapters with emulators or recorded contracts where practical.

### Quality gates

- Full suite, lint, format, type checking, build, package inspection, and clean-wheel smoke test.
- Coverage thresholds per risk-bearing package, not only one repository-wide percentage.
- Backward-compatibility tests for public JSON and supported capsule versions.
- Performance budgets for capture, tracing, capsule loading, impact analysis, and comparison.
- Security tests for archive traversal, decompression bombs, unsafe model formats, secret capture,
  command boundaries, and remote artifact integrity.

## Documentation and UX

- Keep a five-minute local quick start with no server.
- Publish one complete incident narrative from regression discovery to traced root cause.
- Document evidence guarantees and limitations for every command.
- Add recipes for PyTorch training, custom loops, DataLoader replay, ONNX export, GitHub Actions,
  MLflow, DVC, and OpenLineage.
- Provide stable examples for Python APIs and JSON outputs.
- Add troubleshooting for insufficient evidence, nondeterminism, optional dependencies, corrupted
  capsules, and unsupported replay state.
- Maintain a protocol specification, plugin author guide, migration guide, and release policy.

## Release sequence

### 0.1.1 - correctness recovery

Complete all Phase 0 items. Do not advertise exact replay, matched stochastic bisect, capsule
shrinking, generated parity, or strict CI until their acceptance tests pass.

### 0.2 - stable capsule and PyTorch capture

Complete Phase 1, publish the capsule specification, unify capture, and provide first-party
PyTorch state codecs and ring-buffer capture.

### 0.3 - statistical and behavioral comparison

Complete Phase 2 with explicit experimental designs, corrected slice discovery, resource
comparison, and required-evidence policy.

### 0.4 - production stochastic bisect

Complete Phase 3 with matched evidence, adaptive stopping, isolated worktrees, robust caching,
budgets, resume, and non-monotonic-history handling.

### 0.5 - verified replay and common failure signatures

Complete Phase 4 and make structured failure predicates the shared contract for diagnose tools.

### 0.6 - ML-aware hierarchical shrink

Complete Phase 5 and connect counterexamples to replay, parity, and trace.

### 0.7 - bounded tensor trace

Complete Phase 6 with measured overhead and a PyTorch forward/backward diagnosis workflow.

### 0.8 - runtime and feature parity

Complete Phase 7 for PyTorch, ONNX Runtime, Python/Pandas, and DuckDB, with localization and
shrinking.

### 0.9 - impact-driven CI

Complete Phases 8 and 9, including selective validation and CI-native reports.

### 1.0 - stable plugins and integrations

Complete Phase 10, freeze the public protocol and plugin compatibility policy, run the full
forensic corpus, and publish migration guarantees.

### 1.x and 2.x

- Add sklearn, XGBoost, and LightGBM adapters based on real user workflows.
- Add TensorRT/OpenVINO and deeper distributed support when the 1.0 contracts are stable.
- Consider JAX and TensorFlow only when their replay and tracing contracts can meet the same
  evidence standard.

## Workstream ownership and interfaces

The work can be divided without creating disconnected mini-libraries:

| Workstream | Owns | Must consume |
|---|---|---|
| Core protocol | schemas, records, migrations, artifact contracts | nothing outside core |
| Capture | environment, data, runtime, PyTorch hooks | core records and codecs |
| Statistics | experiment design, inference, decisions | core observations |
| Behavior/performance | model and resource diffs | statistics decisions |
| Bisect | revision search, budgets, cache | runner and failure predicates |
| Replay | checkpoint plans and state restoration | core codecs and signatures |
| Shrink | hierarchical reduction | resettable replay fixture and signature |
| Trace | bounded provenance | core trace events and artifact policy |
| Impact | dependency graph and validation planning | core lineage records |
| Parity | backends, input specs, localization | core parity and shrink contracts |
| CI/reporting | policies and output formats | comparison and impact results |
| Integrations | storage, trackers, lineage, GitHub | versioned plugin protocols |
| Forensic corpus | independent end-to-end fixtures | public APIs and CLI only |

One maintainer must own dependency direction, public interfaces, schema compatibility, and the
definition of evidence states across every workstream.

## Immediate execution order

1. Freeze feature additions and turn every Phase 0 reproduction into a regression test.
2. Define `Observation`, `EvidenceStatus`, and the expanded `FailureSignature`.
3. Fix capsule capture so incidents cannot be lost to serialization errors.
4. Fix bisect pairing and cache identity.
5. Implement state codecs and pre-step replay checkpoints.
6. Connect capsule replay inputs and failure signatures to shrink.
7. Add required-evidence CI policy and structured replay IPC.
8. Fix parity dtype/input specifications and relative-import impact analysis.
9. Run the full Phase 0 acceptance suite before starting Phase 1.

This order repairs the trust boundary first. Every later feature depends on correct evidence,
identity, serialization, and failure matching.
