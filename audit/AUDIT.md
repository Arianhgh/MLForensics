# MLForensics 0.1.1 audit resolution

Audited and repaired on September 19, 2026 on Windows with Python 3.11.9. The implementation now passes the complete local suite: **304 tests passed**.

## Resolved findings

1. Grouped metric observations now use the run identity together with the observation identity. Reordering seeded candidate runs no longer changes the result.
2. CI emits and enforces a dedicated non-inferiority check instead of filtering those failures out of the final gate.
3. Directory capsule paths are normalized to portable POSIX member names, so nested evidence round-trips on Windows.
4. Timed and cancellable child execution drains stdout and stderr while waiting, preventing full pipes from causing false timeouts.
5. Offline queue failures block later operations for the same key until the failed operation succeeds, preserving per-key order.
6. Retention budgets account for capsules already selected by age, avoiding excess collection.
7. Capsule publication and run-index updates use Windows file locking as well as POSIX locking.
8. Explicit integer observation identities such as `0, 1, 2` remain identities rather than being inferred to be training steps.

The repair also addressed issues exposed by the Windows test run:

- fresh-process runners serialize closures and local callables with `cloudpickle` on spawn-based platforms;
- replay and shrink parse quoted command strings consistently across platforms;
- child CPU time is recorded through `os.times()` when the POSIX `resource` module is unavailable;
- platform-dependent tests no longer assume LF translation or permission to create Windows symlinks;
- README pairing semantics now match the implementation.

## Verification

- `python -m pytest -q`: 304 passed in 54.84 seconds.
- `python -m ruff check mlforensics tests audit/current_probes.py`: passed.
- `python -m ruff format --check mlforensics tests audit/current_probes.py`: 147 files already formatted.
- `python -m build --no-isolation --sdist --wheel`: built the source and wheel distributions.
- `python audit/current_probes.py`: all eight original reproductions now report the corrected outcomes in `current_results.json`.
- `python audit/reproduce.py`: every historical adversarial probe reports the corrected outcome; the only output outside its JSON report is PyTorch's deprecation warning for `torch.jit.trace`.

The optional ONNX Runtime, MLflow, W&B, cloud-service, GPU, and distributed-runtime paths were not exercised against live external systems in this environment. Their dependency-free adapters and mocked integration contracts remain covered by the project suite.
