"""Command-line entry point for MLForensics."""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import shlex
import subprocess
import sys
import tempfile
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from xml.etree import ElementTree

from ..analysis import compare_runs, render_comparison
from ..capture import (
    capture_dependencies,
    capture_environment,
    capture_git,
    capture_hardware,
    fingerprint_dataset,
)
from ..ci import ci_gate
from ..core import CaptureContext, FailureSignature, RunCapsule, TraceEvent, dumps
from ..core.contracts import (
    CHILD_RESULT_ENV,
    ExecutionResult,
    load_child_result,
    parse_failure_envelope,
)
from ..core.errors import UnresolvedEvaluation
from ..core.index import record_run, resolve_run
from ..core.models import normalize_failure_message
from ..diagnose import (
    BisectCache,
    SubprocessGit,
    bisect_commits,
    capsule_replay_input,
    persist_shrink_capsule,
    shrink,
    trace_incident,
)
from ..diagnose.bisect import (
    DEFAULT_BISECT_SEEDS,
    MIN_STOCHASTIC_OBSERVATIONS,
    bytecode_isolation_env,
)
from ..impact import analyze_impact, impact_from_git
from ..parity import InputSpec, compare_models, generate_input_cases
from ..report import render
from .config import Config, load_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mlforensics", description="Forensic reliability tools for machine-learning runs"
    )
    parser.add_argument("--config", default=None, help="path to mlforensics.toml")
    sub = parser.add_subparsers(dest="subcommand", required=True)

    run = sub.add_parser("run", help="capture an external command")
    run.add_argument("--config", default=argparse.SUPPRESS, help="path to mlforensics.toml")
    run.add_argument("--output", default=None, help="capsule directory or archive path")
    run.add_argument("--repo", default=".")
    run.add_argument("--json", action="store_true", dest="as_json")
    run.add_argument("--no-output", action="store_true", help="do not print child stdout/stderr")
    run.add_argument(
        "--data",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help="fingerprint an input dataset (repeatable)",
    )
    run.add_argument(
        "--env",
        action="append",
        default=[],
        metavar="NAME",
        help="additional environment variable to capture (repeatable)",
    )
    run.add_argument("run_command", nargs=argparse.REMAINDER, help="command to execute")

    compare = sub.add_parser("compare", help="compare two sets of run capsules")
    compare.add_argument("--config", default=argparse.SUPPRESS, help="path to mlforensics.toml")
    compare.add_argument("baseline")
    compare.add_argument("candidate")
    compare.add_argument(
        "--baseline-run",
        action="append",
        default=[],
        metavar="CAPSULE",
        help="additional seeded baseline run; repeat to enable a run-level claim",
    )
    compare.add_argument(
        "--candidate-run",
        action="append",
        default=[],
        metavar="CAPSULE",
        help="additional seeded candidate run; repeat to enable a run-level claim",
    )
    compare.add_argument("--confidence", type=float, default=None)
    compare.add_argument("--resamples", type=int, default=2_000)
    compare.add_argument("--json", action="store_true", dest="as_json")
    compare.add_argument("--threshold", action="append", default=[], metavar="METRIC=VALUE")

    bisect = sub.add_parser("bisect", help="bisect a stochastic Git regression")
    bisect.add_argument("--config", default=argparse.SUPPRESS, help="path to mlforensics.toml")
    bisect.add_argument("--good", required=True)
    bisect.add_argument("--bad", required=True)
    bisect.add_argument("--repo", default=".")
    bisect.add_argument(
        "--command", required=True, help="shell command returning a metric or JSON result"
    )
    bisect.add_argument(
        "--seed",
        action="append",
        type=int,
        dest="seeds",
        default=None,
        help=f"repeatable paired seed (default: {' '.join(map(str, DEFAULT_BISECT_SEEDS))})",
    )
    bisect.add_argument("--regression", type=float, default=0.0)
    bisect.add_argument("--higher-is-better", action="store_true")
    bisect.add_argument("--metric", default=None, help="metric key in JSON command output")
    bisect.add_argument("--confidence", type=float, default=0.95)
    bisect.add_argument("--resamples", type=int, default=2_000)
    bisect.add_argument("--max-runs", type=int, default=None, help="total uncached run budget")
    bisect.add_argument(
        "--min-observations",
        type=int,
        default=None,
        help=(
            "usable paired runs required before a revision is classified "
            f"(default: {MIN_STOCHASTIC_OBSERVATIONS})"
        ),
    )
    bisect.add_argument("--cache", default=".mlforensics/bisect-cache.json")
    bisect.add_argument("--json", action="store_true", dest="as_json")

    replay = sub.add_parser("replay", help="replay a captured failure when a command is supplied")
    replay.add_argument("--config", default=argparse.SUPPRESS, help="path to mlforensics.toml")
    replay.add_argument("incident")
    replay.add_argument("--command", default=None)
    replay.add_argument("--step", type=int, default=None)
    replay.add_argument(
        "--allow-any-failure",
        action="store_true",
        help="accept any non-zero replay command instead of matching the captured signature",
    )
    replay.add_argument("--json", action="store_true", dest="as_json")

    shrink_parser = sub.add_parser("shrink", help="shrink a JSON counterexample")
    shrink_parser.add_argument(
        "--config", default=argparse.SUPPRESS, help="path to mlforensics.toml"
    )
    shrink_parser.add_argument("input")
    shrink_parser.add_argument(
        "--contains", default=None, help="preserve inputs containing this JSON string"
    )
    shrink_parser.add_argument(
        "--command",
        default=None,
        help="predicate command; candidate JSON is sent on stdin and non-zero preserves failure",
    )
    shrink_parser.add_argument(
        "--kind",
        choices=("auto", "rows", "columns", "tokens", "sequence", "tensor", "structured"),
        default="auto",
    )
    shrink_parser.add_argument("--output", default=None, help="write the linked child capsule")
    shrink_parser.add_argument("--json", action="store_true", dest="as_json")

    trace = sub.add_parser("trace", help="inspect bounded trace events in a capsule")
    trace.add_argument("--config", default=argparse.SUPPRESS, help="path to mlforensics.toml")
    trace.add_argument("incident")
    trace.add_argument("--radius", type=int, default=8)
    trace.add_argument("--json", action="store_true", dest="as_json")

    impact = sub.add_parser("impact", help="plan validation after a Git or file change")
    impact.add_argument("--config", default=argparse.SUPPRESS, help="path to mlforensics.toml")
    impact.add_argument("revision_range", nargs="?", default=None, help="BASE..HEAD")
    impact.add_argument("--base", default=None)
    impact.add_argument("--head", default="HEAD")
    impact.add_argument("--repo", default=".")
    impact.add_argument("--json", action="store_true", dest="as_json")

    parity = sub.add_parser("parity", help="compare callable, TorchScript, or ONNX models")
    parity.add_argument("--config", default=argparse.SUPPRESS, help="path to mlforensics.toml")
    parity.add_argument("baseline", help="module:object, pytorch:PATH, or onnx:PATH")
    parity.add_argument("candidate", help="module:object, pytorch:PATH, or onnx:PATH")
    parity.add_argument("--inputs", default=None, help="JSON list of representative inputs")
    parity.add_argument(
        "--input-spec", default=None, help="JSON input specification file or object"
    )
    parity.add_argument("--dtype", default=None, help="generated input dtype, e.g. float32")
    parity.add_argument("--shape", default=None, help="generated input shape, e.g. 1,3,224,224")
    parity.add_argument("--count", type=int, default=100)
    parity.add_argument("--seed", type=int, default=0)
    parity.add_argument("--edge-inputs", action="store_true")
    parity.add_argument("--shrink", action="store_true", help="shrink divergent inputs")
    parity.add_argument("--atol", type=float, default=1e-6)
    parity.add_argument("--rtol", type=float, default=1e-5)
    parity.add_argument("--json", action="store_true", dest="as_json")

    ci = sub.add_parser("ci", help="run a statistical comparison as a CI gate")
    ci.add_argument("--config", default=argparse.SUPPRESS, help="path to mlforensics.toml")
    ci.add_argument(
        "--baseline",
        required=True,
        action="append",
        help="baseline run capsule; repeat once per seed for a run-level claim",
    )
    ci.add_argument(
        "--candidate",
        required=True,
        action="append",
        help="candidate run capsule; repeat once per seed for a run-level claim",
    )
    ci.add_argument("--confidence", type=float, default=None)
    ci.add_argument("--resamples", type=int, default=2_000)
    ci.add_argument("--threshold", action="append", default=[], metavar="METRIC=VALUE")
    ci.add_argument("--json", action="store_true", dest="as_json")
    ci.add_argument(
        "--format",
        dest="output_format",
        choices=("text", "json", "junit", "sarif", "github"),
        default=None,
        help="CI report format (json is also available as --json)",
    )
    return parser


def _traceback_mentions_exception(text: str, error_type: str) -> bool:
    short = error_type.rsplit(".", 1)[-1]
    names = [error_type, short]
    if "." not in error_type or error_type.startswith("builtins."):
        names.append(f"builtins.{short}")
    haystack = text.replace("\r\n", "\n")
    for name in dict.fromkeys(names):
        if f"{name}:" in haystack or haystack.endswith(name) or f"\n{name}\n" in haystack:
            return True
    return False


def _scan_failure_envelopes(text: str) -> FailureSignature | None | UnresolvedEvaluation:
    unresolved = False
    found: FailureSignature | None = None
    for line in reversed(text.splitlines()):
        try:
            parsed = json.loads(line)
        except (TypeError, ValueError):
            continue
        parsed_failure = parse_failure_envelope(parsed)
        if isinstance(parsed_failure, UnresolvedEvaluation):
            unresolved = True
            continue
        if parsed_failure is not None:
            found = parsed_failure
            break
    if found is None and unresolved:
        return UnresolvedEvaluation("malformed failure envelope")
    return found


def _load_optional_child_result(path: Path) -> ExecutionResult | None:
    if not path.exists():
        return None
    try:
        return load_child_result(path)
    except UnresolvedEvaluation:
        raise
    except (OSError, TypeError, ValueError) as exc:
        raise UnresolvedEvaluation(f"malformed child result envelope: {exc}") from exc


def _signature_from_traceback(text: str) -> FailureSignature | None:
    lines = [line.strip() for line in text.replace("\r\n", "\n").splitlines() if line.strip()]
    if not lines:
        return None
    last = lines[-1]
    if ":" not in last:
        return None
    error_type, message = last.split(":", 1)
    error_type = error_type.strip()
    message = message.strip()
    if not error_type or " " in error_type:
        return None
    qualified = error_type if "." in error_type else f"builtins.{error_type}"
    return FailureSignature(
        error_type=qualified,
        message=message,
        normalized_message=normalize_failure_message(message),
        kind="exception",
        exception_chain=(qualified,),
    )


def _failure_from_completed(
    completed: subprocess.CompletedProcess[str],
    result_path: Path | None = None,
) -> FailureSignature | None | UnresolvedEvaluation:
    if result_path is not None:
        loaded = _load_optional_child_result(result_path)
        if loaded is not None:
            if loaded.unresolved:
                return UnresolvedEvaluation(loaded.error or "unresolved child result")
            if loaded.failure is not None:
                return loaded.failure
    combined = "\n".join((completed.stdout or "", completed.stderr or ""))
    scanned = _scan_failure_envelopes(combined)
    if isinstance(scanned, UnresolvedEvaluation) or scanned is not None:
        return scanned
    return _signature_from_traceback(combined)


def _parse_thresholds(values: Sequence[str]) -> dict[str, float]:
    result = {}
    for item in values:
        if "=" not in item:
            raise ValueError(f"threshold must have the form metric=value: {item}")
        name, value = item.split("=", 1)
        name = name.strip()
        if not name:
            raise ValueError(f"threshold metric name cannot be empty: {item}")
        try:
            parsed = float(value)
        except ValueError as exc:
            raise ValueError(f"threshold value must be numeric: {item}") from exc
        if not (parsed >= 0 and parsed < float("inf")):
            raise ValueError(f"threshold value must be finite and non-negative: {item}")
        result[name] = parsed
    return result


def _parse_named_paths(values: Sequence[str]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for item in values:
        if "=" in item:
            name, raw_path = item.split("=", 1)
        else:
            raw_path = item
            name = Path(item).name or "data"
        path = Path(raw_path)
        if not path.exists():
            raise FileNotFoundError(path)
        result[name] = path
    return result


def _config_mapping(config: Config, key: str) -> dict[str, Any]:
    value = config.ci.get(key, {})
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError(f"[ci].{key} must be a table")
    return dict(value)


def _config_bool(config: Config, key: str, default: bool = True) -> bool:
    value = config.ci.get(key, default)
    if not isinstance(value, bool):
        raise ValueError(f"[ci].{key} must be true or false")
    return value


def _config_names(config: Config, key: str) -> tuple[str, ...]:
    value = config.ci.get(key, ()) or ()
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"[ci].{key} must be a list of names")
    names = tuple(str(item) for item in value)
    if any(not name.strip() for name in names):
        raise ValueError(f"[ci].{key} must contain non-empty names")
    return names


def _usage_snapshot() -> Any | None:
    try:
        import resource

        return resource.getrusage(resource.RUSAGE_CHILDREN)
    except (ImportError, OSError):
        try:
            values = os.times()
            return SimpleNamespace(
                ru_utime=values.children_user,
                ru_stime=values.children_system,
                ru_maxrss=None,
            )
        except (AttributeError, OSError):
            return None


def _command_arguments(command: str) -> list[str]:
    """Parse a documented command string consistently on every platform."""
    if os.name == "nt":
        marker = "__MLFORENSICS_BACKSLASH__"
        while marker in command:
            marker += "_"
        values = [item.replace(marker, "\\") for item in shlex.split(command.replace("\\", marker))]
    else:
        values = shlex.split(command, posix=True)
    if not values:
        raise ValueError("command must not be empty")
    if os.name == "nt" and values == ["true"]:
        return [sys.executable, "-c", "pass"]
    if os.name == "nt" and values == ["false"]:
        return [sys.executable, "-c", "raise SystemExit(1)"]
    return values


def _redact_mapping(value: Any) -> Any:
    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            lowered = str(key).casefold()
            if any(token in lowered for token in ("password", "secret", "token", "api_key")):
                result[str(key)] = "<redacted>"
            else:
                result[str(key)] = _redact_mapping(item)
        return result
    if isinstance(value, (list, tuple)):
        return [_redact_mapping(item) for item in value]
    return value


def _merge_identified(
    parent_values: Sequence[Any], child_values: Sequence[Any], key
) -> tuple[Any, ...]:
    merged: list[Any] = []
    positions: dict[Any, int] = {}
    for item in (*parent_values, *child_values):
        identity = key(item)
        if identity in positions:
            merged[positions[identity]] = item
            continue
        positions[identity] = len(merged)
        merged.append(item)
    return tuple(merged)


def _merge_capsules(parent: RunCapsule, child: RunCapsule | None) -> RunCapsule:
    if child is None:
        return parent

    def named(item: Any) -> Any:
        return getattr(item, "name", None) or id(item)

    run = dataclasses.replace(
        parent.run,
        metadata={**dict(parent.run.metadata), **dict(child.run.metadata)},
        datasets=_merge_identified(parent.run.datasets, child.run.datasets, named),
        models=_merge_identified(parent.run.models, child.run.models, named),
        metrics=_merge_identified(parent.run.metrics, child.run.metrics, named),
        resources=tuple(parent.run.resources) + tuple(child.run.resources),
        events=tuple(child.run.events) + tuple(parent.run.events),
        rng_state=child.run.rng_state or parent.run.rng_state,
        failure_signature=child.run.failure_signature or parent.run.failure_signature,
        lineage_nodes=_merge_identified(parent.run.lineage_nodes, child.run.lineage_nodes, named),
        lineage_edges=_merge_identified(
            parent.run.lineage_edges,
            child.run.lineage_edges,
            lambda item: (
                getattr(item, "source", None),
                getattr(item, "target", None),
                getattr(item, "relation", None),
            ),
        ),
        observations=_merge_identified(
            parent.run.observations,
            child.run.observations,
            lambda item: (
                getattr(item, "name", None),
                getattr(item, "identity", None),
                getattr(item, "step", None),
            ),
        ),
        replay_plan=child.run.replay_plan or parent.run.replay_plan,
    )
    artifacts = {ref.sha256: ref for ref in (*parent.artifacts, *child.artifacts)}
    evidence = {**dict(parent.evidence), **dict(child.evidence)}
    for key in ("replay", "shrink", "parity", "behavior"):
        parent_value = parent.evidence.get(key)
        child_value = child.evidence.get(key)
        if isinstance(parent_value, Mapping) and isinstance(child_value, Mapping):
            evidence[key] = {**dict(parent_value), **dict(child_value)}
    return RunCapsule(
        run,
        tuple(artifacts.values()),
        {**dict(parent.payloads), **dict(child.payloads)},
        evidence,
    )


def _capsule_path(value: str, root: Path = Path(".mlforensics")) -> Path:
    resolved = resolve_run(root, value)
    if resolved is not None:
        return resolved
    direct = Path(value)
    if direct.exists():
        return direct
    options = [root / value, root / f"{value}.mlcap"]
    for candidate in options:
        if candidate.exists():
            return candidate
    searched = ", ".join(str(option) for option in (direct, *options))
    raise FileNotFoundError(
        f"no run capsule named {value!r}; looked for {searched}. "
        f"Pass a path to a .mlcap, or a run id recorded under {root}"
    )


def _print(value: Any, *, as_json: bool, formatter=render) -> None:
    if as_json:
        if hasattr(value, "to_json"):
            print(value.to_json())
        else:
            print(
                json.dumps(
                    value.to_dict() if hasattr(value, "to_dict") else value,
                    indent=2,
                    sort_keys=True,
                    default=str,
                )
            )
    else:
        rendered = formatter(value)
        print(rendered, end="" if str(rendered).endswith("\n") else "\n")


def _ci_report(value: Any, output_format: str) -> str:
    """Serialize a CI result for common automation consumers."""
    if output_format == "text":
        return value.summary()
    if output_format == "json":
        return value.to_json()

    checks = tuple(getattr(value, "checks", ()))
    if output_format == "junit":
        suite = ElementTree.Element(
            "testsuite",
            {
                "name": "mlforensics",
                "tests": str(len(checks)),
                "failures": str(sum(check.status == "fail" for check in checks)),
                "skipped": str(sum(check.status in {"warn", "skip"} for check in checks)),
                "errors": "0",
            },
        )
        for check in checks:
            case = ElementTree.SubElement(
                suite,
                "testcase",
                {"classname": "mlforensics", "name": check.name},
            )
            details = json.dumps(check.details, sort_keys=True, default=str)
            if check.status == "fail":
                failure = ElementTree.SubElement(
                    case,
                    "failure",
                    {"type": check.check_id or "check", "message": check.name},
                )
                failure.text = details
            elif check.status in {"warn", "skip"}:
                skipped = ElementTree.SubElement(case, "skipped", {"message": check.name})
                skipped.text = details
        return ElementTree.tostring(suite, encoding="unicode")

    if output_format == "sarif":
        rules = []
        results = []
        for check in checks:
            rule_id = check.check_id or "check"
            rules.append(
                {
                    "id": rule_id,
                    "name": check.name,
                    "shortDescription": {"text": check.name},
                }
            )
            level = {"fail": "error", "warn": "warning"}.get(check.status, "note")
            results.append(
                {
                    "ruleId": rule_id,
                    "level": level,
                    "message": {"text": f"{check.name}: {check.status}"},
                    "properties": {
                        "passed": check.passed,
                        "status": check.status,
                        "details": dict(check.details),
                    },
                }
            )
        return json.dumps(
            {
                "$schema": "https://json.schemastore.org/sarif-2.1.0.json",
                "version": "2.1.0",
                "runs": [
                    {
                        "tool": {"driver": {"name": "MLForensics", "rules": rules}},
                        "results": results,
                        "invocations": [
                            {
                                "executionSuccessful": bool(value.passed),
                                "exitCode": value.exit_code,
                            }
                        ],
                    }
                ],
            },
            indent=2,
            sort_keys=True,
            default=str,
        )

    if output_format == "github":
        annotations = []
        for check in checks:
            if check.status not in {"fail", "warn"}:
                continue
            annotations.append(
                {
                    "annotation_level": "failure" if check.status == "fail" else "warning",
                    "title": check.name,
                    "message": json.dumps(check.details, sort_keys=True, default=str),
                }
            )
        payload = {
            "name": "MLForensics",
            "status": "completed",
            "conclusion": "success" if value.passed else "failure",
            "output": {
                "title": "MLForensics CI gate",
                "summary": value.summary(),
                "text": "\n".join(f"{check.status}: {check.name}" for check in checks),
            },
            "annotations": annotations,
        }
        return json.dumps(payload, indent=2, sort_keys=True, default=str)

    raise ValueError(f"unsupported CI report format: {output_format}")


def _parse_shape(value: str | None) -> tuple[int, ...] | None:
    if value is None:
        return None
    try:
        shape = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise ValueError("shape must be comma-separated integers") from exc
    if not shape or any(item < 0 for item in shape):
        raise ValueError("shape must contain non-negative dimensions")
    return shape


def _parse_dtype(value: str | None) -> Any:
    if value is None:
        return None
    aliases = {
        "float": "float32",
        "double": "float64",
        "half": "float16",
        "long": "int64",
        "int": "int32",
    }
    return aliases.get(value.casefold(), value)


def _run(args: argparse.Namespace, config: Config) -> int:
    command = list(args.run_command)
    while command and command[0] == "--":
        command.pop(0)
    if not command:
        raise ValueError("run requires a command, for example: mlforensics run python train.py")
    output: Path | None = Path(args.output) if args.output else None
    datasets = _parse_named_paths(args.data)
    environment = capture_environment()
    for name in args.env:
        if name in os.environ:
            environment.setdefault("environment", {})[name] = os.environ[name]
            environment.setdefault("variables", {})[name] = os.environ[name]
    dataset_evidence = {
        name: fingerprint_dataset(path, name=name) for name, path in datasets.items()
    }
    context = CaptureContext(
        name="external command",
        metadata={
            "command": command,
            "git": capture_git(args.repo),
            "environment": environment,
            "hardware": capture_hardware(),
            "dependencies": capture_dependencies(),
            "configuration": {
                "path": str(config.path) if config.path else None,
                "storage": _redact_mapping(dict(config.storage)),
                "tracking": _redact_mapping(dict(config.tracking)),
            },
            "data_fingerprints": dataset_evidence,
        },
    )
    result: subprocess.CompletedProcess[str] | None = None
    launch_error: OSError | None = None
    child_capsule: RunCapsule | None = None
    with tempfile.TemporaryDirectory(prefix="mlforensics-child-") as temporary:
        child_path = Path(temporary) / "child.mlcap"
        child_env = os.environ.copy()
        child_env.update(
            {
                "MLFORENSICS_RUN_ID": context.run_id,
                "MLFORENSICS_CHILD_CAPSULE": str(child_path),
                "MLFORENSICS_PARENT_PID": str(os.getpid()),
            }
        )
        started = time.perf_counter()
        usage_before = _usage_snapshot()
        try:
            with context as active:
                for name, path_value in datasets.items():
                    active.add_dataset(str(path_value), name=name, metadata=dataset_evidence[name])
                try:
                    result = subprocess.run(
                        command,
                        cwd=args.repo,
                        text=True,
                        capture_output=True,
                        check=False,
                        env=child_env,
                    )
                except OSError as exc:
                    launch_error = exc
                    active.event(
                        "process_launch_error", message=str(exc), data={"command": command}
                    )
                    raise
                elapsed = time.perf_counter() - started
                usage_after = _usage_snapshot()
                active.record_resource("wall_time_s", elapsed, units="s")
                if usage_before is not None and usage_after is not None:
                    active.record_resource(
                        "cpu_time_s",
                        (usage_after.ru_utime - usage_before.ru_utime)
                        + (usage_after.ru_stime - usage_before.ru_stime),
                        units="s",
                    )
                    if usage_after.ru_maxrss is not None:
                        active.record_resource(
                            "peak_rss", float(usage_after.ru_maxrss), units="platform"
                        )
                active.event(
                    "process",
                    message=f"returncode={result.returncode}",
                    data={
                        "command": command,
                        "returncode": result.returncode,
                        "stdout": result.stdout[-100_000:],
                        "stderr": result.stderr[-100_000:],
                    },
                )
                if result.returncode:
                    raise subprocess.CalledProcessError(
                        result.returncode, command, result.stdout, result.stderr
                    )
        except (OSError, subprocess.CalledProcessError):
            pass
        if child_path.exists():
            try:
                child_capsule = RunCapsule.load(child_path)
            except (OSError, ValueError, RuntimeError):
                child_capsule = None
    path = output or (config.storage_root / f"{context.capsule.run.run_id}.mlcap")
    capsule = _merge_capsules(context.capsule, child_capsule)
    capsule.save(path)
    record_run(
        config.storage_root,
        capsule.run.run_id,
        path,
        status=capsule.run.status,
    )
    if not args.no_output and not args.as_json and result is not None:
        if result.stdout:
            print(result.stdout, end="")
        if result.stderr:
            print(result.stderr, file=sys.stderr, end="")
    print(
        json.dumps(
            {
                "capsule": str(path),
                "status": capsule.run.status,
                "returncode": result.returncode if result else (127 if launch_error else None),
                "child_instrumented": child_capsule is not None,
                "stdout": result.stdout[-100_000:] if result is not None else "",
                "stderr": result.stderr[-100_000:] if result is not None else "",
            },
            sort_keys=True,
        )
        if args.as_json
        else f"Capsule: {path}\nStatus: {capsule.run.status}"
    )
    return int(result.returncode if result else (127 if launch_error else 1))


def _comparison_sides(args: argparse.Namespace, config: Config) -> tuple[list[Path], list[Path]]:
    """Resolve each side of a comparison, which may hold several seeded runs."""

    def resolve(*groups: Any) -> list[Path]:
        names: list[Any] = []
        for group in groups:
            if group is None:
                continue
            names.extend(group if isinstance(group, list) else [group])
        return [_capsule_path(name, config.storage_root) for name in names]

    baseline = resolve(args.baseline, getattr(args, "baseline_run", None))
    candidate = resolve(args.candidate, getattr(args, "candidate_run", None))
    return baseline, candidate


def _compare(args: argparse.Namespace, config: Config, *, gate: bool = False) -> int:
    thresholds = dict(config.ci.get("thresholds", {}) or {})
    thresholds.update(_parse_thresholds(getattr(args, "threshold", ())))
    confidence = args.confidence if args.confidence is not None else config.confidence
    higher_is_better = _config_mapping(config, "higher_is_better")
    noninferiority_margins = _config_mapping(config, "noninferiority_margins")
    required_metrics = _config_names(config, "required_metrics")
    required_resources = _config_names(config, "required_resources")
    required_evidence = _config_names(config, "required_evidence")
    min_sample_count = int(config.ci.get("min_sample_count", 1))
    baseline_paths, candidate_paths = _comparison_sides(args, config)
    if gate:
        result = ci_gate(
            baseline_paths,
            candidate_paths,
            confidence=confidence,
            n_resamples=args.resamples,
            practical_thresholds=thresholds,
            higher_is_better=higher_is_better,
            noninferiority_margins=noninferiority_margins,
            min_slice_support=int(config.ci.get("min_slice_support", 5)),
            fail_on_regression=_config_bool(config, "fail_on_regression"),
            fail_on_failed_run=_config_bool(config, "fail_on_failed_run"),
            fail_on_nonfinite=_config_bool(config, "fail_on_nonfinite"),
            fail_on_missing_evidence=_config_bool(config, "fail_on_missing_evidence"),
            fail_on_behavior_regression=_config_bool(config, "fail_on_behavior_regression"),
            fail_on_parity_failure=_config_bool(config, "fail_on_parity_failure"),
            required_metrics=required_metrics,
            required_resources=required_resources,
            required_evidence=required_evidence,
            min_sample_count=min_sample_count,
        )
        output_format = getattr(args, "output_format", None)
        if output_format is None:
            output_format = "json" if args.as_json else "text"
        print(_ci_report(result, output_format))
        return result.exit_code
    comparison = compare_runs(
        baseline_paths,
        candidate_paths,
        confidence=confidence,
        n_resamples=args.resamples,
        practical_thresholds=thresholds,
        higher_is_better=higher_is_better,
        noninferiority_margins=noninferiority_margins,
        required_metrics=required_metrics,
        required_resources=required_resources,
        required_evidence=required_evidence,
        min_sample_count=min_sample_count,
    )
    if args.as_json:
        print(dumps(comparison))
    else:
        print(render_comparison(comparison))
    return 0


def _bisect(args: argparse.Namespace) -> int:
    git = SubprocessGit(args.repo)
    shell_command = args.command
    bytecode_cache = tempfile.TemporaryDirectory(prefix="mlforensics-bisect-")

    def runner(seed: int) -> Any:
        env = os.environ.copy()
        env["MLFORENSICS_SEED"] = str(seed)
        env.update(bytecode_isolation_env(git.revision, bytecode_cache.name))
        result = subprocess.run(
            shell_command,
            cwd=args.repo,
            shell=True,
            text=True,
            capture_output=True,
            env=env,
            check=False,
        )
        if result.returncode != 0:
            return {
                "error": f"command exited with status {result.returncode}",
                "stderr": result.stderr[-2_000:],
            }
        text = result.stdout.strip()
        if text:
            try:
                parsed = json.loads(text)
                if args.metric is not None:
                    if not isinstance(parsed, dict) or args.metric not in parsed:
                        return {"error": f"metric {args.metric!r} missing from command output"}
                    return parsed[args.metric]
                return parsed
            except json.JSONDecodeError:
                try:
                    return float(text.splitlines()[-1])
                except ValueError:
                    pass
        return True

    try:
        report = bisect_commits(
            git,
            runner,
            args.seeds or list(DEFAULT_BISECT_SEEDS),
            cache=BisectCache(path=args.cache) if args.cache else BisectCache(),
            tolerance=args.regression,
            confidence=args.confidence,
            higher_is_better=args.higher_is_better,
            n_resamples=args.resamples,
            max_runs=args.max_runs,
            min_observations=args.min_observations,
            good=args.good,
            bad=args.bad,
            command=shell_command,
            metric=args.metric,
            environment_fingerprint={
                key: os.environ.get(key)
                for key in ("PYTHONPATH", "CUDA_VISIBLE_DEVICES", "MLFORENSICS_SEED")
                if key in os.environ
            },
        )
    finally:
        bytecode_cache.cleanup()
    if args.as_json:
        print(
            json.dumps(
                {
                    "first_bad": report.first_bad,
                    "evaluations": [item.to_dict() for item in report.evaluations],
                    "inconclusive": report.inconclusive,
                    "metadata": report.metadata,
                },
                indent=2,
                sort_keys=True,
                default=str,
            )
        )
    else:
        print(f"First regression-compatible change: {report.first_bad or 'none found'}")
        for item in report.evaluations:
            if item.inconclusive:
                label = "INCONCLUSIVE"
            elif item.passed:
                label = "GOOD"
            else:
                label = "BAD"
            print(f"{label:<13} {item.target}")
        for item in report.evaluations:
            reason = item.metadata.get("reason")
            if item.inconclusive and reason:
                required = item.metadata.get("required_observations")
                detail = f" (needs {required} usable seeds)" if required else ""
                print(f"\n{item.target}: {reason}{detail}")
                break
    return 0 if report.first_bad is not None and not report.inconclusive else 2


def _replay(args: argparse.Namespace, config: Config) -> int:
    path = _capsule_path(args.incident, config.storage_root)
    capsule = RunCapsule.load(path)
    if not args.command:
        value = {
            "incident": str(path),
            "status": capsule.run.status,
            "failure": capsule.run.failure_signature.to_dict()
            if capsule.run.failure_signature
            else None,
            "message": "supply --command to execute a replay",
        }
        print(
            json.dumps(value, indent=2, sort_keys=True, default=str)
            if args.as_json
            else f"Incident: {path}\nStatus: {capsule.run.status}\nReplay command not supplied.\n"
        )
        return 0
    expected = capsule.run.failure_signature
    if expected is None:
        raise ValueError("the capsule is not a failed run and has no failure signature to replay")
    environment = os.environ.copy()
    environment["MLFORENSICS_REPLAY_CAPSULE"] = str(path.resolve())
    if args.step is not None:
        environment["MLFORENSICS_REPLAY_STEP"] = str(args.step)
    with tempfile.TemporaryDirectory(prefix="mlforensics-replay-") as temporary:
        child_path = Path(temporary) / "child.mlcap"
        result_path = Path(temporary) / "child.result.json"
        environment["MLFORENSICS_CHILD_CAPSULE"] = str(child_path)
        environment[CHILD_RESULT_ENV] = str(result_path)
        completed = subprocess.run(
            _command_arguments(args.command),
            shell=False,
            text=True,
            capture_output=True,
            check=False,
            env=environment,
        )
        child_capsule = None
        if child_path.exists():
            try:
                child_capsule = RunCapsule.load(child_path)
            except (OSError, ValueError, RuntimeError):
                child_capsule = None
        try:
            child_result = _load_optional_child_result(result_path)
        except UnresolvedEvaluation as exc:
            value = {
                "incident": str(path),
                "reproduced": False,
                "status": "inconclusive",
                "reason": str(exc),
                "returncode": completed.returncode,
                "stdout": completed.stdout[-100_000:],
                "stderr": completed.stderr[-100_000:],
            }
            print(
                json.dumps(value, indent=2, sort_keys=True, default=str)
                if args.as_json
                else str(exc)
            )
            return 2
    combined = "\n".join((completed.stdout, completed.stderr))
    normalized_output = normalize_failure_message(combined)
    capsule_failure = child_capsule.run.failure_signature if child_capsule is not None else None
    has_child_result = child_result is not None
    if has_child_result and child_result.unresolved:
        value = {
            "incident": str(path),
            "reproduced": False,
            "status": "inconclusive",
            "reason": child_result.error or "unresolved child result",
            "returncode": completed.returncode,
        }
        print(
            json.dumps(value, indent=2, sort_keys=True, default=str)
            if args.as_json
            else "unresolved"
        )
        return 2
    if has_child_result:
        structured_failure = child_result.failure
        if capsule_failure is not None:
            if structured_failure is None or not structured_failure.matches(capsule_failure):
                value = {
                    "incident": str(path),
                    "reproduced": False,
                    "status": "inconclusive",
                    "reason": "structured child result disagrees with child capsule failure",
                    "returncode": completed.returncode,
                    "signature_match": False,
                    "state_restoration_verified": False,
                }
                print(
                    json.dumps(value, indent=2, sort_keys=True, default=str)
                    if args.as_json
                    else value["reason"]
                )
                return 2
        has_structured = True
    elif capsule_failure is not None:
        structured_failure = capsule_failure
        has_structured = True
    else:
        scanned = _scan_failure_envelopes(combined)
        if isinstance(scanned, UnresolvedEvaluation):
            value = {
                "incident": str(path),
                "reproduced": False,
                "status": "inconclusive",
                "reason": str(scanned),
                "returncode": completed.returncode,
                "stdout": completed.stdout[-100_000:],
                "stderr": completed.stderr[-100_000:],
            }
            print(
                json.dumps(value, indent=2, sort_keys=True, default=str)
                if args.as_json
                else str(scanned)
            )
            return 2
        structured_failure = scanned
        has_structured = scanned is not None
    structured_match = structured_failure is not None and expected.matches(structured_failure)
    type_match = _traceback_mentions_exception(combined, expected.error_type)
    message_match = bool(expected.normalized_message) and (
        expected.normalized_message in normalized_output
    )
    original_exit = next(
        (
            event.data.get("returncode")
            for event in reversed(capsule.run.events)
            if event.kind == "process" and "returncode" in event.data
        ),
        None,
    )
    original_process_output = next(
        (
            "\n".join(
                (
                    str(event.data.get("stdout", "")),
                    str(event.data.get("stderr", "")),
                )
            )
            for event in reversed(capsule.run.events)
            if event.kind == "process"
        ),
        None,
    )
    exit_match = (
        completed.returncode != 0
        and original_exit is not None
        and completed.returncode == original_exit
    )
    log_match = type_match and message_match
    if has_structured:
        signature_match = structured_match
        reproduced = signature_match
    else:
        signature_match = log_match
        output_match = (
            exit_match
            and original_process_output is not None
            and normalize_failure_message(original_process_output) == normalized_output
        )
        reproduced = completed.returncode != 0 and (
            args.allow_any_failure or signature_match or output_match
        )
    output_match = (
        exit_match
        and original_process_output is not None
        and normalize_failure_message(original_process_output) == normalized_output
    )
    value = {
        "incident": str(path),
        "reproduced": reproduced,
        "returncode": completed.returncode,
        "expected_failure": expected.to_dict(),
        "signature_match": signature_match,
        "state_restoration_verified": False,
        "structured_failure": structured_failure.to_dict() if structured_failure else None,
        "exit_code_match": exit_match,
        "process_output_match": output_match,
        "restoration": "delegated to instrumented child via MLFORENSICS_REPLAY_CAPSULE",
        "stdout": completed.stdout[-100_000:],
        "stderr": completed.stderr[-100_000:],
    }
    print(
        json.dumps(value, indent=2, sort_keys=True, default=str)
        if args.as_json
        else (
            f"Replay: {'REPRODUCED' if reproduced else 'NOT REPRODUCED'}\n"
            f"Return code: {completed.returncode}\n"
            f"Signature match: {signature_match}\n"
            f"Process-output match: {output_match}"
        )
    )
    return 0 if reproduced else 1


def _shrink(args: argparse.Namespace, config: Config) -> int:
    try:
        source = _capsule_path(args.input, config.storage_root)
    except FileNotFoundError:
        source = Path(args.input)
    source_capsule: RunCapsule | None = None
    expected_failure: FailureSignature | None = None
    if source.exists() and (source.is_dir() or source.suffix in {".mlcap", ".zip"}):
        try:
            capsule = RunCapsule.load(source)
        except (OSError, ValueError, RuntimeError):
            capsule = None
        if capsule is not None:
            source_capsule = capsule
            expected_failure = capsule.run.failure_signature
            try:
                value = capsule_replay_input(capsule)
            except (KeyError, OSError, TypeError, ValueError, RuntimeError):
                value = next(
                    (
                        capsule.run.metadata[key]
                        for key in ("failure_input", "offending_batch", "last_batch")
                        if key in capsule.run.metadata
                    ),
                    None,
                )
                if value is None:
                    for ref in capsule.artifacts:
                        if any(token in ref.name.casefold() for token in ("batch", "input")):
                            payload = capsule.payloads.get(ref.sha256)
                            if payload is not None:
                                try:
                                    value = json.loads(payload)
                                    break
                                except (UnicodeDecodeError, json.JSONDecodeError):
                                    continue
            if value is None:
                raise ValueError("capsule does not contain a JSON failure input or batch")
        else:
            value = json.loads(source.read_text(encoding="utf-8"))
    elif source.exists():
        value = json.loads(source.read_text(encoding="utf-8"))
    else:
        # The argument doubles as a path and as inline JSON, so a path-like
        # argument that does not parse is almost always a missing file.
        try:
            value = json.loads(args.input)
        except json.JSONDecodeError as exc:
            raise FileNotFoundError(
                f"{args.input!r} is neither an existing file nor valid inline JSON ({exc})"
            ) from exc
    needle = args.contains
    predicate_kind = "command"
    predicate_guarantee = "structured failure signature preservation"
    temp_dir: tempfile.TemporaryDirectory[str] | None = None
    try:
        if args.command:
            temp_dir = tempfile.TemporaryDirectory(prefix="mlforensics-shrink-")
            temporary = temp_dir.name

            def run_candidate(
                candidate: Any,
            ) -> tuple[int, FailureSignature | None | UnresolvedEvaluation]:
                environment = os.environ.copy()
                result_path = Path(temporary) / "candidate.result.json"
                if result_path.exists():
                    result_path.unlink()
                environment[CHILD_RESULT_ENV] = str(result_path)
                if expected_failure is not None:
                    environment["MLFORENSICS_EXPECTED_FAILURE"] = json.dumps(
                        expected_failure.to_dict(), sort_keys=True
                    )
                completed = subprocess.run(
                    _command_arguments(args.command),
                    shell=False,
                    text=True,
                    input=json.dumps(candidate, sort_keys=True, default=str),
                    capture_output=True,
                    check=False,
                    env=environment,
                )
                return completed.returncode, _failure_from_completed(completed, result_path)

            origin_code, origin = run_candidate(value)
            if isinstance(origin, UnresolvedEvaluation):
                raise ValueError(str(origin))
            if expected_failure is None:
                if origin is None:
                    predicate_kind = "exit_code"
                    predicate_guarantee = (
                        "non-zero exit only; does not establish preservation of an incident failure"
                    )
                    if origin_code == 0:
                        raise ValueError("cannot shrink: the original command did not fail")
                else:
                    expected_failure = origin
                    predicate_kind = "failure_signature"

            def predicate(candidate: Any) -> bool:
                returncode, actual = run_candidate(candidate)
                if isinstance(actual, UnresolvedEvaluation):
                    return False
                if expected_failure is not None:
                    return actual is not None and expected_failure.matches(actual)
                return returncode != 0

        elif needle is not None:
            predicate_kind = "contains"
            predicate_guarantee = (
                "substring presence only; does not establish preservation of an incident failure"
            )

            def predicate(candidate: Any) -> bool:
                return needle in json.dumps(candidate, sort_keys=True, default=str)

        else:
            raise ValueError("shrink requires --command or --contains to preserve a real failure")
        result = shrink(value, predicate, kind=args.kind)
        result.metadata.setdefault("predicate_kind", predicate_kind)
        result.metadata.setdefault("predicate_guarantee", predicate_guarantee)
    finally:
        if temp_dir is not None:
            temp_dir.cleanup()
    child_path: str | None = None
    child_digest: str | None = None
    if source_capsule is not None:
        output = (
            Path(args.output) if args.output else source.with_name(source.stem + ".shrunk.mlcap")
        )
        child = persist_shrink_capsule(
            source_capsule,
            result,
            failure=expected_failure,
            output=output,
        )
        child_path = str(output)
        child_digest = child.digest
    output_value = result.to_dict()
    if child_path is not None:
        output_value["child_capsule"] = child_path
        output_value["child_capsule_digest"] = child_digest
    print(
        json.dumps(output_value, indent=2, sort_keys=True, default=str)
        if args.as_json
        else json.dumps(result.value, indent=2, sort_keys=True, default=str)
    )
    return 0


def _format_number(value: Any) -> str | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return f"{number:.6g}"


def _format_trace_event(event: TraceEvent) -> str:
    """Render one trace event with the statistics that were actually recorded."""
    data = event.data
    parts: list[str] = []
    step = event.step if event.step is not None else data.get("step")
    if step is not None:
        parts.append(f"step={step}")
    shape = data.get("shape")
    if shape:
        parts.append("shape=[" + ",".join(str(item) for item in shape) + "]")
    for name in ("dtype", "device"):
        if data.get(name):
            parts.append(f"{name}={data[name]}")
    low, high = _format_number(data.get("minimum")), _format_number(data.get("maximum"))
    if low is not None and high is not None:
        parts.append(f"range=[{low}, {high}]")
    for label, key in (("mean", "mean"), ("std", "std")):
        formatted = _format_number(data.get(key))
        if formatted is not None:
            parts.append(f"{label}={formatted}")
    finite_fraction = data.get("finite_fraction")
    if isinstance(finite_fraction, (int, float)) and finite_fraction < 1:
        parts.append(f"finite={finite_fraction:.1%}")
    if data.get("source"):
        parts.append(f"source={data['source']}")
    marker = " <- first non-finite value" if data.get("abnormal") else ""
    detail = ", ".join(parts) or (event.message or "")
    return f"- {event.kind}: {detail}{marker}"


def _trace(args: argparse.Namespace, config: Config) -> int:
    capsule = RunCapsule.load(_capsule_path(args.incident, config.storage_root))
    analysis = trace_incident(capsule)
    records = [TraceEvent.from_dict(value) for value in analysis["events"]]
    first_index = next(
        (
            index
            for index, event in enumerate(records)
            if event.data.get("abnormal")
            or (
                event.data.get("finite_fraction") is not None
                and event.data.get("finite_fraction") < 1
            )
        ),
        None,
    )
    window = (
        records[
            max(0, first_index - args.radius) : min(len(records), first_index + args.radius + 1)
        ]
        if first_index is not None
        else records
    )
    causal_path = list(analysis["causal_path"])
    ancestry = causal_path[:-1] if first_index is not None and causal_path else causal_path
    value = {
        "run_id": capsule.run.run_id,
        "event_count": len(records),
        "first_abnormal": records[first_index].to_dict() if first_index is not None else None,
        "ancestry": ancestry,
        "causal_path": causal_path,
        "root_candidates": analysis["root_candidates"],
        "missing_parents": analysis["missing_parents"],
        "window": [event.to_dict() for event in window],
    }
    print(
        json.dumps(value, indent=2, sort_keys=True, default=str)
        if args.as_json
        else (
            f"Trace: {len(records)} event(s)\n"
            + (
                "No non-finite tensor event recorded."
                if first_index is None
                else "First abnormal operation: "
                + records[first_index].kind
                + (
                    f" ({records[first_index].data.get('source')})"
                    if records[first_index].data.get("source")
                    else ""
                )
            )
            + "\n"
            + "\n".join(_format_trace_event(event) for event in window)
        )
    )
    return 0


def _impact(args: argparse.Namespace, config: Config) -> int:
    mapping: dict[str, Any] = dict(config.impact)
    if args.revision_range or args.base:
        base, head = (
            args.revision_range.split("..", 1) if args.revision_range else (args.base, args.head)
        )
        report = impact_from_git(args.repo, base, head, mappings=mapping)
    else:
        report = analyze_impact(args.repo, [], mappings=mapping)
    _print(report, as_json=args.as_json, formatter=render)
    return 0


def _parity(args: argparse.Namespace) -> int:
    input_spec = None
    if args.input_spec is not None:
        input_spec = InputSpec.from_json(
            Path(args.input_spec).read_text(encoding="utf-8")
            if Path(args.input_spec).exists()
            else args.input_spec
        )
    if args.inputs is not None:
        inputs = json.loads(
            Path(args.inputs).read_text(encoding="utf-8")
            if Path(args.inputs).exists()
            else args.inputs
        )
        if isinstance(inputs, Mapping):
            if "cases" in inputs:
                inputs = inputs["cases"]
            elif "inputs" in inputs and isinstance(inputs["inputs"], list):
                inputs = inputs["inputs"]
            else:
                inputs = [inputs]
        if not isinstance(inputs, list):
            raise ValueError("inputs must be a JSON list of cases or a named input mapping")
    else:
        if args.count < 1:
            raise ValueError("count must be positive")
        inputs = generate_input_cases(
            shape=_parse_shape(args.shape),
            dtype=_parse_dtype(args.dtype) or "float32",
            seed=args.seed,
            representative_count=args.count,
            include_edge=args.edge_inputs,
            include_adversarial=args.edge_inputs,
            include_non_finite=False,
            input_spec=input_spec,
        )
    report = compare_models(
        args.baseline,
        args.candidate,
        inputs,
        atol=args.atol,
        rtol=args.rtol,
        shrink_failures=args.shrink,
        input_shape=_parse_shape(args.shape),
        input_dtype=_parse_dtype(args.dtype),
        input_spec=input_spec,
    )
    print(report.to_json(indent=2) if args.as_json else report.report())
    return report.exit_code


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        config = load_config(getattr(args, "config", None))
        if args.subcommand == "run":
            return _run(args, config)
        if args.subcommand == "compare":
            return _compare(args, config)
        if args.subcommand == "ci":
            return _compare(args, config, gate=True)
        if args.subcommand == "bisect":
            return _bisect(args)
        if args.subcommand == "replay":
            return _replay(args, config)
        if args.subcommand == "shrink":
            return _shrink(args, config)
        if args.subcommand == "trace":
            return _trace(args, config)
        if args.subcommand == "impact":
            return _impact(args, config)
        if args.subcommand == "parity":
            return _parity(args)
    except (OSError, ValueError, RuntimeError, TypeError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
