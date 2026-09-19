"""Uncertainty-aware Git and configuration bisection."""

from __future__ import annotations

import hashlib
import inspect
import json
import math
import os
import random
import shutil
import statistics
import subprocess
import time
import uuid
from collections.abc import Callable, Mapping, MutableMapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from ..analysis.statistics import paired_bootstrap, paired_differences
from ..core import FailureSignature
from ..core.contracts import evaluate_predicate

DEFAULT_BISECT_SEEDS = (11, 29, 37, 53, 71)
MIN_STOCHASTIC_OBSERVATIONS = 5


def _json_number(value: float | None) -> float | None:
    if value is None:
        return None
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


class SubprocessGit:
    """Minimal Git adapter used only when a caller explicitly requests Git."""

    def __init__(self, repo: str = ".") -> None:
        self.repo = str(Path(repo).resolve())
        self.revision: str | None = None
        self.working_directory = self.repo

    def preflight(self) -> None:
        """Reject a dirty checkout before a bisect can disturb user work."""
        result = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=all"],
            cwd=self.repo,
            text=True,
            capture_output=True,
            check=True,
        )
        if result.stdout.strip():
            raise RuntimeError(
                "bisect requires a clean checkout; use WorktreeGit to preserve local changes"
            )

    def commits(self, good: str | None = None, bad: str = "HEAD") -> list[str]:
        revisions = ["git", "rev-list", "--reverse"]
        if good is not None:
            revisions.extend(["--ancestry-path", f"{good}..{bad}"])
        else:
            revisions.append(bad)
        result = subprocess.run(
            revisions, cwd=self.repo, text=True, capture_output=True, check=True
        )
        values = [line for line in result.stdout.splitlines() if line]
        return [good, *values] if good is not None else values

    def checkout(self, revision: str) -> None:
        subprocess.run(
            ["git", "checkout", "--quiet", revision],
            cwd=self.repo,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            check=True,
            text=True,
        )
        self.revision = revision
        self.working_directory = self.repo

    def current_revision(self) -> str:
        """Return a restorable branch name, or the detached commit hash."""
        branch = subprocess.run(
            ["git", "branch", "--show-current"],
            cwd=self.repo,
            text=True,
            capture_output=True,
            check=True,
        ).stdout.strip()
        if branch:
            return branch
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=self.repo,
            text=True,
            capture_output=True,
            check=True,
        ).stdout.strip()

    def restore(self, revision: str) -> None:
        self.checkout(revision)


class WorktreeGit:
    """Evaluate revisions in isolated Git worktrees instead of the main checkout."""

    def __init__(self, repo: str = ".", *, root: str | os.PathLike[str] | None = None) -> None:
        self.repo = str(Path(repo).resolve())
        self.revision: str | None = None
        self.working_directory: str = self.repo
        self._parent_root = Path(root).resolve() if root is not None else None
        self._root: Path | None = None
        self._worktrees: dict[str, Path] = {}
        self._owns_root = True

    def preflight(self) -> None:
        """Validate the source repository without inspecting its dirty state."""
        subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=self.repo,
            text=True,
            capture_output=True,
            check=True,
        )

    def commits(self, good: str | None = None, bad: str = "HEAD") -> list[str]:
        return SubprocessGit(self.repo).commits(good, bad)

    def checkout(self, revision: str) -> None:
        if revision in self._worktrees:
            self.working_directory = str(self._worktrees[revision])
            self.revision = revision
            return
        if self._root is None:
            import tempfile

            if self._parent_root is None:
                self._root = Path(tempfile.mkdtemp(prefix="mlforensics-worktrees-"))
            else:
                self._parent_root.mkdir(parents=True, exist_ok=True)
                self._root = Path(tempfile.mkdtemp(prefix="bisect-", dir=str(self._parent_root)))
        token = hashlib.sha256(revision.encode("utf-8")).hexdigest()[:20]
        path = self._root / token
        if path.exists():
            # A path from an interrupted run is not evidence of a valid
            # worktree.  Never attach to it or remove it implicitly.
            raise RuntimeError(f"worktree path already exists: {path}")
        subprocess.run(
            ["git", "worktree", "add", "--detach", str(path), revision],
            cwd=self.repo,
            check=True,
            capture_output=True,
            text=True,
        )
        self._worktrees[revision] = path
        self.working_directory = str(path)
        self.revision = revision

    def current_revision(self) -> str:
        return SubprocessGit(self.repo).current_revision()

    def restore(self, revision: str) -> None:
        self.revision = revision
        # The source checkout was never changed; restoration means returning
        # the runner context to it, not selecting an old evaluation worktree.
        self.working_directory = self.repo

    def cleanup(self) -> None:
        for path in self._worktrees.values():
            subprocess.run(
                ["git", "worktree", "remove", "--force", str(path)],
                cwd=self.repo,
                check=False,
                capture_output=True,
                text=True,
            )
        self._worktrees.clear()
        self.working_directory = self.repo
        if self._owns_root and self._root is not None:
            shutil.rmtree(self._root, ignore_errors=True)
        self._root = None

    close = cleanup

    def __enter__(self) -> WorktreeGit:
        self.preflight()
        return self

    def __exit__(self, *_: Any) -> None:
        self.cleanup()


def bytecode_isolation_env(revision: str | None, root: str | os.PathLike[str]) -> dict[str, str]:
    """Return environment overrides that stop bytecode leaking across revisions.

    CPython validates a cached ``.pyc`` against its source's size and an mtime
    truncated to whole seconds. A bisect checks out several revisions per second,
    so a module whose size is unchanged across revisions is silently treated as
    current and every later revision is then measured using the earlier
    revision's code. Giving each revision its own cache prefix keeps caching
    effective while making that impossible, and it bypasses any stale in-tree
    ``__pycache__`` without modifying the caller's working tree.
    """
    token = hashlib.sha256((revision or "working-tree").encode("utf-8")).hexdigest()[:16]
    return {"PYTHONPYCACHEPREFIX": str(Path(root) / token)}


@dataclass(slots=True)
class RunOutcome:
    target: str
    passed: bool
    score: float | None = None
    samples: list[float] = field(default_factory=list)
    confidence_interval: tuple[float, float] | None = None
    runs: int = 1
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def inconclusive(self) -> bool:
        return bool(self.metadata.get("inconclusive", False))

    def to_dict(self) -> dict[str, Any]:
        classification = "inconclusive" if self.inconclusive else "good" if self.passed else "bad"
        return {
            "target": self.target,
            "passed": self.passed,
            "inconclusive": self.inconclusive,
            "classification": classification,
            "score": _json_number(self.score),
            "samples": [_json_number(item) for item in self.samples],
            "confidence_interval": (
                [_json_number(item) for item in self.confidence_interval]
                if self.confidence_interval is not None
                else None
            ),
            "runs": self.runs,
            "metadata": dict(self.metadata),
        }


@dataclass(slots=True)
class BisectResult:
    first_bad: str | None
    good: str
    bad: str
    evaluations: list[RunOutcome] = field(default_factory=list)
    candidate_count: int = 0
    inconclusive: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_conclusive(self) -> bool:
        return self.first_bad is not None and not self.inconclusive

    def to_dict(self) -> dict[str, Any]:
        return {
            "first_bad": self.first_bad,
            "good": self.good,
            "bad": self.bad,
            "evaluations": [item.to_dict() for item in self.evaluations],
            "candidate_count": self.candidate_count,
            "inconclusive": list(self.inconclusive),
            "metadata": dict(self.metadata),
        }


def _invoke(runner: Callable[..., Any], target: str, seed: int) -> Any:
    try:
        parameters = inspect.signature(runner).parameters
    except (TypeError, ValueError):
        return runner(target, seed)
    seed_parameter = parameters.get("seed")
    if seed_parameter is not None and seed_parameter.kind is inspect.Parameter.KEYWORD_ONLY:
        return runner(target, seed=seed)
    positional = [
        parameter
        for parameter in parameters.values()
        if parameter.kind
        in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    ]
    if (
        any(parameter.kind is inspect.Parameter.VAR_POSITIONAL for parameter in parameters.values())
        or len(positional) >= 2
    ):
        return runner(target, seed)
    return runner(target)


def _invoke_bisect_runner(runner: Callable[..., Any], seed: int, git: Any) -> Any:
    """Call a Git runner, optionally supplying an isolated worktree path."""
    working_directory = getattr(git, "working_directory", None)
    if working_directory is None:
        return runner(seed)
    try:
        parameters = inspect.signature(runner).parameters
    except (TypeError, ValueError):
        return runner(seed)
    if "working_directory" in parameters:
        return runner(seed, working_directory=working_directory)
    if "cwd" in parameters:
        return runner(seed, cwd=working_directory)
    return runner(seed)


def _to_score(value: Any) -> tuple[bool | None, float | None, dict[str, Any]]:
    if isinstance(value, Mapping):
        passed = value.get("passed", value.get("good"))
        raw_score = value.get("score", value.get("metric", value.get("value")))
        try:
            score = float(raw_score) if raw_score is not None else None
        except (TypeError, ValueError):
            score = None
        return (bool(passed) if passed is not None else None, score, dict(value))
    if isinstance(value, bool):
        return value, None, {}
    if isinstance(value, (int, float)):
        return None, float(value), {}
    return None, None, {"result": repr(value)}


def _canonical_identity(value: Any) -> str:
    """Serialize an execution identity so cache entries cannot cross runs."""
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError):
        return repr(value)


def _callable_identity(value: Any) -> str | None:
    """Return a conservative identity for a user supplied executable."""
    if value is None:
        return None
    target = getattr(value, "__func__", value)
    code = getattr(target, "__code__", None)
    payload: dict[str, Any] = {
        "module": getattr(target, "__module__", type(target).__module__),
        "qualname": getattr(target, "__qualname__", type(target).__qualname__),
    }
    if code is not None:
        payload["code"] = hashlib.sha256(code.co_code).hexdigest()
        payload["consts"] = repr(code.co_consts)
        closure = getattr(target, "__closure__", None)
        if closure:
            payload["closure"] = [repr(cell.cell_contents) for cell in closure]
    else:
        payload["repr"] = repr(target)
    return _canonical_identity(payload)


def _decode_cache_value(value: Any) -> Any:
    """Decode values written by :class:`BisectCache` without trusting reprs."""
    if isinstance(value, Mapping):
        marker = value.get("__mlforensics_cache_type__")
        if marker == "failure_signature":
            candidate = value.get("value")
            if not isinstance(candidate, Mapping):
                return value
            try:
                return FailureSignature.from_dict(candidate)
            except (TypeError, ValueError):
                return value
        if marker == "unserializable":
            # The original object cannot be reconstructed.  Treating its
            # repr as evidence would make a resume silently change meaning.
            return value
        return {key: _decode_cache_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_decode_cache_value(item) for item in value]
    return value


def _cache_value_is_reconstructable(value: Any) -> bool:
    if isinstance(value, Mapping):
        if value.get("__mlforensics_cache_type__") == "unserializable":
            return False
        return all(_cache_value_is_reconstructable(item) for item in value.values())
    if isinstance(value, list):
        return all(_cache_value_is_reconstructable(item) for item in value)
    return True


def _budget_limit(budget: Any, name: str) -> float | int | None:
    if budget is None:
        return None
    value = budget.get(name) if isinstance(budget, Mapping) else getattr(budget, name, None)
    return value


def _reported_resource(value: Any, names: set[str]) -> float:
    if not isinstance(value, Mapping):
        return 0.0
    total = 0.0
    for key, item in value.items():
        if str(key) in names:
            try:
                number = float(item)
            except (TypeError, ValueError):
                continue
            if math.isfinite(number) and number >= 0:
                total += number
        elif str(key) in {"metadata", "resources"}:
            total += _reported_resource(item, names)
    return total


def _predicate_outcome(
    revision: str,
    values: Sequence[Any],
    predicate: Callable[[Any], Any],
    *,
    metadata: Mapping[str, Any] | None = None,
) -> RunOutcome:
    extra = dict(metadata or {})
    try:
        judged = [
            evaluate_predicate(predicate, _failure_from_value(value) or value) for value in values
        ]
    except Exception as exc:
        return RunOutcome(
            revision,
            True,
            runs=len(values),
            metadata={
                **extra,
                "inconclusive": True,
                "predicate_error": f"{type(exc).__name__}: {exc}",
            },
        )
    if any(item.unresolved for item in judged):
        return RunOutcome(
            revision,
            True,
            runs=len(judged),
            metadata={**extra, "inconclusive": True, "reason": "unresolved predicate"},
        )
    failures = [item.preserved for item in judged]
    failure_count = sum(bool(item) for item in failures)
    inconclusive = not failures or 0 < failure_count < len(failures)
    return RunOutcome(
        revision,
        failure_count == 0,
        runs=len(failures),
        metadata={
            **extra,
            "failure_votes": failures,
            "failure_count": failure_count,
            "inconclusive": inconclusive,
            "predicate": True,
            "reasons": [item.reason for item in judged if item.reason],
        },
    )


def _failure_from_value(value: Any) -> FailureSignature | None:
    if isinstance(value, FailureSignature):
        return value
    if isinstance(value, Mapping):
        candidate = value.get("failure", value.get("failure_signature"))
        if candidate is None and value.get("type") == "failure_signature":
            candidate = value
        if isinstance(candidate, FailureSignature):
            return candidate
        if isinstance(candidate, Mapping):
            try:
                return FailureSignature.from_dict(candidate)
            except (TypeError, ValueError):
                return None
    return None


def _stable_pairs(
    baseline: Mapping[Any, Any], candidate: Mapping[Any, Any]
) -> tuple[list[Any], list[Any], list[Any]]:
    """Return matched values and identities without shifting after failures."""
    identities = [identity for identity in baseline if identity in candidate]
    return (
        [baseline[identity] for identity in identities],
        [candidate[identity] for identity in identities],
        identities,
    )


def _quantile(values: Sequence[float], probability: float) -> float:
    position = (len(values) - 1) * probability
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return values[lower]
    return values[lower] + (values[upper] - values[lower]) * (position - lower)


def _bootstrap_interval(
    values: Sequence[float], confidence: float, seed: int = 0
) -> tuple[float, float]:
    if not 0 < confidence < 1:
        raise ValueError("confidence must be between 0 and 1")
    if len(values) < 2:
        return (values[0], values[0]) if values else (float("nan"), float("nan"))
    rng = random.Random(seed)
    means = [
        statistics.fmean(values[rng.randrange(len(values))] for _ in values)
        for _ in range(max(200, 100 * len(values)))
    ]
    means.sort()
    tail = (1.0 - confidence) / 2.0
    return _quantile(means, tail), _quantile(means, 1.0 - tail)


class StochasticBisector:
    """Binary search a monotonic good-to-bad revision sequence."""

    def __init__(
        self,
        good: str,
        bad: str,
        runner: Callable[..., Any],
        *,
        seeds: Sequence[int] = DEFAULT_BISECT_SEEDS,
        regression_threshold: float = 0.0,
        higher_is_better: bool = True,
        confidence: float = 0.95,
        max_runs_per_target: int | None = None,
        cache: MutableMapping[tuple[str, int], Any] | Mapping[tuple[str, int], Any] | None = None,
        cache_identity: Mapping[str, Any] | str | None = None,
        failure_predicate: Callable[[Any], bool] | None = None,
        predicate: Callable[[Any], bool] | None = None,
        deterministic: bool = False,
        min_observations: int | None = None,
    ) -> None:
        if not seeds:
            raise ValueError("at least one seed is required")
        if max_runs_per_target is not None and max_runs_per_target < 1:
            raise ValueError("max_runs_per_target must be positive")
        if not 0 < confidence < 1:
            raise ValueError("confidence must be between 0 and 1")
        self.good, self.bad, self.runner = good, bad, runner
        self.seeds = [int(seed) for seed in seeds]
        if len(set(self.seeds)) != len(self.seeds):
            raise ValueError("seeds must be unique so observations can be paired safely")
        self.regression_threshold = abs(float(regression_threshold))
        self.higher_is_better = higher_is_better
        self.confidence = confidence
        self.max_runs_per_target = max_runs_per_target or len(self.seeds)
        self.cache = cache if cache is not None else {}
        self.deterministic = bool(deterministic)
        self.failure_predicate = failure_predicate or predicate
        if self.deterministic or self.failure_predicate is not None:
            default_min = 1
        else:
            default_min = MIN_STOCHASTIC_OBSERVATIONS
        self.min_observations = default_min if min_observations is None else int(min_observations)
        if isinstance(self.min_observations, bool) or self.min_observations < 1:
            raise ValueError("min_observations must be a positive integer")
        default_identity = {
            "schema_version": 2,
            "good": good,
            "bad": bad,
            "seeds": self.seeds,
            "regression_threshold": self.regression_threshold,
            "higher_is_better": bool(higher_is_better),
            "confidence": float(confidence),
            "max_runs_per_target": self.max_runs_per_target,
            "deterministic": self.deterministic,
            "failure_predicate": _callable_identity(self.failure_predicate),
        }
        persistent_cache = (
            isinstance(cache, BisectCache) and getattr(cache, "path", None) is not None
        )
        if isinstance(cache_identity, Mapping):
            default_identity.update(dict(cache_identity))
            cache_identity = default_identity
        elif cache_identity is not None:
            cache_identity = {**default_identity, "user_identity": cache_identity}
        elif cache_identity is None:
            if persistent_cache:
                # Arbitrary Python runners cannot be serialized into a stable
                # identity.  Persistent reuse requires an explicit identity.
                default_identity = {
                    **default_identity,
                    "cache_reuse": "disabled",
                    "search_id": uuid.uuid4().hex,
                }
            cache_identity = default_identity
        self.cache_identity = _canonical_identity(cache_identity)
        self.evaluations: list[RunOutcome] = []
        self._good_samples: dict[int, float] | None = None

    def _cache_get(self, target: str, seed: int) -> tuple[bool, Any]:
        key: tuple[Any, ...] = (target, seed)
        # Preserve the historical tuple key for an un-namespaced mapping, but
        # isolate callers that supplied an execution identity.  BisectCache
        # has a first-class namespace; plain dictionaries need an equivalent
        # key-level guard to avoid reusing evidence from another command.
        if self.cache_identity != "{}":
            key = (self.cache_identity, target, seed)
        if isinstance(self.cache, BisectCache):
            sentinel = object()
            value = self.cache.get(target, seed, sentinel, identity=self.cache_identity)
            return value is not sentinel, value
        if key in self.cache:
            return True, self.cache[key]
        return False, None

    def _cache_set(self, target: str, seed: int, value: Any) -> None:
        if isinstance(self.cache, BisectCache):
            self.cache.set(target, seed, value, identity=self.cache_identity)
            return
        try:
            key: tuple[Any, ...] = (target, seed)
            if self.cache_identity != "{}":
                key = (self.cache_identity, target, seed)
            self.cache[key] = value  # type: ignore[index]
        except TypeError:
            pass

    def evaluate(self, target: str) -> RunOutcome:
        observations: dict[int, Any] = {}
        samples: list[float] = []
        sample_ids: list[int] = []
        bool_results: list[bool] = []
        metadata: dict[str, Any] = {}
        for seed in self.seeds[: self.max_runs_per_target]:
            hit, value = self._cache_get(target, seed)
            if hit:
                metadata["cache_hits"] = metadata.get("cache_hits", 0) + 1
            else:
                try:
                    value = _invoke(self.runner, target, seed)
                except Exception as exc:
                    value = FailureSignature.from_exception(exc)
                self._cache_set(target, seed, value)
            observations[seed] = value
            passed, score, details = _to_score(value)
            metadata.update(
                {
                    key: item
                    for key, item in details.items()
                    if key not in {"score", "metric", "value", "passed", "good"}
                }
            )
            if passed is not None:
                bool_results.append(passed)
            elif score is not None and math.isfinite(score):
                samples.append(score)
                sample_ids.append(seed)
        metadata["observation_ids"] = list(observations)
        metadata["missing_observation_ids"] = [
            seed for seed in self.seeds[: self.max_runs_per_target] if seed not in observations
        ]
        metadata["nonfinite_observation_ids"] = [
            seed
            for seed, value in observations.items()
            if _to_score(value)[1] is not None and not math.isfinite(float(_to_score(value)[1]))
        ]
        if self.failure_predicate is not None:
            outcome = _predicate_outcome(
                target, list(observations.values()), self.failure_predicate, metadata=metadata
            )
            self.evaluations.append(outcome)
            return outcome
        if bool_results and not samples:
            passed_count = sum(bool_results)
            passed = passed_count > len(bool_results) / 2
            inconclusive = len(bool_results) % 2 == 0 and passed_count == len(bool_results) / 2
            outcome = RunOutcome(
                target,
                passed,
                runs=len(bool_results),
                metadata={**metadata, "votes": bool_results, "inconclusive": inconclusive},
            )
        elif samples:
            if len(samples) < self.min_observations:
                outcome = RunOutcome(
                    target,
                    True,
                    samples=samples,
                    runs=len(samples),
                    metadata={
                        **metadata,
                        "inconclusive": True,
                        "reason": "insufficient matched observations",
                        "required_observations": self.min_observations,
                    },
                )
                self.evaluations.append(outcome)
                return outcome
            score = statistics.fmean(samples)
            if target == self.good or self._good_samples is None:
                self._good_samples = {
                    seed: float(_to_score(observations[seed])[1])
                    for seed in sample_ids
                    if _to_score(observations[seed])[1] is not None
                }
                passed, interval = True, None
            else:
                candidate_samples = {
                    seed: float(_to_score(observations[seed])[1])
                    for seed in sample_ids
                    if _to_score(observations[seed])[1] is not None
                }
                baseline_values, candidate_values, paired_ids = _stable_pairs(
                    self._good_samples, candidate_samples
                )
                metadata["paired_observation_ids"] = list(paired_ids)
                metadata["unpaired_baseline_ids"] = sorted(
                    set(self._good_samples).difference(candidate_samples)
                )
                metadata["unpaired_candidate_ids"] = sorted(
                    set(candidate_samples).difference(self._good_samples)
                )
                differences = paired_differences(
                    baseline_values,
                    candidate_values,
                    baseline_ids=paired_ids,
                    candidate_ids=paired_ids,
                )
                if not differences:
                    outcome = RunOutcome(
                        target,
                        True,
                        score=score,
                        samples=samples,
                        runs=len(samples),
                        metadata={**metadata, "inconclusive": True, "reason": "no shared seeds"},
                    )
                    self.evaluations.append(outcome)
                    return outcome
                if len(paired_ids) < self.min_observations:
                    outcome = RunOutcome(
                        target,
                        True,
                        score=score,
                        samples=samples,
                        runs=len(samples),
                        metadata={
                            **metadata,
                            "inconclusive": True,
                            "reason": "insufficient matched observations",
                            "required_observations": self.min_observations,
                            "paired_observation_ids": list(paired_ids),
                        },
                    )
                    self.evaluations.append(outcome)
                    return outcome
                interval = _bootstrap_interval(differences, self.confidence)
                harmful_interval = (
                    (-interval[1], -interval[0]) if self.higher_is_better else interval
                )
                passed = not (harmful_interval[0] > self.regression_threshold)
            outcome = RunOutcome(
                target,
                passed,
                score=score,
                samples=samples,
                confidence_interval=interval,
                runs=len(samples),
                metadata=metadata,
            )
        else:
            outcome = RunOutcome(target, True, runs=0, metadata={**metadata, "inconclusive": True})
        self.evaluations.append(outcome)
        return outcome

    def run(self, commits: Sequence[str] | None = None) -> BisectResult:
        sequence = list(commits or [self.good, self.bad])
        if len(sequence) < 2 or sequence[0] != self.good or sequence[-1] != self.bad:
            raise ValueError("commits must start with good and end with bad")
        good_outcome = self.evaluate(self.good)
        if good_outcome.inconclusive or not good_outcome.passed:
            return BisectResult(
                None,
                self.good,
                self.bad,
                self.evaluations,
                len(sequence),
                [self.good],
                {"reason": "good revision is not a healthy, conclusive endpoint"},
            )
        bad_outcome = self.evaluate(self.bad)
        if bad_outcome.inconclusive or bad_outcome.passed:
            return BisectResult(
                None,
                self.good,
                self.bad,
                self.evaluations,
                len(sequence),
                [self.bad],
                {
                    "reason": (
                        "bad revision is inconclusive"
                        if bad_outcome.inconclusive
                        else "bad revision did not reproduce the regression"
                    )
                },
            )
        low, high = 0, len(sequence) - 1
        inconclusive: list[str] = []
        while high - low > 1:
            middle = (low + high) // 2
            outcome = self.evaluate(sequence[middle])
            if outcome.inconclusive:
                inconclusive.append(sequence[middle])
                return BisectResult(
                    None,
                    self.good,
                    self.bad,
                    self.evaluations,
                    len(sequence),
                    inconclusive,
                    {"reason": "an intermediate revision is inconclusive"},
                )
            elif outcome.passed:
                low = middle
            else:
                high = middle
        return BisectResult(
            sequence[high], self.good, self.bad, self.evaluations, len(sequence), inconclusive
        )


def bisect(
    good: str,
    bad: str,
    runner: Callable[..., Any],
    *,
    commits: Sequence[str] | None = None,
    **kwargs: Any,
) -> BisectResult:
    return StochasticBisector(good, bad, runner, **kwargs).run(commits)


def git_bisect(
    good: str, bad: str, runner: Callable[..., Any], *, path: str | None = None, **kwargs: Any
) -> BisectResult:
    from ..capture.git import git_commits

    return bisect(good, bad, runner, commits=git_commits(good, bad, path), **kwargs)


class RegressionStatus(str, Enum):
    REGRESSION = "regression"
    NO_REGRESSION = "no_regression"
    INCONCLUSIVE = "inconclusive"


@dataclass(frozen=True)
class RegressionDecision:
    status: RegressionStatus
    lower: float
    upper: float
    estimate: float
    tolerance: float = 0.0
    confidence: float = 0.95
    samples: int = 0
    higher_is_better: bool = True

    @property
    def regression(self) -> bool:
        return self.status is RegressionStatus.REGRESSION

    @property
    def passed(self) -> bool:
        return self.status is RegressionStatus.NO_REGRESSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status.value,
            "lower": _json_number(self.lower),
            "upper": _json_number(self.upper),
            "estimate": _json_number(self.estimate),
            "tolerance": self.tolerance,
            "confidence": self.confidence,
            "samples": self.samples,
            "higher_is_better": self.higher_is_better,
        }


def decide_regression(
    baseline: Sequence[float],
    candidate: Sequence[float],
    *,
    confidence: float = 0.95,
    tolerance: float = 0.0,
    practical_threshold: float | None = None,
    higher_is_better: bool = True,
    n_resamples: int = 2_000,
    seed: int = 0,
    baseline_ids: Sequence[Any] | None = None,
    candidate_ids: Sequence[Any] | None = None,
    min_observations: int | None = None,
    deterministic: bool = False,
) -> RegressionDecision:
    if practical_threshold is not None:
        tolerance = practical_threshold
    required = (
        1
        if deterministic
        else (MIN_STOCHASTIC_OBSERVATIONS if min_observations is None else int(min_observations))
    )
    if isinstance(required, bool) or required < 1:
        raise ValueError("min_observations must be a positive integer")
    differences = paired_differences(
        baseline,
        candidate,
        baseline_ids=baseline_ids,
        candidate_ids=candidate_ids,
    )
    if len(differences) < required:
        return RegressionDecision(
            RegressionStatus.INCONCLUSIVE,
            float("nan"),
            float("nan"),
            float("nan"),
            abs(tolerance),
            confidence,
            len(differences),
            higher_is_better,
        )
    result = paired_bootstrap(
        baseline,
        candidate,
        confidence=confidence,
        n_resamples=n_resamples,
        seed=seed,
        baseline_ids=baseline_ids,
        candidate_ids=candidate_ids,
    )
    lower, upper = result.confidence_interval
    threshold = abs(float(tolerance))
    if higher_is_better:
        regression, non_regression = upper < -threshold, lower >= -threshold
    else:
        regression, non_regression = lower > threshold, upper <= threshold
    status = (
        RegressionStatus.REGRESSION
        if regression
        else RegressionStatus.NO_REGRESSION
        if non_regression
        else RegressionStatus.INCONCLUSIVE
    )
    return RegressionDecision(
        status,
        lower,
        upper,
        result.estimate,
        threshold,
        confidence,
        len(differences),
        higher_is_better,
    )


class BisectCache:
    """Cache revision/seed evidence under a complete execution identity.

    The public ``values`` mapping keeps the original ``(revision, seed)`` API
    for callers that do not need namespacing.  New callers should pass an
    ``identity`` so a changed command, environment, metric, or direction can
    never reuse an old result.
    """

    def __init__(
        self,
        values: Mapping[tuple[str, int], Any] | None = None,
        *,
        path: str | Path | None = None,
        identity: Mapping[str, Any] | str | None = None,
    ) -> None:
        self.path = Path(path) if path is not None else None
        self.values: dict[tuple[str, int], Any] = {}
        self.namespaced_values: dict[tuple[str, str, int], Any] = {}
        self.identity = _canonical_identity(identity or {})
        if self.path is not None and self.path.is_file():
            try:
                raw = json.loads(self.path.read_text(encoding="utf-8"))
                if not isinstance(raw, Mapping):
                    raise ValueError("bisect cache root must be an object")
                if raw.get("version") != 2:
                    raise ValueError("unsupported bisect cache version")
                for item in raw.get("entries", []):
                    revision = str(item["revision"])
                    seed = int(item["seed"])
                    namespace = str(item.get("identity", ""))
                    if not _cache_value_is_reconstructable(item["value"]):
                        continue
                    value = _decode_cache_value(item["value"])
                    if namespace:
                        self.namespaced_values[(namespace, revision, seed)] = value
                    else:
                        self.values[(revision, seed)] = value
            except (OSError, ValueError, TypeError, KeyError):
                # A stale or interrupted cache must never prevent a diagnosis.
                self.values = {}
                self.namespaced_values = {}
        self.values.update(dict(values or {}))

    def save(self) -> None:
        if self.path is None:
            return

        def encode(value: Any) -> Any:
            if isinstance(value, FailureSignature):
                return {
                    "__mlforensics_cache_type__": "failure_signature",
                    "value": value.to_dict(),
                }
            if isinstance(value, Mapping):
                return {str(key): encode(item) for key, item in value.items()}
            if isinstance(value, (list, tuple)):
                return [encode(item) for item in value]
            try:
                json.dumps(value, allow_nan=False)
            except (TypeError, ValueError):
                return {
                    "__mlforensics_cache_type__": "unserializable",
                    "repr": repr(value),
                }
            return value

        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": 2,
            "entries": [
                {"revision": revision, "seed": seed, "value": encode(value)}
                for (revision, seed), value in sorted(self.values.items())
            ]
            + [
                {
                    "identity": identity,
                    "revision": revision,
                    "seed": seed,
                    "value": encode(value),
                }
                for (identity, revision, seed), value in sorted(self.namespaced_values.items())
            ],
        }
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        temporary.replace(self.path)

    def get(
        self,
        revision: str,
        seed: int,
        default: Any = None,
        *,
        identity: Mapping[str, Any] | str | None = None,
    ) -> Any:
        namespace = _canonical_identity(identity) if identity is not None else self.identity
        if namespace:
            return self.namespaced_values.get((namespace, str(revision), int(seed)), default)
        return self.values.get((str(revision), int(seed)), default)

    def set(
        self,
        revision: str,
        seed: int,
        value: Any,
        *,
        identity: Mapping[str, Any] | str | None = None,
    ) -> None:
        namespace = _canonical_identity(identity) if identity is not None else self.identity
        if namespace:
            self.namespaced_values[(namespace, str(revision), int(seed))] = value
        else:
            self.values[(str(revision), int(seed))] = value
        self.save()

    def __getitem__(self, key: tuple[str, int]) -> Any:
        return self.values[key]

    def __setitem__(self, key: tuple[str, int], value: Any) -> None:
        revision, seed = key
        self.values[(str(revision), int(seed))] = value
        self.save()

    def __iter__(self):
        return iter(self.values)

    def __len__(self) -> int:
        return len(self.values)

    def __contains__(self, key: tuple[str, int]) -> bool:
        return key in self.values


@dataclass
class BisectReport:
    first_bad: str | None
    decisions: dict[str, RegressionDecision] = field(default_factory=dict)
    evaluations: list[RunOutcome] = field(default_factory=list)
    inconclusive: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "first_bad": self.first_bad,
            "decisions": {key: value.to_dict() for key, value in self.decisions.items()},
            "evaluations": [value.to_dict() for value in self.evaluations],
            "inconclusive": list(self.inconclusive),
            "metadata": dict(self.metadata),
        }


def bisect_commits(
    git: Any,
    runner: Callable[[int], Any],
    seeds: Sequence[int],
    *,
    cache: BisectCache | None = None,
    tolerance: float = 0.0,
    confidence: float = 0.95,
    higher_is_better: bool = False,
    n_resamples: int = 2_000,
    max_runs: int | None = None,
    budget: Mapping[str, Any] | Any | None = None,
    min_observations: int | None = None,
    good: str | None = None,
    bad: str = "HEAD",
    execution_identity: Mapping[str, Any] | str | None = None,
    command: Sequence[str] | str | None = None,
    metric: str | None = None,
    environment_fingerprint: Any = None,
    configuration: Mapping[str, Any] | None = None,
    failure_predicate: Callable[[Any], bool] | None = None,
    predicate: Callable[[Any], bool] | None = None,
    detect_nonmonotonic: bool = True,
) -> BisectReport:
    """Checkout and evaluate only the logarithmic set of revisions needed."""
    if failure_predicate is None:
        failure_predicate = predicate
    if min_observations is not None and (
        isinstance(min_observations, bool) or int(min_observations) < 1
    ):
        raise ValueError("min_observations must be a positive integer")
    seed_list = [int(seed) for seed in seeds]
    if len(set(seed_list)) != len(seed_list):
        raise ValueError("seeds must be unique so observations can be paired safely")
    if good is None:
        revisions = list(git.commits())
    else:
        try:
            revisions = list(git.commits(good, bad))
        except TypeError:
            # Small user-supplied Git adapters often only expose commits().
            all_revisions = list(git.commits())
            if good not in all_revisions:
                raise ValueError(f"good revision {good!r} is not present")
            end = all_revisions.index(bad) + 1 if bad in all_revisions else len(all_revisions)
            revisions = all_revisions[all_revisions.index(good) : end]
    if len(revisions) < 2:
        return BisectReport(None, metadata={"reason": "fewer than two revisions"})
    if max_runs is not None and (
        isinstance(max_runs, bool) or not isinstance(max_runs, int) or max_runs < 2
    ):
        raise ValueError("max_runs must allow at least the good and bad revisions")
    budget_run_count = _budget_limit(budget, "run_count")
    if budget_run_count is not None and (
        isinstance(budget_run_count, bool)
        or not isinstance(budget_run_count, int)
        or budget_run_count < 0
    ):
        raise ValueError("budget.run_count must be a non-negative integer")
    budget_limits = {
        name: _budget_limit(budget, name)
        for name in ("run_count", "wall_clock_s", "gpu_hours", "currency")
    }
    for name in ("wall_clock_s", "gpu_hours", "currency"):
        value = budget_limits[name]
        if value is not None:
            try:
                valid = math.isfinite(float(value)) and float(value) >= 0
            except (TypeError, ValueError):
                valid = False
            if not valid:
                raise ValueError(f"budget.{name} must be a non-negative finite number")
    run_limit = max_runs
    if budget_run_count is not None:
        run_limit = budget_run_count if run_limit is None else min(run_limit, int(budget_run_count))
    identity_defaults: dict[str, Any] = {
        "schema_version": 2,
        "repository": str(getattr(git, "repo", "")),
        "code_revision": {"good": good, "bad": bad},
        "command": command,
        "harness": command,
        "metric": metric,
        "higher_is_better": bool(higher_is_better),
        "tolerance": float(tolerance),
        "confidence": float(confidence),
        "n_resamples": int(n_resamples),
        "environment_fingerprint": environment_fingerprint,
        "configuration": dict(configuration or {}),
        "seeds": seed_list,
        "min_observations": min_observations,
        "failure_predicate": _callable_identity(failure_predicate),
    }
    if isinstance(execution_identity, Mapping):
        # An explicit identity may add harness-specific inputs, but cannot
        # accidentally remove the standard inputs that protect cached data.
        identity = {**identity_defaults, **dict(execution_identity)}
    elif execution_identity is not None:
        identity = {**identity_defaults, "user_identity": execution_identity}
    else:
        identity = identity_defaults
        if command is None:
            identity = {
                **identity,
                "cache_reuse": "disabled",
                "search_id": uuid.uuid4().hex,
            }
    identity_token = _canonical_identity(identity)
    cache = cache if cache is not None else BisectCache()
    if min_observations is not None:
        required_seeds = int(min_observations)
    elif failure_predicate is not None:
        required_seeds = 1
    else:
        required_seeds = MIN_STOCHASTIC_OBSERVATIONS
    if len(seed_list) < required_seeds:
        # Reject this before spending any compute: with fewer seeds than the
        # evidence requirement every revision can only come back inconclusive.
        raise ValueError(
            f"a stochastic bisect needs at least {required_seeds} seeds to classify a "
            f"revision but {len(seed_list)} were supplied; pass more seeds or lower "
            f"min_observations (accepting weaker evidence)"
        )
    required_observations = required_seeds
    raw: dict[str, list[Any]] = {}
    decisions: dict[str, RegressionDecision] = {}
    evaluations: list[RunOutcome] = []
    executed_runs = 0
    cache_hits = 0
    budget_consumed = {"wall_clock_s": 0.0, "gpu_hours": 0.0, "currency": 0.0}
    budget_exhausted = False
    started_at = time.monotonic()
    original_revision = None
    preflight = getattr(git, "preflight", None)
    if callable(preflight):
        preflight()
    current_revision = getattr(git, "current_revision", None)
    if callable(current_revision):
        original_revision = current_revision()

    def evaluate(revision: str) -> list[Any]:
        nonlocal budget_exhausted, cache_hits, executed_runs
        if revision in raw:
            return raw[revision]
        git.checkout(revision)
        values = []
        for seed in seed_list:
            sentinel = object()
            if isinstance(cache, BisectCache):
                value = cache.get(revision, seed, sentinel, identity=identity_token)
            else:
                key: tuple[Any, ...] = (revision, seed)
                if identity_token != "{}":
                    key = (identity_token, revision, seed)
                value = cache.get(key, sentinel)  # type: ignore[union-attr]
            if value is not sentinel:
                cache_hits += 1
            else:
                elapsed = time.monotonic() - started_at
                budget_consumed["wall_clock_s"] = elapsed
                if run_limit is not None and executed_runs >= run_limit:
                    budget_exhausted = True
                    break
                if budget_limits["wall_clock_s"] is not None and elapsed >= float(
                    budget_limits["wall_clock_s"]
                ):
                    budget_exhausted = True
                    break
                if any(
                    budget_limits[name] is not None
                    and budget_consumed[name] >= float(budget_limits[name])
                    for name in ("gpu_hours", "currency")
                ):
                    budget_exhausted = True
                    break
                try:
                    value = _invoke_bisect_runner(runner, seed, git)
                except Exception as exc:
                    # A revision-level exception is evidence, not a reason to
                    # abort the entire search.  Signature predicates can
                    # classify it; numeric searches remain inconclusive.
                    value = FailureSignature.from_exception(exc)
                if isinstance(cache, BisectCache):
                    cache.set(revision, seed, value, identity=identity_token)
                else:
                    key = (revision, seed)
                    if identity_token != "{}":
                        key = (identity_token, revision, seed)
                    cache[key] = value  # type: ignore[index]
                executed_runs += 1
                budget_consumed["gpu_hours"] += _reported_resource(value, {"gpu_hours"})
                budget_consumed["currency"] += _reported_resource(
                    value, {"currency", "cost", "cost_usd"}
                )
                budget_consumed["wall_clock_s"] = time.monotonic() - started_at
                if any(
                    budget_limits[name] is not None
                    and budget_consumed[name] >= float(budget_limits[name])
                    for name in ("gpu_hours", "currency")
                ):
                    budget_exhausted = True
            values.append(value)
        raw[revision] = values
        return values

    def outcome(
        revision: str,
        values: list[Any],
        baseline: list[Any] | None = None,
        baseline_ids: Sequence[int] | None = None,
    ) -> RunOutcome:
        normalized = [_to_score(value) for value in values]
        observation_ids = seed_list[: len(values)]
        observation_metadata = {
            "observation_ids": list(observation_ids),
            "missing_observation_ids": seed_list[len(values) :],
        }
        if failure_predicate is not None:
            return _predicate_outcome(
                revision,
                values,
                failure_predicate,
                metadata=observation_metadata,
            )
        flags = [flag for flag, _score, _details in normalized if flag is not None]
        if flags:
            if len(flags) < required_observations:
                return RunOutcome(
                    revision,
                    True,
                    runs=len(values),
                    metadata={
                        **observation_metadata,
                        "votes": flags,
                        "inconclusive": True,
                        "reason": "insufficient observations",
                        "required_observations": required_observations,
                    },
                )
            passing = sum(bool(flag) for flag in flags)
            tied = len(flags) % 2 == 0 and passing == len(flags) / 2
            passed = passing > len(flags) / 2
            return RunOutcome(
                revision,
                passed,
                runs=len(values),
                metadata={**observation_metadata, "votes": flags, "inconclusive": tied},
            )
        ids = list(observation_ids)
        scores_by_seed = {
            seed: score
            for seed, (_flag, score, _details) in zip(ids, normalized)
            if score is not None and math.isfinite(score)
        }
        scores = list(scores_by_seed.values())
        if baseline is None:
            if not scores:
                return RunOutcome(
                    revision,
                    True,
                    runs=0,
                    metadata={
                        **observation_metadata,
                        "inconclusive": True,
                        "reason": "no usable endpoint evidence",
                    },
                )
            if len(scores) < required_observations:
                return RunOutcome(
                    revision,
                    True,
                    score=statistics.fmean(scores),
                    samples=scores,
                    runs=len(scores),
                    metadata={
                        **observation_metadata,
                        "inconclusive": True,
                        "reason": "insufficient observations",
                        "required_observations": required_observations,
                    },
                )
            return RunOutcome(
                revision,
                True,
                score=statistics.fmean(scores) if scores else None,
                samples=scores,
                runs=len(scores),
                metadata=observation_metadata,
            )
        base_ids = list(baseline_ids or seed_list[: len(baseline)])
        base_scores_by_seed = {
            seed: score
            for seed, (_flag, score, _details) in zip(
                base_ids, [_to_score(value) for value in baseline]
            )
            if score is not None and math.isfinite(score)
        }
        paired_base, paired_candidate, paired_ids = _stable_pairs(
            base_scores_by_seed, scores_by_seed
        )
        if not paired_base or not paired_candidate:
            return RunOutcome(
                revision,
                True,
                samples=scores,
                runs=len(scores),
                metadata={**observation_metadata, "inconclusive": True},
            )
        if len(paired_ids) < required_observations:
            return RunOutcome(
                revision,
                True,
                samples=scores,
                runs=len(scores),
                metadata={
                    **observation_metadata,
                    "inconclusive": True,
                    "reason": "insufficient matched observations",
                    "required_observations": required_observations,
                    "paired_observation_ids": list(paired_ids),
                },
            )
        decision = decide_regression(
            paired_base,
            paired_candidate,
            confidence=confidence,
            tolerance=tolerance,
            higher_is_better=higher_is_better,
            n_resamples=n_resamples,
            baseline_ids=paired_ids,
            candidate_ids=paired_ids,
            min_observations=required_observations,
        )
        decisions[revision] = decision
        return RunOutcome(
            revision,
            decision.status is not RegressionStatus.REGRESSION,
            score=statistics.fmean(scores),
            samples=scores,
            confidence_interval=(decision.lower, decision.upper),
            runs=len(scores),
            metadata={
                **observation_metadata,
                "status": decision.status.value,
                "inconclusive": decision.status is RegressionStatus.INCONCLUSIVE,
                "paired_observation_ids": list(paired_ids),
                "unpaired_baseline_ids": sorted(
                    set(base_scores_by_seed).difference(scores_by_seed)
                ),
                "unpaired_candidate_ids": sorted(
                    set(scores_by_seed).difference(base_scores_by_seed)
                ),
            },
        )

    try:
        good_values = evaluate(revisions[0])
        bad_values = evaluate(revisions[-1])
        good_outcome = outcome(revisions[0], good_values)
        bad_outcome = outcome(revisions[-1], bad_values, good_values)
        evaluations.extend([good_outcome, bad_outcome])
        metadata = {
            "executed_runs": executed_runs,
            "cache_hits": cache_hits,
            "resumed": cache_hits > 0,
            "max_runs": max_runs,
            "budget": budget_limits,
            "budget_consumed": dict(budget_consumed),
            "budget_exhausted": budget_exhausted,
            "execution_identity": identity_token,
            "paired_seeds": list(seed_list),
        }
        if good_outcome.inconclusive or (failure_predicate is not None and not good_outcome.passed):
            return BisectReport(
                None,
                decisions,
                evaluations,
                [revisions[0]],
                {
                    **metadata,
                    "reason": (
                        "good revision does not reproduce a healthy endpoint"
                        if failure_predicate is not None and not good_outcome.passed
                        else "good revision has insufficient evidence"
                    ),
                },
            )
        if bad_outcome.inconclusive or bad_outcome.passed:
            return BisectReport(
                None,
                decisions,
                evaluations,
                [revisions[-1]],
                {**metadata, "reason": "bad revision is not conclusively regressed"},
            )
        low, high = 0, len(revisions) - 1
        inconclusive: list[str] = []
        outcome_by_revision = {
            item.target: item for item in evaluations if item.target in revisions
        }
        while high - low > 1:
            middle = (low + high) // 2
            current_values = evaluate(revisions[middle])
            current = outcome(revisions[middle], current_values, good_values)
            evaluations.append(current)
            outcome_by_revision[revisions[middle]] = current
            if current.inconclusive:
                inconclusive.append(revisions[middle])
                return BisectReport(
                    None,
                    decisions,
                    evaluations,
                    inconclusive,
                    {
                        **metadata,
                        "reason": "an intermediate revision is statistically inconclusive",
                        "executed_runs": executed_runs,
                        "budget_consumed": dict(budget_consumed),
                        "budget_exhausted": budget_exhausted,
                    },
                )
            if current.passed:
                low = middle
            else:
                high = middle

        if detect_nonmonotonic:
            # A binary search cannot observe a good revision hidden after an
            # earlier bad one.  The bounded scan turns that ambiguity into an
            # explicit result and, on monotonic histories, records evidence for
            # every revision that was needed to prove the claim.
            for revision in revisions:
                if revision in outcome_by_revision:
                    continue
                current_values = evaluate(revision)
                current = outcome(revision, current_values, good_values)
                evaluations.append(current)
                outcome_by_revision[revision] = current
                if current.inconclusive:
                    inconclusive.append(revision)
                    return BisectReport(
                        None,
                        decisions,
                        evaluations,
                        inconclusive,
                        {
                            **metadata,
                            "executed_runs": executed_runs,
                            "budget_consumed": dict(budget_consumed),
                            "budget_exhausted": budget_exhausted,
                            "reason": "nonmonotonicity scan is inconclusive",
                        },
                    )
            statuses = [not outcome_by_revision[revision].passed for revision in revisions]
            first_bad_index = next((index for index, is_bad in enumerate(statuses) if is_bad), None)
            violation = first_bad_index is not None and any(
                statuses[index] is False for index in range(first_bad_index + 1, len(statuses))
            )
            if violation:
                return BisectReport(
                    None,
                    decisions,
                    evaluations,
                    [revisions[index] for index, is_bad in enumerate(statuses) if is_bad],
                    {
                        **metadata,
                        "executed_runs": executed_runs,
                        "budget_consumed": dict(budget_consumed),
                        "budget_exhausted": budget_exhausted,
                        "nonmonotonic": True,
                        "reason": "revision history is nonmonotonic; first bad is ambiguous",
                    },
                )
            high = first_bad_index if first_bad_index is not None else high
        metadata["executed_runs"] = executed_runs
        metadata["budget_consumed"] = dict(budget_consumed)
        metadata["budget_exhausted"] = budget_exhausted
        return BisectReport(
            revisions[high],
            decisions,
            evaluations,
            inconclusive,
            metadata,
        )
    finally:
        try:
            restore = getattr(git, "restore", None)
            if original_revision is not None and callable(restore):
                restore(original_revision)
        finally:
            cleanup = getattr(git, "cleanup", None)
            if callable(cleanup):
                cleanup()
