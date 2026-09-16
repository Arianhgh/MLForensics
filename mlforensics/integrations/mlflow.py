"""Optional MLflow bridge implemented through public fluent APIs."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

from ..core import MetricSeries, Run, RunCapsule

_CAPSULE_ARTIFACT_PATH = "mlforensics/capsule.mlcap.zip"


def _field(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, dict):
        return value.get(name, default)
    return getattr(value, name, default)


class MLflowAdapter:
    """Import and export run capsules through an MLflow-compatible client.

    The default client is the public ``mlflow`` fluent module. Supplying a
    duck-typed client makes the adapter straightforward to test and permits
    compatible tracking frontends without importing MLflow.
    """

    def __init__(self, client: Any | None = None) -> None:
        if client is None:
            try:
                import mlflow  # type: ignore[import-not-found]

                client = mlflow
            except ImportError as exc:
                raise ImportError(
                    "MLflowAdapter requires MLflow; install 'mlforensics[mlflow]'"
                ) from exc
        self.client = client

    def _log(self, active: Any, name: str, *args: Any, **kwargs: Any) -> Any:
        logger = getattr(active, name, None) or getattr(self.client, name, None)
        if logger is None:
            raise TypeError(f"MLflow client must expose {name}()")
        return logger(*args, **kwargs)

    def export_capsule(self, capsule: RunCapsule, *, run_id: str | None = None) -> str | None:
        """Log metrics and a complete portable capsule, returning the MLflow run id."""
        start_run = getattr(self.client, "start_run", None)
        if start_run is None:
            raise TypeError("MLflow client must expose start_run()")
        context = start_run(run_id=run_id) if run_id else start_run()
        with context as active:
            info = _field(active, "info")
            active_run_id = _field(info, "run_id", run_id)
            for key, value in capsule.run.metadata.items():
                encoded = value if isinstance(value, str) else json.dumps(value, sort_keys=True)
                self._log(active, "log_param", str(key), encoded)
            for metric in capsule.run.metrics:
                for index, value in enumerate(metric.values):
                    step = metric.steps[index] if index < len(metric.steps) else index
                    self._log(active, "log_metric", metric.name, value, step=step)
            self._log(active, "set_tag", "mlforensics.run_id", capsule.run_id)
            self._log(active, "set_tag", "mlforensics.schema_version", str(capsule.schema_version))
            self._log(active, "set_tag", "mlforensics.capsule_digest", capsule.digest)
            with tempfile.TemporaryDirectory(prefix="mlforensics-mlflow-") as temporary:
                capsule_path = Path(temporary) / "capsule.mlcap.zip"
                capsule.save(capsule_path, zipped=True)
                self._log(active, "log_artifact", str(capsule_path), artifact_path="mlforensics")
            return str(active_run_id) if active_run_id is not None else None

    def import_capsule(self, run_id: str) -> RunCapsule:
        """Download and validate the capsule artifact attached to ``run_id``."""
        artifacts_api = getattr(self.client, "artifacts", None)
        downloader = getattr(artifacts_api, "download_artifacts", None) or getattr(
            self.client, "download_artifacts", None
        )
        if downloader is None:
            raise TypeError("MLflow client must expose artifacts.download_artifacts()")
        with tempfile.TemporaryDirectory(prefix="mlforensics-mlflow-") as temporary:
            downloaded = downloader(
                run_id=run_id,
                artifact_path=_CAPSULE_ARTIFACT_PATH,
                dst_path=temporary,
            )
            path = Path(downloaded)
            if path.is_dir():
                path = path / "capsule.mlcap.zip"
            if not path.is_file():
                candidates = list(Path(temporary).rglob("capsule.mlcap.zip"))
                if len(candidates) != 1:
                    raise FileNotFoundError(
                        f"MLflow run {run_id!r} does not contain {_CAPSULE_ARTIFACT_PATH!r}"
                    )
                path = candidates[0]
            return RunCapsule.load(path)

    def import_run(self, run_id: str) -> Run:
        """Import MLflow's run summary when a portable capsule is unavailable."""
        getter = getattr(self.client, "get_run", None)
        if getter is None:
            raise TypeError("MLflow client must expose get_run()")
        remote = getter(run_id)
        info = _field(remote, "info", {})
        data = _field(remote, "data", {})
        metric_values = dict(_field(data, "metrics", {}) or {})
        history_getter = getattr(self.client, "get_metric_history", None)
        metrics: list[MetricSeries] = []
        for name, value in metric_values.items():
            history = history_getter(run_id, name) if history_getter is not None else ()
            if history:
                metrics.append(
                    MetricSeries(
                        name,
                        values=tuple(float(_field(item, "value")) for item in history),
                        steps=tuple(
                            int(_field(item, "step", index)) for index, item in enumerate(history)
                        ),
                    )
                )
            else:
                metrics.append(MetricSeries.from_value(name, value))
        remote_status = str(_field(info, "status", "finished")).lower()
        status = {
            "finished": "completed",
            "success": "completed",
            "failed": "failed",
            "killed": "failed",
        }.get(remote_status, remote_status)
        metadata = {
            **dict(_field(data, "params", {}) or {}),
            **dict(_field(data, "tags", {}) or {}),
        }
        return Run(
            run_id=str(_field(info, "run_id", run_id)),
            status=status,
            metrics=metrics,
            metadata=metadata,
        )
