"""Optional Weights & Biases capsule import/export bridge."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

from ..core import RunCapsule

_ARTIFACT_TYPE = "mlforensics-capsule"
_CAPSULE_NAME = "capsule.mlcap.zip"


class WandBAdapter:
    """Publish portable capsules as W&B artifacts and retrieve them later."""

    def __init__(self, client: Any | None = None, *, api: Any | None = None) -> None:
        if client is None:
            try:
                import wandb  # type: ignore[import-not-found]

                client = wandb
            except ImportError as exc:
                raise ImportError(
                    "WandBAdapter requires Weights & Biases; install 'mlforensics[wandb]'"
                ) from exc
        self.client = client
        self.api = api

    def export_capsule(
        self,
        capsule: RunCapsule,
        *,
        project: str | None = None,
        entity: str | None = None,
        run_id: str | None = None,
    ) -> str | None:
        """Create a W&B run, log metrics, and attach the complete capsule."""
        initializer = getattr(self.client, "init", None)
        artifact_type = getattr(self.client, "Artifact", None)
        if initializer is None or artifact_type is None:
            raise TypeError("W&B client must expose init() and Artifact")
        run = initializer(
            project=project,
            entity=entity,
            id=run_id,
            resume="allow" if run_id else None,
            config=dict(capsule.run.metadata),
        )
        if run is None:
            raise RuntimeError("wandb.init() returned no run (is W&B disabled?)")
        try:
            for metric in capsule.run.metrics:
                for index, value in enumerate(metric.values):
                    step = metric.steps[index] if index < len(metric.steps) else index
                    run.log({metric.name: value}, step=step)
            artifact = artifact_type(
                f"mlforensics-{capsule.run_id}",
                type=_ARTIFACT_TYPE,
                metadata={
                    "run_id": capsule.run_id,
                    "schema_version": capsule.schema_version,
                    "digest": capsule.digest,
                },
            )
            with tempfile.TemporaryDirectory(prefix="mlforensics-wandb-") as temporary:
                capsule_path = Path(temporary) / _CAPSULE_NAME
                capsule.save(capsule_path, zipped=True)
                artifact.add_file(str(capsule_path), name=_CAPSULE_NAME)
                run.log_artifact(artifact)
            identifier = getattr(run, "id", run_id)
            return str(identifier) if identifier is not None else None
        finally:
            finish = getattr(run, "finish", None)
            if finish is not None:
                finish()

    def import_capsule(self, run_path: str, *, artifact_name: str | None = None) -> RunCapsule:
        """Retrieve and validate a capsule from a W&B run or named artifact."""
        api = self.api
        if api is None:
            api_factory = getattr(self.client, "Api", None)
            if api_factory is None:
                raise TypeError("W&B client must expose Api() or receive api= explicitly")
            api = api_factory()
        if artifact_name is not None:
            artifact = api.artifact(artifact_name)
        else:
            remote_run = api.run(run_path)
            artifacts = list(remote_run.logged_artifacts())
            matching = [item for item in artifacts if getattr(item, "type", None) == _ARTIFACT_TYPE]
            if not matching:
                raise FileNotFoundError(f"W&B run {run_path!r} has no {_ARTIFACT_TYPE!r} artifact")
            artifact = matching[-1]
        with tempfile.TemporaryDirectory(prefix="mlforensics-wandb-") as temporary:
            downloaded = Path(artifact.download(root=temporary))
            path = downloaded / _CAPSULE_NAME if downloaded.is_dir() else downloaded
            if not path.is_file():
                candidates = list(Path(temporary).rglob(_CAPSULE_NAME))
                if len(candidates) != 1:
                    raise FileNotFoundError(f"W&B artifact does not contain {_CAPSULE_NAME!r}")
                path = candidates[0]
            return RunCapsule.load(path)
