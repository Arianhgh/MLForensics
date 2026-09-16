import shutil
from pathlib import Path
from types import SimpleNamespace

from mlforensics import ArtifactRef, MetricSeries, Run, RunCapsule
from mlforensics.integrations import MLflowAdapter, WandBAdapter


def capsule_fixture() -> RunCapsule:
    payload = b"portable model"
    ref = ArtifactRef.from_bytes("model.bin", payload)
    return RunCapsule(
        Run(
            run_id="source-run",
            status="completed",
            metrics=(MetricSeries("loss", (2.0, 1.0), steps=(10, 20)),),
            metadata={"seed": 7, "nested": {"safe": True}},
        ),
        artifacts=(ref,),
        payloads={ref.sha256: payload},
    )


class FakeMLflowRun:
    def __init__(self, client, run_id):
        self.client = client
        self.info = SimpleNamespace(run_id=run_id)

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    def log_param(self, key, value):
        self.client.params[key] = value

    def log_metric(self, key, value, *, step):
        self.client.metrics.append((key, value, step))

    def set_tag(self, key, value):
        self.client.tags[key] = value

    def log_artifact(self, path, *, artifact_path):
        assert artifact_path == "mlforensics"
        shutil.copy2(path, self.client.capsule_path)


class FakeMLflow:
    def __init__(self, root: Path):
        self.params = {}
        self.metrics = []
        self.tags = {}
        self.capsule_path = root / "stored-capsule.zip"
        self.artifacts = self

    def start_run(self, run_id=None):
        return FakeMLflowRun(self, run_id or "mlflow-generated")

    def download_artifacts(self, *, run_id, artifact_path, dst_path):
        assert run_id == "mlflow-generated"
        assert artifact_path == "mlforensics/capsule.mlcap.zip"
        target = Path(dst_path) / "capsule.mlcap.zip"
        shutil.copy2(self.capsule_path, target)
        return str(target)

    def get_run(self, run_id):
        return SimpleNamespace(
            info=SimpleNamespace(run_id=run_id, status="FINISHED"),
            data=SimpleNamespace(metrics={"loss": 1.0}, params=self.params, tags=self.tags),
        )

    def get_metric_history(self, run_id, name):
        return [SimpleNamespace(value=2.0, step=10), SimpleNamespace(value=1.0, step=20)]


def test_mlflow_capsule_round_trip_and_summary_import(tmp_path) -> None:
    client = FakeMLflow(tmp_path)
    adapter = MLflowAdapter(client)
    capsule = capsule_fixture()

    assert adapter.export_capsule(capsule) == "mlflow-generated"
    assert client.metrics == [("loss", 2.0, 10), ("loss", 1.0, 20)]
    assert client.params["nested"] == '{"safe": true}'
    assert client.tags["mlforensics.capsule_digest"] == capsule.digest
    imported = adapter.import_capsule("mlflow-generated")
    assert imported.run == capsule.run
    assert imported.payloads == capsule.payloads

    summary = adapter.import_run("mlflow-generated")
    assert summary.status == "completed"
    assert summary.metrics[0].values == (2.0, 1.0)
    assert summary.metrics[0].steps == (10, 20)


class FakeWandBArtifact:
    def __init__(self, root: Path, name: str, type: str, metadata):
        self.root = root
        self.name = name
        self.type = type
        self.metadata = metadata
        self.stored = root / f"{name}.zip"

    def add_file(self, path, *, name):
        assert name == "capsule.mlcap.zip"
        shutil.copy2(path, self.stored)

    def download(self, *, root):
        destination = Path(root) / self.name
        destination.mkdir()
        shutil.copy2(self.stored, destination / "capsule.mlcap.zip")
        return str(destination)


class FakeWandBRun:
    def __init__(self, client, run_id):
        self.client = client
        self.id = run_id
        self.logged = []
        self.finished = False

    def log(self, values, *, step):
        self.logged.append((values, step))

    def log_artifact(self, artifact):
        self.client.artifacts.append(artifact)

    def finish(self):
        self.finished = True


class FakeRemoteWandBRun:
    def __init__(self, artifacts):
        self.artifacts = artifacts

    def logged_artifacts(self):
        return self.artifacts


class FakeWandBApi:
    def __init__(self, client):
        self.client = client

    def run(self, path):
        assert path == "team/project/wandb-generated"
        return FakeRemoteWandBRun(self.client.artifacts)


class FakeWandB:
    def __init__(self, root: Path):
        self.root = root
        self.artifacts = []
        self.run = None
        self.init_kwargs = None

    def init(self, **kwargs):
        self.init_kwargs = kwargs
        self.run = FakeWandBRun(self, kwargs["id"] or "wandb-generated")
        return self.run

    def Artifact(self, name, *, type, metadata):
        return FakeWandBArtifact(self.root, name, type, metadata)


def test_wandb_capsule_round_trip_finishes_run(tmp_path) -> None:
    client = FakeWandB(tmp_path)
    adapter = WandBAdapter(client, api=FakeWandBApi(client))
    capsule = capsule_fixture()

    assert adapter.export_capsule(capsule, project="demo") == "wandb-generated"
    assert client.run.logged == [({"loss": 2.0}, 10), ({"loss": 1.0}, 20)]
    assert client.run.finished is True
    assert client.artifacts[0].metadata["digest"] == capsule.digest
    imported = adapter.import_capsule("team/project/wandb-generated")
    assert imported.run == capsule.run
    assert imported.payloads == capsule.payloads
