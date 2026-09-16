import json
import sys
from pathlib import Path

import pytest

from mlforensics.cli.config import load_config
from mlforensics.cli.main import main
from mlforensics.core import RunCapsule
from mlforensics.core.index import load_index


def test_unknown_config_keys_are_rejected(tmp_path: Path) -> None:
    path = tmp_path / "mlforensics.toml"
    path.write_text("[ci]\nfail_on_regresion = true\n", encoding="utf-8")
    with pytest.raises(ValueError, match="unsupported \\[ci\\] key"):
        load_config(path)

    path.write_text("[unknown]\nroot = '.'\n", encoding="utf-8")
    with pytest.raises(ValueError, match="unsupported configuration section"):
        load_config(path)


def test_commands_resolve_run_ids_under_configured_storage_root(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root = tmp_path / "store"
    config = tmp_path / "mlforensics.toml"
    config.write_text(f"[storage]\nroot = {json.dumps(str(root))}\n", encoding="utf-8")

    exit_code = main(
        [
            "--config",
            str(config),
            "run",
            "--repo",
            str(tmp_path),
            "--json",
            "--no-output",
            sys.executable,
            "-c",
            "print('ok')",
        ]
    )
    assert exit_code == 0
    summary = json.loads(capsys.readouterr().out)
    capsule_path = Path(summary["capsule"])
    assert capsule_path.parent == root
    capsule = RunCapsule.load(capsule_path)
    run_id = capsule.run.run_id
    index = load_index(root)
    assert run_id in index["runs"]

    assert main(["--config", str(config), "replay", run_id, "--json"]) == 0
    replayed = json.loads(capsys.readouterr().out)
    assert Path(replayed["incident"]) == capsule_path
    assert replayed["status"] == capsule.run.status
