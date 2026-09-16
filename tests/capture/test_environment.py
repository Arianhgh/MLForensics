from mlforensics.capture import capture_environment, capture_python_environment


def test_environment_capture_respects_allowlist() -> None:
    result = capture_environment(
        allowlist=("SAFE_CAPTURE_VALUE",),
        environ={"SAFE_CAPTURE_VALUE": "yes", "SECRET_TOKEN": "do-not-capture"},
    )
    assert result["variables"] == {"SAFE_CAPTURE_VALUE": "yes"}
    assert "SECRET_TOKEN" not in result["variables"]
    assert result["python_version"]


def test_python_runtime_capture_has_interpreter_identity() -> None:
    result = capture_python_environment()
    assert result["implementation"]
    assert result["executable"]
    assert len(result["version_info"]) == 5
