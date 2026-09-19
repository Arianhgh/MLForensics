"""Torch-free ReplayFixture used by isolated worker tests."""


class DummyReplayFixture:
    def __init__(self) -> None:
        self.state: dict[str, object] = {}

    def construct(self) -> dict[str, object]:
        self.state = {"ready": True}
        return self.state

    def restore(self, checkpoint: object) -> None:
        if isinstance(checkpoint, dict) and "model" in checkpoint:
            self.state = {"model": checkpoint["model"], "ready": True}
        elif isinstance(checkpoint, dict):
            self.state.update(checkpoint)

    def restore_component(self, name: str, state: object) -> None:
        self.restore({name: state})

    def execute(self, value: object) -> object:
        if value == "boom" or (isinstance(value, dict) and value.get("fail")):
            raise ValueError("target failure")
        return value

    def close(self) -> None:
        self.state = {}
