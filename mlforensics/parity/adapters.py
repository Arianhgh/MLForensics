"""Optional model runtime adapters."""

from __future__ import annotations

from typing import Any

from .backends import (
    CallableBackendAdapter,
    GenericBackendAdapter,
    ONNXRuntimeBackendAdapter,
    PyTorchBackendAdapter,
)


class CallableAdapter:
    def __init__(self, model: Any, *, name: str | None = None) -> None:
        if not callable(model):
            raise TypeError("model must be callable")
        self.model = model
        self.name = name or getattr(model, "__name__", type(model).__name__)

    def predict(self, inputs: Any) -> Any:
        return self.model(inputs)


class PyTorchAdapter(CallableAdapter):
    def __init__(self, model: Any, *, device: str | None = None, name: str | None = None) -> None:
        try:
            import torch  # type: ignore
        except ImportError as exc:
            raise ImportError("PyTorchAdapter requires the optional 'torch' dependency") from exc
        self.torch = torch
        self.device = device
        super().__init__(model, name=name)

    def predict(self, inputs: Any) -> Any:
        tensor = inputs
        if not isinstance(tensor, self.torch.Tensor):
            tensor = self.torch.tensor(inputs, dtype=self.torch.float32)
        if self.device:
            tensor = tensor.to(self.device)
        with self.torch.no_grad():
            output = self.model(tensor)
        if isinstance(output, (tuple, list)):
            output = output[0]
        return output.detach().cpu().tolist() if hasattr(output, "detach") else output


class ONNXRuntimeAdapter:
    def __init__(
        self, session: Any, *, input_name: str | None = None, name: str = "onnxruntime"
    ) -> None:
        if not hasattr(session, "run"):
            raise TypeError("session must expose run()")
        self.session = session
        self.name = name
        self.input_name = input_name or self._discover_input_name()

    def _discover_input_name(self) -> str:
        try:
            return self.session.get_inputs()[0].name
        except Exception as exc:
            raise ValueError(
                "input_name is required when the ONNX session has no get_inputs()"
            ) from exc

    def predict(self, inputs: Any) -> Any:
        value = inputs
        try:
            import numpy as np  # type: ignore

            if not isinstance(value, np.ndarray):
                value = np.asarray(value, dtype=np.float32)
        except ImportError:
            pass
        result = self.session.run(None, {self.input_name: value})
        return result[0] if isinstance(result, (list, tuple)) and len(result) == 1 else result


def as_adapter(model: Any, *, name: str | None = None):
    if hasattr(model, "predict") and callable(model.predict):
        return model
    return CallableAdapter(model, name=name)


# New generic names live in ``backends``; these aliases keep the original
# parity adapter module useful to callers that imported it directly.
CallableBackend = CallableBackendAdapter
GenericAdapter = GenericBackendAdapter
TorchAdapter = PyTorchBackendAdapter
ONNXAdapter = ONNXRuntimeBackendAdapter
