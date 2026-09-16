"""Backend adapters used by :mod:`mlforensics.parity`.

The parity runner deliberately depends on a very small interface.  A backend
only needs a ``predict`` method; ``predict_batch`` is optional and has a safe
per-item fallback.  This makes the comparison useful for plain Python
callables as well as framework-backed models.
"""

from __future__ import annotations

import importlib
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol


class BackendAdapter(Protocol):
    """Minimal protocol consumed by the parity comparator."""

    name: str

    def predict(self, inputs: Any) -> Any: ...

    def predict_batch(self, inputs: Sequence[Any]) -> list[Any]: ...


class BackendError(RuntimeError):
    """Raised when a backend cannot execute a prediction."""


class OptionalDependencyError(BackendError, ImportError):
    """Raised only when an optional framework adapter is actually used."""


@dataclass(frozen=True)
class BackendSpec:
    """A declarative backend reference understood by :func:`load_backend`.

    Supported kinds are ``python`` (``package.module:object``), ``onnx``, and
    ``torchscript``. General pickle-based PyTorch loading is intentionally not
    supported because loading an untrusted pickle can execute arbitrary code.
    """

    kind: str
    source: str
    name: str | None = None
    options: Mapping[str, Any] = field(default_factory=dict)


_PYTHON_TARGET = re.compile(
    r"^(?P<module>[A-Za-z_]\w*(?:\.[A-Za-z_]\w)*):"
    r"(?P<object>[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*)$"
)


def parse_backend_spec(spec: str | Path | Mapping[str, Any] | BackendSpec) -> BackendSpec:
    """Parse a backend specification without importing optional frameworks."""

    if isinstance(spec, BackendSpec):
        return spec
    if isinstance(spec, Mapping):
        unknown = set(spec) - {"kind", "source", "name", "options"}
        if unknown:
            raise ValueError(f"unknown backend specification fields: {sorted(unknown)}")
        if "kind" not in spec or "source" not in spec:
            raise ValueError("backend specification requires 'kind' and 'source'")
        options = spec.get("options", {})
        if not isinstance(options, Mapping):
            raise TypeError("backend specification options must be a mapping")
        return BackendSpec(
            str(spec["kind"]),
            str(spec["source"]),
            None if spec.get("name") is None else str(spec["name"]),
            dict(options),
        )

    source = str(spec)
    lowered = source.lower()
    for prefix, kind in (
        ("python:", "python"),
        ("onnx:", "onnx"),
        ("torchscript:", "torchscript"),
        ("torch:", "torchscript"),
        ("pytorch:", "torchscript"),
    ):
        if lowered.startswith(prefix):
            target = source[len(prefix) :]
            if not target:
                raise ValueError(f"{kind} backend specification has an empty source")
            return BackendSpec(kind, target)
    suffix = Path(source).suffix.lower()
    if suffix == ".onnx":
        return BackendSpec("onnx", source)
    if suffix in {".pt", ".pth", ".torchscript"}:
        return BackendSpec("torchscript", source)
    if _PYTHON_TARGET.fullmatch(source):
        return BackendSpec("python", source)
    raise ValueError(
        "backend must be a callable, adapter, module:object, .onnx path, or torchscript:path"
    )


def _checked_model_path(source: str, kind: str) -> Path:
    path = Path(source).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"{kind} model does not exist: {path}")
    if not path.is_file():
        raise ValueError(f"{kind} model path is not a file: {path}")
    return path


def _load_python_target(source: str) -> Any:
    match = _PYTHON_TARGET.fullmatch(source)
    if match is None:
        raise ValueError("Python backend must use package.module:object syntax")
    module = importlib.import_module(match.group("module"))
    value: Any = module
    for component in match.group("object").split("."):
        value = getattr(value, component)
    if not callable(value) and not (hasattr(value, "predict") and callable(value.predict)):
        raise TypeError(f"Python backend target is not callable: {source}")
    return value


def load_backend(
    spec: Any,
    *,
    name: str | None = None,
    options: Mapping[str, Any] | None = None,
) -> BackendAdapter:
    """Load a callable, Python target, ONNX model, or TorchScript model.

    Optional runtime packages are imported only after their backend kind has
    been selected. The loader never uses ``torch.load`` or pickle.
    """

    if hasattr(spec, "predict") and callable(spec.predict):
        return spec
    if callable(spec):
        return CallableBackendAdapter(
            spec, name=_as_name(name, getattr(spec, "__name__", "callable"))
        )
    parsed = parse_backend_spec(spec)
    merged_options = {**dict(parsed.options), **dict(options or {})}
    backend_name = name or parsed.name
    kind = parsed.kind.lower().replace("_", "-")
    if kind in {"python", "callable"}:
        value = _load_python_target(parsed.source)
        return CallableBackendAdapter(
            value, name=_as_name(backend_name, getattr(value, "__name__", parsed.source))
        )
    if kind in {"onnx", "onnxruntime", "onnx-runtime"}:
        path = _checked_model_path(parsed.source, "ONNX")
        return ONNXRuntimeBackendAdapter(
            path, name=_as_name(backend_name, "onnxruntime"), **merged_options
        )
    if kind in {"torch", "pytorch", "torchscript", "torch-script"}:
        path = _checked_model_path(parsed.source, "TorchScript")
        return PyTorchBackendAdapter(path, name=_as_name(backend_name, "pytorch"), **merged_options)
    raise ValueError(f"unsupported backend kind: {parsed.kind}")


def _as_name(value: Any, fallback: str) -> str:
    return str(value) if value is not None else fallback


@dataclass
class CallableBackendAdapter:
    """Adapter for any callable model.

    ``prepare_input`` and ``postprocess`` are useful when two backends expose
    different input/output conventions.  They are intentionally kept as
    ordinary callables so the package does not impose a tensor library.
    """

    model: Callable[[Any], Any]
    name: str = "callable"
    prepare_input: Callable[[Any], Any] | None = None
    postprocess: Callable[[Any], Any] | None = None

    def predict(self, inputs: Any) -> Any:
        prepared = self.prepare_input(inputs) if self.prepare_input else inputs
        output = self.model(prepared)
        return self.postprocess(output) if self.postprocess else output

    def predict_batch(self, inputs: Sequence[Any]) -> list[Any]:
        # The fallback is correct for callables that do not define a batch
        # convention and is also useful for heterogeneous input cases.
        return [self.predict(item) for item in inputs]


GenericBackendAdapter = CallableBackendAdapter
FunctionBackendAdapter = CallableBackendAdapter


def as_backend(model: Any, *, name: str | None = None) -> BackendAdapter:
    """Return ``model`` as an adapter, preserving already-adapted objects."""

    try:
        return load_backend(model, name=name)
    except (ValueError, TypeError) as exc:
        if not isinstance(model, (str, Path, Mapping, BackendSpec)):
            raise TypeError(
                "backend must be callable, expose predict(), or be a supported backend spec"
            ) from exc
        raise


def predict_batch(backend: BackendAdapter, inputs: Sequence[Any]) -> list[Any]:
    """Execute a batch through an adapter, normalizing the result to a list."""

    method = getattr(backend, "predict_batch", None)
    if callable(method):
        result = method(inputs)
    else:
        result = [backend.predict(item) for item in inputs]
    if isinstance(result, list):
        return result
    if isinstance(result, tuple):
        return list(result)
    try:
        if len(result) == len(inputs):
            return [result[index] for index in range(len(inputs))]
    except (TypeError, IndexError, KeyError):
        pass
    return [result]


class PyTorchBackendAdapter:
    """Optional PyTorch adapter with lazy import and CPU-safe defaults."""

    def __init__(
        self,
        model: Any,
        *,
        name: str = "pytorch",
        device: str | Any = "cpu",
        input_transform: Callable[[Any], Any] | None = None,
        output_transform: Callable[[Any], Any] | None = None,
        unpack_inputs: bool = False,
    ) -> None:
        try:
            import torch  # type: ignore
        except ImportError as exc:
            raise OptionalDependencyError(
                "PyTorchBackendAdapter requires the optional 'torch' dependency; "
                "install mlforensics[torch]."
            ) from exc
        self._torch = torch
        self.name = name
        self.device = torch.device(device)
        if isinstance(model, (str, Path)):
            path = _checked_model_path(str(model), "TorchScript")
            try:
                model = torch.jit.load(str(path), map_location=self.device)
            except Exception as exc:
                raise BackendError(f"could not load TorchScript model {path}: {exc}") from exc
        self.model = model
        self.input_transform = input_transform
        self.output_transform = output_transform
        self.unpack_inputs = unpack_inputs
        if hasattr(model, "to"):
            model.to(self.device)
        if hasattr(model, "eval"):
            model.eval()

    def _prepare(self, value: Any) -> Any:
        if self.input_transform:
            value = self.input_transform(value)
        return self._move(value)

    def _move(self, value: Any) -> Any:
        torch = self._torch
        if isinstance(value, torch.Tensor):
            return value.to(self.device)
        if isinstance(value, Mapping):
            return {key: self._move(item) for key, item in value.items()}
        if isinstance(value, (tuple, list)):
            try:
                return torch.as_tensor(value, device=self.device)
            except (TypeError, ValueError):
                converted = [self._move(item) for item in value]
                return tuple(converted) if isinstance(value, tuple) else converted
        # Numeric values and NumPy arrays are converted in the same manner as
        # torch.as_tensor, but strings and custom objects pass through.
        if isinstance(value, (int, float, complex)):
            return torch.as_tensor(value, device=self.device)
        try:
            return torch.as_tensor(value, device=self.device)
        except (TypeError, ValueError):
            return value

    def predict(self, inputs: Any) -> Any:
        prepared = self._prepare(inputs)
        with self._torch.no_grad():
            if self.unpack_inputs and isinstance(prepared, Mapping):
                output = self.model(**prepared)
            elif self.unpack_inputs and isinstance(prepared, (tuple, list)):
                output = self.model(*prepared)
            else:
                output = self.model(prepared)
        return self.output_transform(output) if self.output_transform else output

    def predict_batch(self, inputs: Sequence[Any]) -> list[Any]:
        return [self.predict(item) for item in inputs]


class ONNXRuntimeBackendAdapter:
    """Optional ONNX Runtime adapter.

    ``session`` may be an existing InferenceSession or a path to an ONNX
    model.  Imports and session construction happen in ``__init__`` so merely
    importing this module never requires onnxruntime.
    """

    def __init__(
        self,
        session: Any,
        *,
        name: str = "onnxruntime",
        input_names: Sequence[str] | None = None,
        output_transform: Callable[[Any], Any] | None = None,
        providers: Sequence[str] | None = None,
    ) -> None:
        self.name = name
        if isinstance(session, (str, Path)):
            try:
                import onnxruntime as ort  # type: ignore
            except ImportError as exc:
                raise OptionalDependencyError(
                    "ONNXRuntimeBackendAdapter requires the optional 'onnxruntime' "
                    "dependency; install mlforensics[onnx]."
                ) from exc
            path = _checked_model_path(str(session), "ONNX")
            kwargs = {"providers": list(providers)} if providers else {}
            try:
                session = ort.InferenceSession(str(path), **kwargs)
            except Exception as exc:
                raise BackendError(f"could not load ONNX model {path}: {exc}") from exc
        elif not hasattr(session, "run"):
            raise TypeError("ONNX session must expose run()")
        self.session = session
        self.input_names = list(input_names) if input_names else None
        self.output_transform = output_transform

    def _feed(self, inputs: Any) -> dict[str, Any]:
        names = self.input_names or [item.name for item in self.session.get_inputs()]
        if isinstance(inputs, Mapping):
            missing = [name for name in names if name not in inputs]
            extra = [name for name in inputs if name not in names]
            if missing or extra:
                raise ValueError(f"ONNX input names differ; missing={missing}, unexpected={extra}")
            feed = {name: inputs[name] for name in names}
            return {key: self._numpy_value(value) for key, value in feed.items()}
        if len(names) != 1 and not isinstance(inputs, (tuple, list)):
            raise ValueError("ONNX input requires a mapping or one value per model input")
        values = list(inputs) if isinstance(inputs, (tuple, list)) and len(names) != 1 else [inputs]
        return {key: self._numpy_value(value) for key, value in zip(names, values)}

    @staticmethod
    def _numpy_value(value: Any) -> Any:
        if hasattr(value, "detach"):
            value = value.detach().cpu()
        if hasattr(value, "numpy"):
            try:
                return value.numpy()
            except (TypeError, RuntimeError):
                pass
        try:
            import numpy as np  # type: ignore

            return value if isinstance(value, np.ndarray) else np.asarray(value)
        except ImportError:
            return value

    def predict(self, inputs: Any) -> Any:
        outputs = self.session.run(None, self._feed(inputs))
        value: Any = outputs[0] if len(outputs) == 1 else outputs
        return self.output_transform(value) if self.output_transform else value

    def predict_batch(self, inputs: Sequence[Any]) -> list[Any]:
        return [self.predict(item) for item in inputs]


TorchBackendAdapter = PyTorchBackendAdapter
TorchAdapter = PyTorchBackendAdapter
OnnxRuntimeBackendAdapter = ONNXRuntimeBackendAdapter
ONNXAdapter = ONNXRuntimeBackendAdapter
