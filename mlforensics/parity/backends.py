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

    Supported kinds are ``python`` (``package.module:object``), ``onnx``,
    ``torchscript``, eager ``torch-export``, ``tensorrt``, and ``openvino``.
    General pickle-based PyTorch loading is intentionally not supported because
    loading an untrusted pickle can execute arbitrary code.
    """

    kind: str
    source: str
    name: str | None = None
    options: Mapping[str, Any] = field(default_factory=dict)


_PYTHON_TARGET = re.compile(
    r"^(?P<module>[A-Za-z_]\w*(?:\.[A-Za-z_]\w)*):"
    r"(?P<object>[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*)$"
)

_MISSING = object()


def _normalise_kind(kind: str) -> str:
    return kind.lower().strip().replace("_", "-")


def _normalise_dtype_name(value: Any) -> str:
    """Return the small, portable precision vocabulary used by the adapters."""

    if isinstance(value, str):
        name = value.lower().strip().replace(" ", "").replace("torch.", "")
        aliases = {
            "fp32": "float32",
            "float": "float32",
            "single": "float32",
            "fp16": "float16",
            "half": "float16",
            "bf16": "bfloat16",
            "bfloat": "bfloat16",
        }
        return aliases.get(name, name)
    name = str(value).lower().replace("torch.", "")
    return _normalise_dtype_name(name)


def _resolve_torch_dtype(torch: Any, value: Any) -> Any:
    if value is None:
        return None
    if not isinstance(value, str):
        # torch.dtype objects are intentionally accepted without importing or
        # depending on their concrete implementation here.
        if str(value).startswith("torch.") or type(value).__name__ == "dtype":
            return value
    name = _normalise_dtype_name(value)
    mapping = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    try:
        return mapping[name]
    except KeyError as exc:
        raise ValueError(
            "PyTorch dtype must be one of float32/fp32, float16/fp16, or bfloat16/bf16"
        ) from exc


def _sequence_items(value: Any) -> list[Any] | None:
    if isinstance(value, (tuple, list)):
        return list(value)
    return None


def _normalise_names(names: Sequence[str] | None) -> tuple[str, ...] | None:
    if names is None:
        return None
    if isinstance(names, str):
        names = (names,)
    result = tuple(str(name).strip() for name in names)
    if not result or any(not name for name in result):
        raise ValueError("input names must be non-empty")
    if len(set(result)) != len(result):
        raise ValueError("input names must be unique")
    return result


def _named_inputs(value: Any, names: Sequence[str] | None) -> Any:
    if not names or isinstance(value, Mapping):
        return value
    items = _sequence_items(value)
    if items is None:
        if len(names) == 1:
            return {names[0]: value}
        raise ValueError("named model inputs require one value per input name")
    if len(items) != len(names):
        raise ValueError(
            f"named model input count differs from the model: expected {len(names)}, "
            f"received {len(items)}"
        )
    return dict(zip(names, items))


def _invoke_model(
    model: Any,
    inputs: Any,
    *,
    unpack_inputs: bool = False,
    input_names: Sequence[str] | None = None,
    call_mode: str | None = None,
) -> Any:
    """Invoke a model with explicit, predictable single/args/kwargs modes."""

    inputs = _named_inputs(inputs, input_names)
    default_mode = "auto" if unpack_inputs else ("kwargs" if input_names else "single")
    mode = (call_mode or default_mode).lower().replace("_", "-")
    if mode in {"single", "one"}:
        return model(inputs)
    if mode in {"kwargs", "keyword", "named"}:
        if not isinstance(inputs, Mapping):
            raise ValueError("keyword model calls require a mapping of named inputs")
        return model(**dict(inputs))
    if mode in {"args", "positional", "tuple"}:
        items = _sequence_items(inputs)
        if items is None:
            raise ValueError("positional model calls require a tuple or list of inputs")
        return model(*items)
    if mode != "auto":
        raise ValueError("call_mode must be single, args, kwargs, or auto")
    if isinstance(inputs, Mapping):
        return model(**dict(inputs))
    items = _sequence_items(inputs)
    if items is not None:
        return model(*items)
    return model(inputs)


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
        kind = str(spec["kind"]).strip()
        source = str(spec["source"]).strip()
        if not kind or not source:
            raise ValueError("backend specification kind and source cannot be empty")
        return BackendSpec(
            kind,
            source,
            None if spec.get("name") is None else str(spec["name"]),
            dict(options),
        )

    source = str(spec).strip()
    if not source:
        raise ValueError("backend specification cannot be empty")
    lowered = source.lower()
    for prefix, kind in (
        ("python:", "python"),
        ("onnx:", "onnx"),
        ("torchscript:", "torchscript"),
        ("torch-export:", "torch-export"),
        ("exported:", "torch-export"),
        ("torch:", "torchscript"),
        ("pytorch:", "torchscript"),
        ("tensorrt:", "tensorrt"),
        ("trt:", "tensorrt"),
        ("openvino:", "openvino"),
        ("ov:", "openvino"),
    ):
        if lowered.startswith(prefix):
            target = source[len(prefix) :]
            if not target:
                raise ValueError(f"{kind} backend specification has an empty source")
            return BackendSpec(kind, target)
    suffix = Path(source).suffix.lower()
    if suffix == ".onnx":
        return BackendSpec("onnx", source)
    if suffix in {".pt2", ".pte", ".exported_program"}:
        return BackendSpec("torch-export", source)
    if suffix in {".pt", ".pth", ".torchscript"}:
        return BackendSpec("torchscript", source)
    if suffix in {".engine", ".plan"}:
        return BackendSpec("tensorrt", source)
    if suffix in {".xml", ".blob"}:
        return BackendSpec("openvino", source)
    if _PYTHON_TARGET.fullmatch(source):
        return BackendSpec("python", source)
    raise ValueError(
        "backend must be a callable, adapter, module:object, or a supported model path"
    )


def _checked_model_path(source: str, kind: str) -> Path:
    path = Path(source).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"{kind} model does not exist: {path}")
    if not path.is_file():
        raise ValueError(f"{kind} model path is not a file: {path}")
    return path


def _model_format_from_suffix(path: Path) -> str:
    return (
        "export" if path.suffix.lower() in {".pt2", ".pte", ".exported_program"} else "torchscript"
    )


def _torch_import() -> Any:
    try:
        import torch  # type: ignore
    except ImportError as exc:
        raise OptionalDependencyError(
            "PyTorch backend support requires the optional 'torch' dependency; "
            "install mlforensics[torch]."
        ) from exc
    return torch


def _export_args(example_inputs: Any) -> tuple[Any, ...]:
    if example_inputs is None:
        return ()
    # A tuple is the unambiguous representation of positional model inputs.
    # A list remains one structured input, which is important for token and
    # sequence models.
    return example_inputs if isinstance(example_inputs, tuple) else (example_inputs,)


def export_eager_pytorch(
    model: Any,
    destination: str | Path,
    example_inputs: Any = None,
    *,
    example_kwargs: Mapping[str, Any] | None = None,
    input_names: Sequence[str] | None = None,
    structured_inputs: bool = False,
    dynamic_shapes: Any = None,
    strict: bool = False,
) -> Path:
    """Export an eager module using PyTorch's non-pickle ``torch.export`` format.

    ``example_inputs`` is required by ``torch.export``.  Positional arguments
    are represented by a tuple; a list is treated as one structured input.
    The returned path can be passed to ``PyTorchBackendAdapter`` with
    ``model_format="export"`` or to ``load_backend`` as a ``.pt2`` path.
    """

    torch = _torch_import()
    exporter = getattr(torch, "export", None)
    export = getattr(exporter, "export", None)
    save = getattr(exporter, "save", None)
    if not callable(export) or not callable(save):
        raise BackendError("this PyTorch version does not provide torch.export.save/export")
    if example_inputs is None and not example_kwargs:
        raise ValueError("example_inputs or example_kwargs is required for eager PyTorch export")
    path = Path(destination).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    if example_kwargs is not None:
        args = _export_args(example_inputs)
        kwargs = dict(example_kwargs)
    elif structured_inputs:
        args, kwargs = (example_inputs,), {}
    elif isinstance(example_inputs, Mapping):
        args, kwargs = (), dict(example_inputs)
    elif input_names:
        values = _export_args(example_inputs)
        if len(values) != len(input_names):
            raise ValueError(
                f"named model input count differs from the example: expected {len(input_names)}, "
                f"received {len(values)}"
            )
        args, kwargs = (), dict(zip(input_names, values))
    else:
        args, kwargs = _export_args(example_inputs), {}
    try:
        exported = export(
            model,
            args=args,
            kwargs=kwargs,
            dynamic_shapes=dynamic_shapes,
            strict=strict,
        )
        save(exported, str(path))
    except Exception as exc:
        raise BackendError(f"could not export eager PyTorch model to {path}: {exc}") from exc
    return path


def export_pytorch(*args: Any, **kwargs: Any) -> Path:
    """Backward-friendly name for :func:`export_eager_pytorch`."""

    return export_eager_pytorch(*args, **kwargs)


def export_torchscript(
    model: Any,
    destination: str | Path,
    example_inputs: Any = None,
    *,
    example_kwargs: Mapping[str, Any] | None = None,
    method: str = "script",
    strict: bool = False,
) -> Path:
    """Export a module to TorchScript without serializing a Python object pickle."""

    torch = _torch_import()
    path = Path(destination).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    selected = method.lower().strip()
    try:
        if selected in {"script", "jit-script"}:
            scripted = torch.jit.script(model)
        elif selected in {"trace", "jit-trace"}:
            if example_inputs is None:
                raise ValueError("example_inputs is required for TorchScript tracing")
            scripted = torch.jit.trace(
                model,
                _export_args(example_inputs),
                strict=strict,
                example_kwarg_inputs=dict(example_kwargs or {}) or None,
            )
        else:
            raise ValueError("TorchScript export method must be script or trace")
        torch.jit.save(scripted, str(path))
    except BackendError:
        raise
    except Exception as exc:
        raise BackendError(f"could not export TorchScript model to {path}: {exc}") from exc
    return path


def load_eager_pytorch(source: str | Path, **adapter_options: Any) -> PyTorchBackendAdapter:
    """Load a safe ``torch.export`` artifact through the regular adapter."""

    return PyTorchBackendAdapter(source, model_format="export", **adapter_options)


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
    kind = _normalise_kind(parsed.kind)
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
        merged_options.setdefault("model_format", "torchscript")
        return PyTorchBackendAdapter(
            path,
            name=_as_name(backend_name, "pytorch"),
            **merged_options,
        )
    if kind in {"torch-export", "torch-eager", "eager", "exported", "exported-program"}:
        path = _checked_model_path(parsed.source, "eager PyTorch")
        merged_options.setdefault("model_format", "export")
        return PyTorchBackendAdapter(path, name=_as_name(backend_name, "pytorch"), **merged_options)
    if kind in {"tensorrt", "trt", "tensor-rt"}:
        path = _checked_model_path(parsed.source, "TensorRT")
        return TensorRTBackendAdapter(
            path, name=_as_name(backend_name, "tensorrt"), **merged_options
        )
    if kind in {"openvino", "open-vino", "ov"}:
        path = _checked_model_path(parsed.source, "OpenVINO")
        return OpenVINOBackendAdapter(
            path, name=_as_name(backend_name, "openvino"), **merged_options
        )
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
    unpack_inputs: bool = False
    input_names: Sequence[str] | None = None
    call_mode: str | None = None
    stateful: bool = False
    state_input_name: str | None = "state"
    state_output_key: str | None = None
    state_output_index: int | None = None
    return_state: bool = False
    initial_state: Any = field(default=_MISSING, repr=False)
    _state: Any = field(default=_MISSING, init=False, repr=False)

    def __post_init__(self) -> None:
        self.input_names = _normalise_names(self.input_names)
        self._state = self.initial_state

    @property
    def state(self) -> Any:
        return None if self._state is _MISSING else self._state

    def reset_state(self, state: Any = _MISSING) -> None:
        """Reset adapter-managed state and call a model reset hook when present."""

        self._state = state if state is not _MISSING else self.initial_state
        reset = getattr(self.model, "reset_state", None)
        if callable(reset):
            reset()

    def _stateful_inputs(self, value: Any) -> Any:
        if not self.stateful or self.state_input_name is None or self._state is _MISSING:
            return value
        value = _named_inputs(value, self.input_names)
        if isinstance(value, Mapping):
            result = dict(value)
            result.setdefault(self.state_input_name, self._state)
            return result
        items = _sequence_items(value)
        if items is None:
            items = [value]
        items.append(self._state)
        return tuple(items) if isinstance(value, tuple) else items

    def _capture_state(self, output: Any) -> Any:
        if not self.stateful:
            return output
        if self.state_output_key is not None:
            if not isinstance(output, Mapping) or self.state_output_key not in output:
                raise BackendError(f"state output key is missing: {self.state_output_key}")
            self._state = output[self.state_output_key]
            if not self.return_state:
                output = {
                    key: value for key, value in output.items() if key != self.state_output_key
                }
        elif self.state_output_index is not None or isinstance(output, (tuple, list)):
            if not isinstance(output, (tuple, list)):
                raise BackendError("state output index requires a tuple or list model output")
            state_index = 1 if self.state_output_index is None else self.state_output_index
            try:
                self._state = output[state_index]
            except IndexError as exc:
                raise BackendError("state output index is outside the model output") from exc
            if not self.return_state:
                remaining = [value for index, value in enumerate(output) if index != state_index]
                output = type(output)(remaining)
                if len(remaining) == 1:
                    output = remaining[0]
        return output

    def predict(self, inputs: Any) -> Any:
        prepared = self.prepare_input(inputs) if self.prepare_input else inputs
        prepared = self._stateful_inputs(prepared)
        effective_mode = self.call_mode
        if effective_mode is None and self.stateful:
            effective_mode = "kwargs" if isinstance(prepared, Mapping) else "args"
        output = _invoke_model(
            self.model,
            prepared,
            unpack_inputs=self.unpack_inputs,
            input_names=self.input_names,
            call_mode=effective_mode,
        )
        output = self._capture_state(output)
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
    """Optional PyTorch adapter with lazy import and CPU-safe defaults.

    Paths are loaded only through ``torch.jit.load`` (TorchScript) or
    ``torch.export.load`` (the safe eager-export format).  In particular this
    adapter never calls ``torch.load`` and therefore does not load arbitrary
    Python pickle state dictionaries.
    """

    def __init__(
        self,
        model: Any,
        *,
        name: str = "pytorch",
        device: str | Any = "cpu",
        input_transform: Callable[[Any], Any] | None = None,
        output_transform: Callable[[Any], Any] | None = None,
        unpack_inputs: bool = False,
        input_names: Sequence[str] | None = None,
        call_mode: str | None = None,
        structured_inputs: bool = False,
        dtype: Any = None,
        input_dtype: Any = None,
        model_dtype: Any = None,
        precision: Any = None,
        stateful: bool = False,
        state_input_name: str | None = "state",
        state_output_key: str | None = None,
        state_output_index: int | None = None,
        return_state: bool = False,
        initial_state: Any = _MISSING,
        model_format: str | None = None,
        format: str | None = None,
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
        selected_dtype = input_dtype if input_dtype is not None else (dtype or precision)
        if selected_dtype is None and model_dtype is not None:
            selected_dtype = model_dtype
        self.input_dtype = _resolve_torch_dtype(torch, selected_dtype)
        self.dtype = self.input_dtype
        selected_model_dtype = model_dtype
        if selected_model_dtype is None and input_dtype is None:
            selected_model_dtype = dtype or precision
        self.model_dtype = _resolve_torch_dtype(torch, selected_model_dtype)
        self.input_names = _normalise_names(input_names)
        self.unpack_inputs = unpack_inputs
        self.call_mode = call_mode
        self.structured_inputs = structured_inputs
        self.stateful = stateful
        self.state_input_name = state_input_name
        self.state_output_key = state_output_key
        self.state_output_index = state_output_index
        self.return_state = return_state
        self.initial_state = initial_state
        self._state = initial_state
        requested_format = model_format or format
        self._exported_program = False
        if isinstance(model, (str, Path)):
            path_kind = (
                "eager PyTorch"
                if requested_format in {"export", "eager", "torch-export", "exported"}
                else "TorchScript"
            )
            path = _checked_model_path(str(model), path_kind)
            selected_format = (requested_format or _model_format_from_suffix(path)).lower()
            try:
                if selected_format in {"export", "eager", "torch-export", "exported"}:
                    exporter = getattr(torch, "export", None)
                    loader = getattr(exporter, "load", None)
                    if not callable(loader):
                        raise BackendError(
                            "this PyTorch version does not provide torch.export.load; "
                            "install PyTorch 2.0 or newer"
                        )
                    exported = loader(str(path))
                    self._exported_program = True
                    model = (
                        exported.module()
                        if callable(getattr(exported, "module", None))
                        else exported
                    )
                elif selected_format in {"torchscript", "jit", "script", "auto"}:
                    model = torch.jit.load(str(path), map_location=self.device)
                else:
                    raise ValueError("model_format must be torchscript or export")
            except BackendError:
                raise
            except Exception as exc:
                raise BackendError(f"could not load PyTorch model {path}: {exc}") from exc
        elif requested_format in {"export", "eager", "torch-export", "exported"}:
            module = getattr(model, "module", None)
            if callable(module):
                model = module()
                self._exported_program = True
        self.model = model
        self.input_transform = input_transform
        self.output_transform = output_transform
        if hasattr(model, "to"):
            model.to(self.device)
            if self.model_dtype is not None:
                model.to(dtype=self.model_dtype)
        if hasattr(model, "eval"):
            try:
                model.eval()
            except NotImplementedError:
                # ExportedProgram's generated module is already in inference
                # form on PyTorch versions where eval() is not implemented.
                pass

    @property
    def state(self) -> Any:
        return None if self._state is _MISSING else self._state

    def reset_state(self, state: Any = _MISSING) -> None:
        self._state = state if state is not _MISSING else self.initial_state
        reset = getattr(self.model, "reset_state", None)
        if callable(reset):
            reset()

    def _prepare(self, value: Any, *, preserve_sequences: bool = False) -> Any:
        if self.input_transform:
            value = self.input_transform(value)
        return self._move(value, preserve_sequences=preserve_sequences)

    def _tensor(self, value: Any) -> Any:
        tensor = self._torch.as_tensor(value, device=self.device)
        if self.input_dtype is not None and tensor.dtype != self._torch.bool:
            tensor = tensor.to(dtype=self.input_dtype)
        return tensor

    def _move(self, value: Any, *, preserve_sequences: bool = False) -> Any:
        torch = self._torch
        if isinstance(value, torch.Tensor):
            result = value.to(self.device)
            if self.input_dtype is not None and result.dtype != torch.bool:
                result = result.to(dtype=self.input_dtype)
            return result
        if isinstance(value, Mapping):
            return {key: self._move(item) for key, item in value.items()}
        if isinstance(value, (tuple, list)):
            if preserve_sequences:
                converted = [self._move(item) for item in value]
                return tuple(converted) if isinstance(value, tuple) else converted
            try:
                return self._tensor(value)
            except (TypeError, ValueError):
                converted = [self._move(item) for item in value]
                return tuple(converted) if isinstance(value, tuple) else converted
        # Numeric values and NumPy arrays are converted in the same manner as
        # torch.as_tensor, but strings and custom objects pass through.
        if isinstance(value, (int, float, complex)):
            return self._tensor(value)
        try:
            return self._tensor(value)
        except (TypeError, ValueError):
            return value

    def _stateful_inputs(self, value: Any) -> Any:
        if not self.stateful or self.state_input_name is None or self._state is _MISSING:
            return value
        value = _named_inputs(value, self.input_names)
        if isinstance(value, Mapping):
            result = dict(value)
            result.setdefault(self.state_input_name, self._move(self._state))
            return result
        items = _sequence_items(value)
        if items is None:
            items = [value]
        items.append(self._move(self._state))
        return tuple(items) if isinstance(value, tuple) else items

    def _capture_state(self, output: Any) -> Any:
        if not self.stateful:
            return output
        if self.state_output_key is not None:
            if not isinstance(output, Mapping) or self.state_output_key not in output:
                raise BackendError(f"state output key is missing: {self.state_output_key}")
            self._state = output[self.state_output_key]
            if not self.return_state:
                output = {
                    key: value for key, value in output.items() if key != self.state_output_key
                }
        elif self.state_output_index is not None or isinstance(output, (tuple, list)):
            if not isinstance(output, (tuple, list)):
                raise BackendError("state output index requires a tuple or list model output")
            state_index = 1 if self.state_output_index is None else self.state_output_index
            try:
                self._state = output[state_index]
            except IndexError as exc:
                raise BackendError("state output index is outside the model output") from exc
            if not self.return_state:
                remaining = [value for i, value in enumerate(output) if i != state_index]
                output = type(output)(remaining)
                if len(remaining) == 1:
                    output = remaining[0]
        return output

    def predict(self, inputs: Any) -> Any:
        preserve_sequences = bool(
            self.unpack_inputs or self.input_names or self.call_mode in {"args", "kwargs", "auto"}
        )
        prepared = self._prepare(inputs, preserve_sequences=preserve_sequences)
        prepared = _named_inputs(prepared, self.input_names)
        prepared = self._stateful_inputs(prepared)
        effective_mode = self.call_mode
        if effective_mode is None and self.stateful:
            effective_mode = "kwargs" if isinstance(prepared, Mapping) else "args"
        with self._torch.no_grad():
            try:
                if self._exported_program and isinstance(prepared, Mapping):
                    # Exported modules retain the positional/keyword tree
                    # from export. Most positional exports reject kwargs
                    # even when their source names are known.
                    values = (
                        [prepared[name] for name in self.input_names]
                        if self.input_names
                        else list(prepared.values())
                    )
                    output = self.model(*values)
                else:
                    output = _invoke_model(
                        self.model,
                        prepared,
                        unpack_inputs=self.unpack_inputs,
                        input_names=self.input_names,
                        call_mode=effective_mode,
                    )
            except (TypeError, ValueError) as exc:
                if not (self._exported_program and isinstance(prepared, Mapping)):
                    raise
                if "keyword" not in str(exc).lower() and "argument" not in str(exc).lower():
                    raise
                output = self.model(**dict(prepared))
        output = self._capture_state(output)
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
        input_dtype: Any = None,
        dtype: Any = None,
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
        self.input_names = list(_normalise_names(input_names) or ()) or None
        self.output_transform = output_transform
        self.input_dtype = input_dtype if input_dtype is not None else dtype

    def _feed(self, inputs: Any) -> dict[str, Any]:
        names = self.input_names or [item.name for item in self.session.get_inputs()]
        if not names or any(not isinstance(name, str) or not name for name in names):
            raise ValueError("ONNX session must expose one or more named inputs")
        if isinstance(inputs, Mapping):
            missing = [name for name in names if name not in inputs]
            extra = [name for name in inputs if name not in names]
            if missing or extra:
                raise ValueError(f"ONNX input names differ; missing={missing}, unexpected={extra}")
            feed = {name: inputs[name] for name in names}
            return {
                key: self._numpy_value(value, dtype=self.input_dtype) for key, value in feed.items()
            }
        if len(names) == 1:
            values = [inputs]
        else:
            if not isinstance(inputs, (tuple, list)):
                raise ValueError("ONNX input requires a mapping or one value per model input")
            values = list(inputs)
            if len(values) != len(names):
                raise ValueError(
                    "ONNX input count differs from the model: "
                    f"expected {len(names)}, received {len(values)}"
                )
        return {
            key: self._numpy_value(value, dtype=self.input_dtype)
            for key, value in zip(names, values)
        }

    @staticmethod
    def _numpy_value(value: Any, *, dtype: Any = None) -> Any:
        if hasattr(value, "detach"):
            value = value.detach().cpu()
        if hasattr(value, "numpy"):
            try:
                return value.numpy()
            except (TypeError, RuntimeError):
                pass
        try:
            import numpy as np  # type: ignore

            if dtype is not None:
                dtype_name = _normalise_dtype_name(dtype)
                # NumPy has no portable bfloat16 dtype; float32 is the
                # lossless interchange representation for runtimes that
                # expose BF16 at the model boundary.
                dtype_name = "float32" if dtype_name == "bfloat16" else dtype_name
                return np.asarray(value, dtype=dtype_name)
            return value if isinstance(value, np.ndarray) else np.asarray(value)
        except ImportError:
            return value

    def predict(self, inputs: Any) -> Any:
        outputs = self.session.run(None, self._feed(inputs))
        value: Any = outputs[0] if len(outputs) == 1 else outputs
        return self.output_transform(value) if self.output_transform else value

    def predict_batch(self, inputs: Sequence[Any]) -> list[Any]:
        return [self.predict(item) for item in inputs]


class TensorRTBackendAdapter:
    """Lazy TensorRT adapter.

    The adapter accepts a TensorRT engine, an engine-like test double exposing
    ``infer``/``predict``, or a serialized ``.engine``/``.plan`` path.  The
    path form only imports TensorRT when it is actually selected.  Deployments
    with a CUDA-specific execution wrapper can pass that wrapper as ``engine``
    and keep the parity package independent of CUDA Python packages.
    """

    def __init__(
        self,
        engine: Any,
        *,
        name: str = "tensorrt",
        input_names: Sequence[str] | None = None,
        output_names: Sequence[str] | None = None,
        input_transform: Callable[[Any], Any] | None = None,
        output_transform: Callable[[Any], Any] | None = None,
    ) -> None:
        self.name = name
        self.input_names = _normalise_names(input_names)
        self.output_names = _normalise_names(output_names)
        self.input_transform = input_transform
        self.output_transform = output_transform
        self._runtime = None
        if isinstance(engine, (str, Path)):
            try:
                import tensorrt as trt  # type: ignore
            except ImportError as exc:
                raise OptionalDependencyError(
                    "TensorRTBackendAdapter requires the optional 'tensorrt' dependency; "
                    "install TensorRT and its CUDA runtime."
                ) from exc
            path = _checked_model_path(str(engine), "TensorRT")
            try:
                logger = trt.Logger(trt.Logger.WARNING)
                self._runtime = trt.Runtime(logger)
                with path.open("rb") as handle:
                    engine = self._runtime.deserialize_cuda_engine(handle.read())
                if engine is None:
                    raise BackendError(f"TensorRT could not deserialize engine {path}")
            except BackendError:
                raise
            except Exception as exc:
                raise BackendError(f"could not load TensorRT engine {path}: {exc}") from exc
        self.engine = engine
        self.context = self._create_context(engine)
        if self.input_names is None:
            self.input_names = self._discover_io_names(engine, want_inputs=True)
        if self.output_names is None:
            self.output_names = self._discover_io_names(engine, want_inputs=False)

    @staticmethod
    def _create_context(engine: Any) -> Any:
        create = getattr(engine, "create_execution_context", None)
        return create() if callable(create) else engine

    @staticmethod
    def _discover_io_names(engine: Any, *, want_inputs: bool) -> tuple[str, ...] | None:
        num = getattr(engine, "num_io_tensors", None)
        get_name = getattr(engine, "get_tensor_name", None)
        get_mode = getattr(engine, "get_tensor_mode", None)
        if isinstance(num, int) and callable(get_name) and callable(get_mode):
            names = []
            for index in range(num):
                name = str(get_name(index))
                mode = str(get_mode(name)).lower()
                if ("input" in mode) is want_inputs:
                    names.append(name)
            return tuple(names)
        return None

    def _feed(self, inputs: Any) -> dict[str, Any]:
        names = self.input_names
        if names:
            if isinstance(inputs, Mapping):
                missing = [name for name in names if name not in inputs]
                extra = [name for name in inputs if name not in names]
                if missing or extra:
                    raise ValueError(
                        f"TensorRT input names differ; missing={missing}, unexpected={extra}"
                    )
                values = [inputs[name] for name in names]
            elif len(names) == 1:
                values = [inputs]
            else:
                values = _sequence_items(inputs)
                if values is None or len(values) != len(names):
                    raise ValueError(f"TensorRT expects {len(names)} named inputs")
            return {
                name: ONNXRuntimeBackendAdapter._numpy_value(value)
                for name, value in zip(names, values)
            }
        if isinstance(inputs, Mapping):
            return {
                str(key): ONNXRuntimeBackendAdapter._numpy_value(value)
                for key, value in inputs.items()
            }
        values = _sequence_items(inputs)
        if values is None:
            values = [inputs]
        return {
            str(index): ONNXRuntimeBackendAdapter._numpy_value(value)
            for index, value in enumerate(values)
        }

    def predict(self, inputs: Any) -> Any:
        value = self.input_transform(inputs) if self.input_transform else inputs
        feed = self._feed(value)
        infer = getattr(self.context, "infer", None) or getattr(self.context, "predict", None)
        if callable(infer):
            output = infer(feed)
        else:
            run = getattr(self.context, "run", None)
            if callable(run):
                output = run(feed)
            else:
                execute = getattr(self.context, "execute", None)
                if not callable(execute):
                    raise BackendError(
                        "TensorRT engine requires an infer(), predict(), run(), or "
                        "execute() wrapper"
                    )
                output = execute(feed)
        if isinstance(output, Mapping) and self.output_names:
            ordered = [output[name] for name in self.output_names if name in output]
            output = ordered[0] if len(ordered) == 1 else ordered
        return self.output_transform(output) if self.output_transform else output

    def predict_batch(self, inputs: Sequence[Any]) -> list[Any]:
        return [self.predict(item) for item in inputs]


class OpenVINOBackendAdapter:
    """Lazy OpenVINO adapter for compiled models or IR model paths."""

    def __init__(
        self,
        model: Any,
        *,
        name: str = "openvino",
        device: str = "CPU",
        input_names: Sequence[str] | None = None,
        output_transform: Callable[[Any], Any] | None = None,
    ) -> None:
        self.name = name
        self.device = device
        self.output_transform = output_transform
        self._core = None
        if isinstance(model, (str, Path)):
            try:
                try:
                    from openvino import Core  # type: ignore
                except ImportError:
                    from openvino.runtime import Core  # type: ignore
            except ImportError as exc:
                raise OptionalDependencyError(
                    "OpenVINOBackendAdapter requires the optional 'openvino' dependency; "
                    "install openvino."
                ) from exc
            path = _checked_model_path(str(model), "OpenVINO")
            try:
                self._core = Core()
                read_model = self._core.read_model(str(path))
                model = self._core.compile_model(read_model, device)
            except Exception as exc:
                raise BackendError(f"could not load OpenVINO model {path}: {exc}") from exc
        if not callable(model) and not any(
            callable(getattr(model, attr, None)) for attr in ("infer", "infer_new_request")
        ):
            raise TypeError("OpenVINO model must be callable or expose infer()")
        self.model = model
        self.input_names = _normalise_names(input_names) or self._discover_input_names(model)

    @staticmethod
    def _discover_input_names(model: Any) -> tuple[str, ...] | None:
        inputs = getattr(model, "inputs", None)
        if callable(inputs):
            inputs = inputs()
        if inputs is None:
            return None
        names = []
        try:
            for item in inputs:
                any_name = getattr(item, "any_name", None)
                if callable(any_name):
                    any_name = any_name()
                names.append(str(any_name or getattr(item, "name", item)))
        except TypeError:
            return None
        return tuple(names) or None

    def _feed(self, inputs: Any) -> Any:
        names = self.input_names
        if not names:
            if isinstance(inputs, Mapping):
                return dict(inputs)
            return inputs
        if isinstance(inputs, Mapping):
            missing = [name for name in names if name not in inputs]
            extra = [name for name in inputs if name not in names]
            if missing or extra:
                raise ValueError(
                    f"OpenVINO input names differ; missing={missing}, unexpected={extra}"
                )
            return {name: ONNXRuntimeBackendAdapter._numpy_value(inputs[name]) for name in names}
        if len(names) == 1:
            values = [inputs]
        else:
            values = _sequence_items(inputs)
            if values is None or len(values) != len(names):
                raise ValueError(f"OpenVINO expects {len(names)} named inputs")
        return {
            name: ONNXRuntimeBackendAdapter._numpy_value(value)
            for name, value in zip(names, values)
        }

    def predict(self, inputs: Any) -> Any:
        feed = self._feed(inputs)
        infer = getattr(self.model, "infer_new_request", None) or getattr(self.model, "infer", None)
        if callable(infer):
            output = infer(feed)
        elif callable(self.model):
            output = self.model(feed)
        else:
            raise BackendError("OpenVINO model is not executable")
        if isinstance(output, Mapping) and len(output) == 1:
            output = next(iter(output.values()))
        return self.output_transform(output) if self.output_transform else output

    def predict_batch(self, inputs: Sequence[Any]) -> list[Any]:
        return [self.predict(item) for item in inputs]


TorchBackendAdapter = PyTorchBackendAdapter
TorchAdapter = PyTorchBackendAdapter
OnnxRuntimeBackendAdapter = ONNXRuntimeBackendAdapter
ONNXAdapter = ONNXRuntimeBackendAdapter
TensorRTAdapter = TensorRTBackendAdapter
TrtBackendAdapter = TensorRTBackendAdapter
TRTBackendAdapter = TensorRTBackendAdapter
TensorRTBackend = TensorRTBackendAdapter
OpenVINOAdapter = OpenVINOBackendAdapter
OpenvinoBackendAdapter = OpenVINOBackendAdapter
OpenVINOBackend = OpenVINOBackendAdapter
