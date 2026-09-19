"""Safe eager-PyTorch export helpers for parity workflows.

The helpers in this module intentionally use only ``torch.export.save`` and
``torch.export.load`` for exported programs.  They never call ``torch.load``
and do not accept ordinary ``.pt``/pickle model files as an eager model
loading mechanism.

The public functions are dependency-light at import time: PyTorch is imported
only when an export or load operation is requested.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .backends import BackendError, OptionalDependencyError
from .inputs import InputSpec

_IDENTIFIER = re.compile(r"[^A-Za-z0-9_]+")


@dataclass(frozen=True)
class ExportInputBundle:
    """Normalized positional and keyword inputs for ``torch.export``.

    ``torch.export.export`` takes positional arguments as a tuple and keyword
    arguments as a mapping.  Keeping the two pieces explicit prevents a
    mapping used as one structured input from being accidentally interpreted
    as several named model inputs.
    """

    args: tuple[Any, ...] = ()
    kwargs: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "args", tuple(self.args))
        object.__setattr__(self, "kwargs", dict(self.kwargs or {}))


def prepare_export_inputs(
    example_inputs: Any,
    *,
    input_names: Sequence[str] | None = None,
    structured_inputs: bool = False,
) -> ExportInputBundle:
    """Normalize common input forms for eager export.

    A mapping is treated as named keyword inputs by default.  Set
    ``structured_inputs=True`` when the model's single argument is itself a
    mapping/list structure.  A sequence is positional unless ``input_names``
    is supplied, in which case it becomes keyword inputs in that order.
    """

    names = _normalize_names(input_names)
    if isinstance(example_inputs, ExportInputBundle):
        if names or structured_inputs:
            raise ValueError("input_names/structured_inputs cannot modify an ExportInputBundle")
        return example_inputs

    if structured_inputs:
        if names:
            raise ValueError("structured_inputs cannot be combined with input_names")
        return ExportInputBundle((example_inputs,))

    if isinstance(example_inputs, Mapping):
        values = dict(example_inputs)
        if names:
            _validate_named_values(values, names)
            values = {name: values[name] for name in names}
        return ExportInputBundle((), values)

    if names:
        values = _as_sequence(example_inputs)
        if len(values) != len(names):
            raise ValueError(
                f"named model input count differs from the example: expected {len(names)}, "
                f"received {len(values)}"
            )
        return ExportInputBundle((), dict(zip(names, values)))

    if isinstance(example_inputs, (tuple, list)):
        return ExportInputBundle(tuple(example_inputs))
    return ExportInputBundle((example_inputs,))


def dynamic_shapes_from_specs(
    specs: InputSpec
    | Mapping[str, InputSpec | Mapping[str, Any]]
    | Sequence[InputSpec | Mapping[str, Any]],
    *,
    input_names: Sequence[str] | None = None,
    structured_inputs: bool = False,
) -> Any:
    """Build a ``torch.export`` dynamic-shape specification from ``InputSpec``.

    A dynamic dimension's choices become the conservative ``min``/``max`` of
    a PyTorch ``Dim``.  Dimensions without choices remain static.  The
    returned shape structure mirrors :func:`prepare_export_inputs`: a tuple
    for positional inputs and a mapping for named keyword inputs.
    """

    torch = _torch()
    dim_factory = getattr(torch, "export", None)
    dim_factory = getattr(dim_factory, "Dim", None)
    if not callable(dim_factory):
        raise BackendError("this PyTorch version does not provide torch.export.Dim")

    normalized = _normalize_specs(specs)
    names = _normalize_names(input_names)
    dimensions: dict[str, Any] = {}
    if structured_inputs:
        if names or len(normalized) != 1:
            raise ValueError("structured dynamic shapes require exactly one input spec")
        return (_shape_spec(normalized[0], dim_factory, dimensions),)

    if names:
        by_name = {spec.name: spec for spec in normalized}
        missing = [name for name in names if name not in by_name]
        if missing:
            raise ValueError(f"dynamic-shape specs are missing named inputs: {missing}")
        return {name: _shape_spec(by_name[name], dim_factory, dimensions) for name in names}

    if isinstance(specs, Mapping):
        return {spec.name: _shape_spec(spec, dim_factory, dimensions) for spec in normalized}

    if len(normalized) == 1:
        # One spec still represents one positional argument.  This is useful
        # for a tensor with a dynamic batch dimension.
        return (_shape_spec(normalized[0], dim_factory, dimensions),)
    return tuple(_shape_spec(spec, dim_factory, dimensions) for spec in normalized)


def export_torch_model(
    model: Any,
    example_inputs: Any,
    path: str | Path | None = None,
    *,
    input_names: Sequence[str] | None = None,
    structured_inputs: bool = False,
    dynamic_shapes: Any = None,
    input_specs: InputSpec
    | Mapping[str, InputSpec | Mapping[str, Any]]
    | Sequence[InputSpec | Mapping[str, Any]]
    | None = None,
    strict: bool = False,
    eval_mode: bool = True,
    metadata: Mapping[str, Any] | None = None,
) -> Any:
    """Export an eager ``torch.nn.Module`` without pickle-based model loading.

    ``example_inputs`` may be a tensor/scalar, a positional tuple/list, a
    mapping of named inputs, or a structured single input.  If ``path`` is
    provided the resulting ``ExportedProgram`` is saved in PyTorch's export
    format (normally using a ``.pt2`` or ``.pte`` suffix) and is still
    returned to the caller.
    """

    torch = _torch()
    export = getattr(torch, "export", None)
    exporter = getattr(export, "export", None)
    if not callable(exporter):
        raise BackendError("this PyTorch version does not provide torch.export.export")
    if not callable(getattr(model, "forward", None)):
        raise TypeError("model must expose a callable forward() method")

    bundle = prepare_export_inputs(
        example_inputs,
        input_names=input_names,
        structured_inputs=structured_inputs,
    )
    if dynamic_shapes is not None and input_specs is not None:
        raise ValueError("provide dynamic_shapes or input_specs, not both")
    shape_spec = dynamic_shapes
    if input_specs is not None:
        shape_spec = dynamic_shapes_from_specs(
            input_specs,
            input_names=input_names,
            structured_inputs=structured_inputs,
        )

    was_training = getattr(model, "training", None)
    if eval_mode and hasattr(model, "eval"):
        model.eval()
    try:
        try:
            exported = exporter(
                model,
                bundle.args,
                bundle.kwargs,
                dynamic_shapes=shape_spec,
                strict=bool(strict),
            )
        except TypeError as exc:
            # Older supported PyTorch releases may not expose the newer
            # ``strict`` keyword.  Do not hide genuine export failures.
            if "strict" not in str(exc) or strict:
                raise
            exported = exporter(
                model,
                bundle.args,
                bundle.kwargs,
                dynamic_shapes=shape_spec,
            )
    except Exception as exc:
        if isinstance(exc, (BackendError, OptionalDependencyError, ValueError, TypeError)):
            raise
        raise BackendError(f"could not export eager PyTorch model: {exc}") from exc
    finally:
        if eval_mode and was_training is True and hasattr(model, "train"):
            model.train()

    if path is not None:
        save_exported_program(exported, path, metadata=metadata)
    return exported


def save_exported_program(
    exported_program: Any,
    path: str | Path,
    *,
    metadata: Mapping[str, Any] | None = None,
) -> Path:
    """Save an exported program using ``torch.export.save`` only."""

    torch = _torch()
    saver = getattr(getattr(torch, "export", None), "save", None)
    if not callable(saver):
        raise BackendError("this PyTorch version does not provide torch.export.save")
    destination = Path(path).expanduser()
    destination.parent.mkdir(parents=True, exist_ok=True)
    extra_files: dict[str, str] | None = None
    if metadata is not None:
        try:
            encoded = json.dumps(dict(metadata), sort_keys=True, allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise ValueError("export metadata must be JSON-compatible") from exc
        extra_files = {"mlforensics-metadata.json": encoded}
    try:
        saver(exported_program, destination, extra_files=extra_files)
    except Exception as exc:
        raise BackendError(f"could not save eager PyTorch export {destination}: {exc}") from exc
    return destination


def load_exported_program(
    path: str | Path,
    *,
    with_metadata: bool = False,
) -> Any:
    """Load a safe eager export via ``torch.export.load``.

    This function deliberately has no fallback to ``torch.load``.  Ordinary
    pickle-backed PyTorch checkpoints are unsupported by this API.
    """

    torch = _torch()
    loader = getattr(getattr(torch, "export", None), "load", None)
    if not callable(loader):
        raise BackendError("this PyTorch version does not provide torch.export.load")
    source = Path(path).expanduser()
    if not source.exists():
        raise FileNotFoundError(f"eager PyTorch export does not exist: {source}")
    if not source.is_file():
        raise ValueError(f"eager PyTorch export path is not a file: {source}")
    extra_files = {"mlforensics-metadata.json": ""} if with_metadata else None
    try:
        loaded = loader(source, extra_files=extra_files)
    except Exception as exc:
        raise BackendError(
            f"could not load eager PyTorch export {source}; only torch.export files "
            f"are supported: {exc}"
        ) from exc
    if not with_metadata:
        return loaded
    raw = (extra_files or {}).get("mlforensics-metadata.json", "")
    if not raw:
        return loaded, {}
    try:
        metadata = json.loads(raw)
    except (TypeError, ValueError) as exc:
        raise BackendError("eager export metadata is not valid JSON") from exc
    if not isinstance(metadata, dict):
        raise BackendError("eager export metadata must contain a JSON object")
    return loaded, metadata


def load_exported_module(path: str | Path) -> Any:
    """Load an eager export and materialize its callable module wrapper."""

    return load_exported_program(path).module()


def export_eager_pytorch(
    model: Any,
    destination: str | Path,
    example_inputs: Any = None,
    *,
    example_kwargs: Mapping[str, Any] | None = None,
    input_names: Sequence[str] | None = None,
    structured_inputs: bool = False,
    dynamic_shapes: Any = None,
    input_specs: InputSpec
    | Mapping[str, InputSpec | Mapping[str, Any]]
    | Sequence[InputSpec | Mapping[str, Any]]
    | None = None,
    strict: bool = False,
    metadata: Mapping[str, Any] | None = None,
) -> Path:
    """Compatibility-shaped helper returning the saved export path."""

    if example_kwargs is not None:
        if example_inputs is None:
            bundle: Any = ExportInputBundle((), dict(example_kwargs))
        else:
            bundle = ExportInputBundle(
                prepare_export_inputs(example_inputs).args,
                dict(example_kwargs),
            )
    else:
        bundle = example_inputs
    export_torch_model(
        model,
        bundle,
        destination,
        input_names=input_names,
        structured_inputs=structured_inputs,
        dynamic_shapes=dynamic_shapes,
        input_specs=input_specs,
        strict=strict,
        metadata=metadata,
    )
    return Path(destination).expanduser()


# Short aliases make the intended format clear at call sites and preserve a
# compact API for parity scripts.
export_eager_torch = export_torch_model
export_model = export_torch_model
export_pytorch = export_torch_model
save_eager_export = save_exported_program
load_eager_export = load_exported_program
load_torch_export = load_exported_program


__all__ = [
    "ExportInputBundle",
    "dynamic_shapes_from_specs",
    "export_eager_pytorch",
    "export_eager_torch",
    "export_model",
    "export_pytorch",
    "export_torch_model",
    "load_eager_export",
    "load_exported_module",
    "load_exported_program",
    "load_torch_export",
    "prepare_export_inputs",
    "save_eager_export",
    "save_exported_program",
]


def _torch() -> Any:
    try:
        import torch  # type: ignore
    except ImportError as exc:
        raise OptionalDependencyError(
            "eager PyTorch export requires the optional 'torch' dependency; "
            "install mlforensics[torch]."
        ) from exc
    return torch


def _normalize_names(names: Sequence[str] | None) -> tuple[str, ...] | None:
    if names is None:
        return None
    if isinstance(names, str):
        names = (names,)
    normalized = tuple(str(name).strip() for name in names)
    if not normalized or any(not name for name in normalized):
        raise ValueError("input_names must contain non-empty names")
    if len(set(normalized)) != len(normalized):
        raise ValueError("input_names must be unique")
    return normalized


def _as_sequence(value: Any) -> list[Any]:
    if isinstance(value, (tuple, list)):
        return list(value)
    return [value]


def _validate_named_values(values: Mapping[str, Any], names: Sequence[str]) -> None:
    missing = [name for name in names if name not in values]
    extra = [name for name in values if name not in names]
    if missing or extra:
        raise ValueError(f"named example inputs differ; missing={missing}, unexpected={extra}")


def _normalize_specs(
    specs: InputSpec
    | Mapping[str, InputSpec | Mapping[str, Any]]
    | Sequence[InputSpec | Mapping[str, Any]],
) -> list[InputSpec]:
    if isinstance(specs, InputSpec):
        return [specs]
    if isinstance(specs, Mapping):
        result = []
        for name, value in specs.items():
            if isinstance(value, InputSpec):
                result.append(value)
            elif isinstance(value, Mapping):
                result.append(InputSpec.from_dict({"name": name, **dict(value)}))
            else:
                raise TypeError("input spec mappings must contain InputSpec or mapping values")
        return result
    result = []
    for index, value in enumerate(specs):
        if isinstance(value, InputSpec):
            result.append(value)
        elif isinstance(value, Mapping):
            result.append(InputSpec.from_dict({"name": f"input-{index}", **dict(value)}))
        else:
            raise TypeError("input specs must be InputSpec instances or mappings")
    if not result:
        raise ValueError("at least one input spec is required")
    return result


def _shape_spec(
    spec: InputSpec,
    dim_factory: Any,
    dimensions: dict[str, Any],
) -> dict[int, Any]:
    if spec.shape is None:
        raise ValueError(f"input spec {spec.name!r} needs a shape for dynamic export")
    shape: dict[int, Any] = {}
    for index, dimension in enumerate(spec.shape):
        choices = spec.dynamic_dims.get(index, spec.dynamic_dims.get(str(index), ()))
        if not choices and isinstance(dimension, str):
            choices = spec.dynamic_dims.get(dimension, ())
        if not choices:
            continue
        values = tuple(int(value) for value in choices)
        if len(set(values)) == 1:
            continue
        raw_name = dimension if isinstance(dimension, str) else f"{spec.name}_dim{index}"
        safe_name = _IDENTIFIER.sub("_", raw_name).strip("_") or f"dim{index}"
        if safe_name[0].isdigit():
            safe_name = f"dim_{safe_name}"
        if safe_name not in dimensions:
            dimensions[safe_name] = dim_factory(safe_name, min=min(values), max=max(values))
        shape[index] = dimensions[safe_name]
    return shape
