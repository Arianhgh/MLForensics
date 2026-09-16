"""Lazy state codecs used by replay and capture.

The core registry has no framework imports.  Codecs are selected by runtime
capability and encode arrays/tensors as raw bytes plus explicit dtype/shape/
device metadata, so a replay reader can tell exactly what was restored.

Nested application state is encoded as a typed tree: integer mapping keys,
tuples, and binary leaves keep their identity instead of being silently
coerced into JSON lists or string-key objects.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

CONTAINER_CODEC = "mlforensics.container"
STATE_TREE_KEY = "mlforensics.state"
STATE_TREE_VERSION = 1
_JSON_ATOMS = (type(None), str, int, float, bool)


@dataclass(frozen=True)
class EncodedState:
    """Result of encoding one state value."""

    codec: str
    value: Any = None
    payload: bytes | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def marker(self, digest: str | None = None) -> dict[str, Any]:
        result = {"codec": self.codec, **dict(self.metadata)}
        if digest is not None:
            result["artifact_sha256"] = digest
        else:
            result["value"] = self.value
        return result


class StateCodec(Protocol):
    name: str

    def can_encode(self, value: Any) -> bool: ...

    def encode(self, value: Any) -> EncodedState: ...

    def decode(self, payload: bytes | None, value: Any = None, **metadata: Any) -> Any: ...


class JsonStateCodec:
    name = "json"

    def can_encode(self, value: Any) -> bool:
        try:
            json.dumps(value, allow_nan=False)
        except (TypeError, ValueError):
            return False
        return True

    def encode(self, value: Any) -> EncodedState:
        if not self.can_encode(value):
            raise TypeError("state is not finite JSON")
        return EncodedState(self.name, value=value)

    def decode(self, payload: bytes | None, value: Any = None, **metadata: Any) -> Any:
        if payload is not None:
            return json.loads(payload.decode("utf-8"))
        return value


class BytesStateCodec:
    name = "bytes"

    def can_encode(self, value: Any) -> bool:
        return isinstance(value, bytes)

    def encode(self, value: Any) -> EncodedState:
        if not isinstance(value, bytes):
            raise TypeError("bytes codec requires bytes")
        return EncodedState(self.name, payload=value)

    def decode(self, payload: bytes | None, value: Any = None, **metadata: Any) -> bytes:
        if payload is None:
            raise ValueError("bytes state is missing its payload")
        return payload


def _array_metadata(value: Any) -> dict[str, Any]:
    dtype = getattr(value, "dtype", None)
    shape = getattr(value, "shape", ())
    return {
        "dtype": str(dtype) if dtype is not None else None,
        "shape": [int(item) for item in shape] if shape is not None else [],
    }


class NumpyStateCodec:
    name = "numpy.ndarray"

    def can_encode(self, value: Any) -> bool:
        return type(value).__module__.split(".", 1)[0] == "numpy" and hasattr(value, "tobytes")

    def encode(self, value: Any) -> EncodedState:
        if not self.can_encode(value):
            raise TypeError("numpy codec requires a numpy ndarray")
        flags = getattr(value, "flags", None)
        contiguous = value if getattr(flags, "c_contiguous", False) else value.copy()
        return EncodedState(
            self.name,
            payload=contiguous.tobytes(order="C"),
            metadata=_array_metadata(value),
        )

    def decode(self, payload: bytes | None, value: Any = None, **metadata: Any) -> Any:
        if payload is None:
            raise ValueError("numpy state is missing its payload")
        try:
            import numpy as np  # type: ignore
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("decoding numpy state requires numpy") from exc
        dtype = metadata.get("dtype")
        shape = tuple(int(item) for item in metadata.get("shape", ()))
        if not dtype:
            raise ValueError("numpy state is missing dtype metadata")
        array = np.frombuffer(payload, dtype=np.dtype(dtype)).copy()
        return array.reshape(shape)


def _torch_payload(torch: Any, tensor: Any) -> bytes:
    """Copy dense tensor bytes, including scalars and empty tensors."""
    contiguous = tensor.detach().cpu().contiguous()
    if int(contiguous.numel()) == 0:
        return b""
    flat = contiguous.reshape(-1)
    try:
        return flat.view(torch.uint8).numpy().tobytes(order="C")
    except Exception as exc:
        if hasattr(flat, "untyped_storage"):
            storage = flat.untyped_storage()
            nbytes = int(flat.numel()) * int(flat.element_size())
            return bytes(storage)[:nbytes]
        raise TypeError("torch tensor cannot be represented as raw bytes") from exc


class TorchStateCodec:
    name = "torch.tensor"

    def __init__(self, torch_module: Any | None = None) -> None:
        self._torch = torch_module

    @property
    def torch(self) -> Any:
        if self._torch is None:
            try:
                import torch  # type: ignore
            except ImportError as exc:  # pragma: no cover - optional dependency
                raise RuntimeError("decoding torch state requires PyTorch") from exc
            self._torch = torch
        return self._torch

    def can_encode(self, value: Any) -> bool:
        module = type(value).__module__.split(".", 1)[0]
        return module == "torch" and hasattr(value, "detach") and hasattr(value, "dtype")

    def encode(self, value: Any) -> EncodedState:
        if not self.can_encode(value):
            raise TypeError("torch codec requires a tensor")
        tensor = value.detach().cpu().contiguous()
        payload = _torch_payload(self.torch, value)
        metadata = _array_metadata(tensor)
        metadata["device"] = str(getattr(value, "device", "cpu"))
        metadata["requires_grad"] = bool(getattr(value, "requires_grad", False))
        return EncodedState(self.name, payload=payload, metadata=metadata)

    def decode(self, payload: bytes | None, value: Any = None, **metadata: Any) -> Any:
        if payload is None:
            raise ValueError("torch state is missing its payload")
        torch = self.torch
        dtype_name = str(metadata.get("dtype", "torch.float32"))
        dtype = getattr(torch, dtype_name.rsplit(".", 1)[-1], None)
        if dtype is None:
            raise ValueError(f"unsupported torch dtype metadata: {dtype_name}")
        shape = tuple(int(item) for item in metadata.get("shape", ()))
        if not payload:
            tensor = torch.empty(shape, dtype=dtype)
        else:
            try:
                import numpy as np  # type: ignore

                raw = np.frombuffer(payload, dtype=np.uint8).copy()
                tensor = torch.from_numpy(raw).view(dtype).reshape(shape)
            except ImportError as exc:  # pragma: no cover - torch normally depends on numpy
                raise RuntimeError("decoding torch state requires numpy") from exc
        device = metadata.get("device") or "cpu"
        tensor = tensor.to(device)
        if metadata.get("requires_grad") and getattr(tensor, "is_floating_point", lambda: False)():
            tensor.requires_grad_(True)
        return tensor


class StateCodecRegistry:
    """Ordered registry with safe defaults and explicit plugin registration."""

    def __init__(self, codecs: list[StateCodec] | None = None) -> None:
        self._codecs: list[StateCodec] = list(
            codecs or [BytesStateCodec(), NumpyStateCodec(), TorchStateCodec(), JsonStateCodec()]
        )

    def register(self, codec: StateCodec, *, first: bool = False) -> None:
        if not getattr(codec, "name", None):
            raise ValueError("state codec must define a non-empty name")
        self._codecs = [item for item in self._codecs if item.name != codec.name]
        if first:
            self._codecs.insert(0, codec)
        else:
            self._codecs.append(codec)

    def get(self, name: str) -> StateCodec:
        for codec in self._codecs:
            if codec.name == name:
                return codec
        raise KeyError(name)

    def encode(self, value: Any) -> EncodedState:
        for codec in self._codecs:
            try:
                if codec.can_encode(value):
                    return codec.encode(value)
            except (ImportError, RuntimeError):
                continue
        raise TypeError(
            f"no registered state codec can encode {type(value).__module__}."
            f"{type(value).__qualname__}"
        )

    def decode(
        self,
        codec: str,
        *,
        payload: bytes | None = None,
        value: Any = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> Any:
        selected = self.get(codec)
        return selected.decode(payload, value, **dict(metadata or {}))

    @property
    def codecs(self) -> tuple[StateCodec, ...]:
        return tuple(self._codecs)


default_state_codecs = StateCodecRegistry()

PayloadStore = Callable[[bytes, Mapping[str, Any]], str]
PayloadLoader = Callable[[str], bytes]


def _unsupported_state(value: Any) -> TypeError:
    return TypeError(
        f"state value {type(value).__module__}.{type(value).__qualname__} is not replayable; "
        "no registered codec can encode it"
    )


def _state_envelope(payload: Mapping[str, Any]) -> dict[str, Any]:
    return {STATE_TREE_KEY: STATE_TREE_VERSION, **dict(payload)}


def encode_state_tree(
    value: Any,
    store_payload: PayloadStore | None = None,
    *,
    registry: StateCodecRegistry | None = None,
    name: str = "state",
) -> Any:
    """Encode nested state while preserving key types, tuples, and tensors.

    Containers are wrapped in a versioned envelope so user dictionaries that
    happen to contain ``codec`` / ``kind`` keys cannot be mistaken for markers.
    Unsupported values raise ``TypeError`` instead of falling back to
    ``tolist()`` or ``repr()``.
    """
    selected = registry or default_state_codecs
    if isinstance(value, Mapping):
        items = []
        for key, item in value.items():
            items.append(
                {
                    "key": encode_state_tree(
                        key, store_payload, registry=selected, name=f"{name}-key"
                    ),
                    "value": encode_state_tree(
                        item, store_payload, registry=selected, name=f"{name}-{key}"
                    ),
                }
            )
        return _state_envelope({"codec": CONTAINER_CODEC, "kind": "mapping", "items": items})
    if isinstance(value, tuple):
        return _state_envelope(
            {
                "codec": CONTAINER_CODEC,
                "kind": "tuple",
                "items": [
                    encode_state_tree(
                        item, store_payload, registry=selected, name=f"{name}-{index}"
                    )
                    for index, item in enumerate(value)
                ],
            }
        )
    if isinstance(value, list):
        return _state_envelope(
            {
                "codec": CONTAINER_CODEC,
                "kind": "list",
                "items": [
                    encode_state_tree(
                        item, store_payload, registry=selected, name=f"{name}-{index}"
                    )
                    for index, item in enumerate(value)
                ],
            }
        )
    if isinstance(value, _JSON_ATOMS):
        if isinstance(value, float) and not math.isfinite(value):
            raise TypeError(f"replay value {name!r} contains a non-finite float")
        if isinstance(value, bool) or not isinstance(value, int) or abs(value) < 2**53:
            return value
        return _state_envelope({"codec": CONTAINER_CODEC, "kind": "int", "value": str(value)})
    try:
        encoded = selected.encode(value)
    except TypeError as exc:
        raise _unsupported_state(value) from exc
    if encoded.payload is None:
        return encoded.value
    if store_payload is None:
        raise TypeError(f"binary state {name!r} requires an artifact store")
    digest = store_payload(encoded.payload, {"codec": encoded.codec, **dict(encoded.metadata)})
    return _state_envelope(encoded.marker(digest))


def _decode_container(
    value: Mapping[str, Any],
    load_payload: PayloadLoader | None,
    registry: StateCodecRegistry,
    *,
    legacy: bool,
) -> Any:
    codec = value.get("codec")
    if codec == CONTAINER_CODEC:
        kind = value.get("kind")
        items = value.get("items", ())
        if kind == "mapping":
            result: dict[Any, Any] = {}
            if not isinstance(items, Sequence) or isinstance(items, (str, bytes)):
                raise ValueError("typed mapping items must be a sequence")
            for item in items:
                if not isinstance(item, Mapping):
                    raise ValueError("typed mapping item must be an object")
                key = decode_state_tree(item.get("key"), load_payload, registry=registry)
                if key in result:
                    raise ValueError(f"duplicate mapping key {key!r}")
                result[key] = decode_state_tree(item.get("value"), load_payload, registry=registry)
            return result
        if kind == "tuple":
            if not isinstance(items, Sequence) or isinstance(items, (str, bytes)):
                raise ValueError("tuple items must be a sequence")
            return tuple(decode_state_tree(item, load_payload, registry=registry) for item in items)
        if kind == "list":
            if not isinstance(items, Sequence) or isinstance(items, (str, bytes)):
                raise ValueError("list items must be a sequence")
            return [decode_state_tree(item, load_payload, registry=registry) for item in items]
        if kind == "int":
            return int(value.get("value", "0"))
        raise ValueError(f"unsupported container kind {kind!r}")
    if "artifact_sha256" in value:
        if load_payload is None:
            raise ValueError("binary state is missing its payload loader")
        payload = load_payload(str(value["artifact_sha256"]))
        codec_name = str(value.get("codec", "bytes"))
        metadata = {
            str(key): item
            for key, item in value.items()
            if key not in {"artifact_sha256", "codec", STATE_TREE_KEY}
        }
        if codec_name == "bytes":
            return payload
        if codec_name == "json":
            return json.loads(payload)
        try:
            return registry.decode(codec_name, payload=payload, metadata=metadata)
        except KeyError as exc:
            raise ValueError(f"unsupported replay artifact codec {codec_name!r}") from exc
    if legacy:
        return {
            str(key): decode_state_tree(item, load_payload, registry=registry)
            for key, item in value.items()
        }
    raise ValueError("state envelope is missing a codec marker")


def decode_state_tree(
    value: Any,
    load_payload: PayloadLoader | None = None,
    *,
    registry: StateCodecRegistry | None = None,
) -> Any:
    """Invert :func:`encode_state_tree`, including older artifact markers."""
    selected = registry or default_state_codecs
    if isinstance(value, Mapping):
        if STATE_TREE_KEY in value:
            version = value.get(STATE_TREE_KEY)
            if version != STATE_TREE_VERSION:
                raise ValueError(f"unsupported state-tree version {version!r}")
            return _decode_container(value, load_payload, selected, legacy=False)
        return _decode_container(value, load_payload, selected, legacy=True)
    if isinstance(value, list):
        return [decode_state_tree(item, load_payload, registry=selected) for item in value]
    return value


def collect_artifact_digests(value: Any) -> set[str]:
    """Return every ``artifact_sha256`` referenced by encoded state."""
    found: set[str] = set()

    def visit(item: Any) -> None:
        if isinstance(item, Mapping):
            digest = item.get("artifact_sha256")
            if isinstance(digest, str) and digest:
                found.add(digest)
            for nested in item.values():
                visit(nested)
        elif isinstance(item, (list, tuple)):
            for nested in item:
                visit(nested)

    visit(value)
    return found


def digest_state_tree(value: Any, *, registry: StateCodecRegistry | None = None) -> str | None:
    """Return a content digest of a replay value, or ``None`` if it cannot be encoded.

    Tensor and array payloads are hashed in full. Print formatting is never used
    as a substitute for content identity.
    """

    payloads: dict[str, bytes] = {}

    def store(payload: bytes, metadata: Mapping[str, Any]) -> str:
        digest = hashlib.sha256(payload).hexdigest()
        payloads[digest] = payload
        return digest

    try:
        encoded = encode_state_tree(value, store, registry=registry)
        canonical = json.dumps(encoded, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError):
        return None
    hasher = hashlib.sha256(canonical.encode("utf-8"))
    for digest in sorted(payloads):
        hasher.update(b"\0")
        hasher.update(digest.encode("ascii"))
        hasher.update(payloads[digest])
    return hasher.hexdigest()


__all__ = [
    "BytesStateCodec",
    "CONTAINER_CODEC",
    "STATE_TREE_KEY",
    "STATE_TREE_VERSION",
    "EncodedState",
    "JsonStateCodec",
    "NumpyStateCodec",
    "StateCodec",
    "StateCodecRegistry",
    "TorchStateCodec",
    "collect_artifact_digests",
    "decode_state_tree",
    "default_state_codecs",
    "digest_state_tree",
    "encode_state_tree",
]
