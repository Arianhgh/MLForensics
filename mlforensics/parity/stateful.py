"""Small state-carrying adapters for parity execution.

State is explicit at the adapter boundary: a model receives the current state
alongside its normal inputs and returns the next state in a mapping or tuple.
This keeps sequential inference deterministic and makes state reset between
parity cases straightforward.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

from .backends import BackendError, _invoke_model

_MISSING = object()


class StatefulModelAdapter:
    """Adapt a callable model whose output contains its next recurrent state.

    The adapter accepts scalar, positional, or named inputs.  For named inputs
    the current state is inserted under ``state_input_name``.  For positional
    inputs it is appended by default (or inserted at ``state_input_index``).
    The state is captured from ``state_output_key`` or ``state_output_index``;
    tuple/list outputs default to index 1.
    """

    def __init__(
        self,
        model: Callable[..., Any],
        *,
        name: str = "stateful",
        initial_state: Any = None,
        state_input_name: str = "state",
        state_input_index: int | None = None,
        state_output_key: str | None = None,
        state_output_index: int | None = None,
        input_names: Sequence[str] | None = None,
        call_mode: str | None = None,
        return_state: bool = False,
        prepare_input: Callable[[Any], Any] | None = None,
        postprocess: Callable[[Any], Any] | None = None,
    ) -> None:
        if not callable(model):
            raise TypeError("model must be callable")
        self.model = model
        self.name = name
        self.initial_state = initial_state
        self.state_input_name = str(state_input_name) if state_input_name else None
        self.state_input_index = state_input_index
        self.state_output_key = state_output_key
        self.state_output_index = state_output_index
        self.input_names = tuple(str(item) for item in input_names) if input_names else None
        self.call_mode = call_mode
        self.return_state = bool(return_state)
        self.prepare_input = prepare_input
        self.postprocess = postprocess
        self._state = initial_state

    @property
    def state(self) -> Any:
        """Return the current state, which may legitimately be ``None``."""

        return self._state

    def reset_state(self, state: Any = _MISSING) -> None:
        """Reset adapter state and invoke an optional model reset hook."""

        self._state = self.initial_state if state is _MISSING else state
        reset = getattr(self.model, "reset_state", None)
        if callable(reset):
            reset()

    def predict(self, inputs: Any) -> Any:
        value = self.prepare_input(inputs) if self.prepare_input else inputs
        prepared = self._inject_state(value)
        call_mode = self.call_mode
        if call_mode is None:
            # A state injected into a scalar/sequence creates positional
            # arguments.  A mapping remains one structured argument unless
            # input_names explicitly opts into named keyword invocation.
            call_mode = (
                "kwargs"
                if self.input_names
                else ("single" if isinstance(prepared, Mapping) else "args")
            )
        input_names = self._effective_input_names(prepared)
        output = _invoke_model(
            self.model,
            prepared,
            input_names=input_names,
            call_mode=call_mode,
        )
        output = self._capture_state(output)
        return self.postprocess(output) if self.postprocess else output

    __call__ = predict

    def predict_batch(self, inputs: Sequence[Any]) -> list[Any]:
        # Stateful models are intentionally evaluated in order; batching them
        # would change the state transition semantics.
        return [self.predict(value) for value in inputs]

    def _inject_state(self, value: Any) -> Any:
        if isinstance(value, Mapping):
            result = dict(value)
            if self.state_input_name is not None:
                result.setdefault(self.state_input_name, self._state)
            return result
        if isinstance(value, (tuple, list)):
            values = list(value)
        else:
            values = [value]
        index = self.state_input_index
        if index is None:
            values.append(self._state)
        else:
            if index < 0:
                index += len(values) + 1
            if index < 0 or index > len(values):
                raise ValueError("state_input_index is outside the model inputs")
            values.insert(index, self._state)
        return tuple(values) if isinstance(value, tuple) else values

    def _effective_input_names(self, value: Any) -> Sequence[str] | None:
        if self.input_names is None or isinstance(value, Mapping) or self.state_input_name is None:
            return self.input_names
        if self.state_input_name in self.input_names:
            return self.input_names
        names = list(self.input_names)
        index = self.state_input_index
        if index is None:
            names.append(self.state_input_name)
        else:
            if index < 0:
                index += len(names) + 1
            if index < 0 or index > len(names):
                raise ValueError("state_input_index is outside the model inputs")
            names.insert(index, self.state_input_name)
        return tuple(names)

    def _capture_state(self, output: Any) -> Any:
        key = self.state_output_key
        index = self.state_output_index
        if key is not None:
            if not isinstance(output, Mapping) or key not in output:
                raise BackendError(f"state output key is missing: {key}")
            self._state = output[key]
            if not self.return_state:
                return {item_key: item for item_key, item in output.items() if item_key != key}
            return output

        if index is None and isinstance(output, (tuple, list)):
            index = 1
        if index is None:
            raise BackendError(
                "stateful output must specify state_output_key or return a tuple/list "
                "with state at index 1"
            )
        if not isinstance(output, (tuple, list)):
            raise BackendError("state_output_index requires a tuple or list model output")
        try:
            self._state = output[index]
        except IndexError as exc:
            raise BackendError("state output index is outside the model output") from exc
        if self.return_state:
            return output
        remaining = [value for item_index, value in enumerate(output) if item_index != index]
        if len(remaining) == 1:
            return remaining[0]
        return type(output)(remaining)


StatefulCallableAdapter = StatefulModelAdapter
StatefulAdapter = StatefulModelAdapter


__all__ = ["StatefulAdapter", "StatefulCallableAdapter", "StatefulModelAdapter"]
