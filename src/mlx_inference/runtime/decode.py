from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from time import perf_counter
from typing import Any, Callable, Sequence

import mlx.core as mx

LogitsProcessor = Callable[[mx.array], mx.array]

_model_call_cache: dict[int, str] = {}


@dataclass(slots=True)
class DecodeStepResult:
    token_id: int
    logits: mx.array
    cache: Any
    elapsed_ms: float
    metadata: dict[str, Any] = field(default_factory=dict)


def decode_step(
    *,
    model: Any,
    input_ids: mx.array,
    cache: Any = None,
    temperature: float = 0.0,
    seed: int | None = None,
    logits_processors: Sequence[LogitsProcessor] | None = None,
) -> DecodeStepResult:
    started = perf_counter()
    output = _call_model(model, input_ids, cache)
    logits, next_cache = _extract_logits_and_cache(output, cache)

    for processor in logits_processors or ():
        logits = processor(logits)

    token_id = _select_token(logits, temperature=temperature, seed=seed)
    elapsed_ms = (perf_counter() - started) * 1000
    return DecodeStepResult(
        token_id=token_id,
        logits=logits,
        cache=next_cache,
        elapsed_ms=elapsed_ms,
        metadata={"temperature": temperature, "evaluated": True},
    )


def select_tokens_from_logits(
    logits: mx.array,
    *,
    temperature: float = 0.0,
    seed: int | None = None,
) -> list[int]:
    next_token_logits = logits[:, -1, :]
    if temperature <= 0:
        selected = mx.argmax(next_token_logits, axis=-1)
    elif seed is None:
        selected = mx.random.categorical(next_token_logits / temperature, axis=-1)
    else:
        selected = mx.random.categorical(next_token_logits / temperature, axis=-1, key=mx.random.key(seed))

    mx.eval(selected)
    return [int(token_id) for token_id in selected.tolist()]


def _call_model(model: Any, input_ids: mx.array, cache: Any) -> Any:
    model_id = id(model)

    if model_id not in _model_call_cache:
        try:
            signature = inspect.signature(model)
            parameters = list(signature.parameters.values())
        except (ValueError, TypeError):
            _model_call_cache[model_id] = "fallback"
        else:
            _model_call_cache[model_id] = _call_style_from_parameters(parameters)

    call_type = _model_call_cache[model_id]

    if call_type == "kwarg":
        return model(input_ids, cache=cache)
    if call_type == "positional":
        return model(input_ids, cache)
    if call_type == "single":
        return model(input_ids)

    if call_type == "fallback":
        try:
            return model(input_ids, cache=cache)
        except TypeError as first_error:
            try:
                return model(input_ids, cache)
            except TypeError:
                try:
                    return model(input_ids)
                except TypeError:
                    raise first_error

    raise ValueError(f"Unsupported model call style: {call_type}")


def _call_style_from_parameters(parameters: list[inspect.Parameter]) -> str:
    if any(parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in parameters):
        return "kwarg"

    cache_parameter = next((parameter for parameter in parameters if parameter.name == "cache"), None)
    if cache_parameter is not None:
        if cache_parameter.kind in (inspect.Parameter.KEYWORD_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD):
            return "kwarg"
        if cache_parameter.kind == inspect.Parameter.POSITIONAL_ONLY:
            return "positional"

    positional_parameters = [
        parameter
        for parameter in parameters
        if parameter.kind
        in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        )
    ]
    if len(positional_parameters) >= 2:
        return "positional"
    return "single"


def _extract_logits_and_cache(output: Any, cache: Any) -> tuple[mx.array, Any]:
    if isinstance(output, tuple):
        if len(output) >= 2:
            return output[0], output[1]
        return output[0], cache
    if isinstance(output, dict):
        logits = output.get("logits")
        if logits is None:
            raise ValueError("Model output dictionary did not contain logits")
        return logits, output.get("cache", cache)
    return output, cache


def _select_token(logits: mx.array, *, temperature: float, seed: int | None) -> int:
    return select_tokens_from_logits(logits, temperature=temperature, seed=seed)[0]
