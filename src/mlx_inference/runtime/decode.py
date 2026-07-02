from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter
from typing import Any, Callable, Sequence

import mlx.core as mx
import numpy as np

LogitsProcessor = Callable[[mx.array], mx.array]


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

    mx.eval(logits)
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
    mx.eval(logits)
    values = np.array(logits[:, -1, :])
    if temperature <= 0:
        return [int(row.argmax()) for row in values]

    rng = np.random.default_rng(seed)
    selected: list[int] = []
    for row in values:
        scaled = row / temperature
        scaled = scaled - scaled.max()
        probabilities = np.exp(scaled)
        probabilities = probabilities / probabilities.sum()
        selected.append(int(rng.choice(len(row), p=probabilities)))
    return selected


def _call_model(model: Any, input_ids: mx.array, cache: Any) -> Any:
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
