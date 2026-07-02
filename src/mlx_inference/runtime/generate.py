from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from time import perf_counter
from typing import Any, Callable, Iterable, Sequence

import mlx.core as mx

from mlx_inference.runtime.decode import decode_step


@dataclass(slots=True)
class GenerationTiming:
    prefill_ms: float = 0.0
    decode_ms: float = 0.0
    decode_steps: int = 0
    step_ms: list[float] = field(default_factory=list)


@dataclass(slots=True)
class GenerationResult:
    text: str
    token_ids: list[int]
    prompt_token_ids: list[int]
    stop_reason: str
    timing: GenerationTiming
    metadata: dict[str, Any] = field(default_factory=dict)


def generate_greedy(
    *,
    model: Any,
    tokenizer: Any,
    prompt: str | None = None,
    input_ids: Sequence[int] | None = None,
    max_new_tokens: int = 128,
    temperature: float = 0.0,
    seed: int | None = None,
    eos_token_ids: Iterable[int] | None = None,
    cache: Any = None,
    collect_step_timings: bool = True,
) -> GenerationResult:
    if max_new_tokens < 0:
        raise ValueError("max_new_tokens must be non-negative")
    prompt_token_ids = _resolve_input_ids(tokenizer, prompt, input_ids)
    if not prompt_token_ids:
        raise ValueError("input_ids must not be empty")

    eos = set(eos_token_ids or _tokenizer_eos(tokenizer))
    decode_tokens = _token_decoder(tokenizer)
    generated: list[int] = []
    timing = GenerationTiming()
    stop_reason = "max_tokens"
    next_input = list(prompt_token_ids)

    for step in range(max_new_tokens):
        step_input = mx.array([next_input], dtype=mx.int32)
        started = perf_counter()
        step_result = decode_step(
            model=model,
            input_ids=step_input,
            cache=cache,
            temperature=temperature,
            seed=None if seed is None else seed + step,
        )
        elapsed = (perf_counter() - started) * 1000
        cache = step_result.cache
        token_id = step_result.token_id
        generated.append(token_id)
        next_input[0:] = [token_id]
        timing.decode_steps += 1
        if collect_step_timings:
            timing.step_ms.append(step_result.elapsed_ms)
        if step == 0:
            timing.prefill_ms += elapsed
        else:
            timing.decode_ms += elapsed

        if token_id in eos:
            stop_reason = "eos"
            break

    return GenerationResult(
        text=decode_tokens(generated),
        token_ids=generated,
        prompt_token_ids=prompt_token_ids,
        stop_reason=stop_reason,
        timing=timing,
        metadata={"owned_decode": True, "mx_eval_boundary": "decode_step"},
    )


def _resolve_input_ids(tokenizer: Any, prompt: str | None, input_ids: Sequence[int] | None) -> list[int]:
    if input_ids is not None:
        return [int(token) for token in input_ids]
    if prompt is None:
        raise ValueError("Either prompt or input_ids is required")
    encoded = _encode_prompt(tokenizer.encode, prompt)
    return [int(token) for token in encoded]


def _encode_prompt(encode: Callable[[str], Sequence[int]], prompt: str) -> tuple[int, ...]:
    try:
        return _encode_prompt_cached(encode, prompt)
    except TypeError:
        return tuple(int(token) for token in encode(prompt))


@lru_cache(maxsize=256)
def _encode_prompt_cached(encode: Callable[[str], Sequence[int]], prompt: str) -> tuple[int, ...]:
    return tuple(int(token) for token in encode(prompt))


def _token_decoder(tokenizer: Any):
    decode = getattr(tokenizer, "decode", None)
    if decode is None:
        return lambda token_ids: " ".join(str(token_id) for token_id in token_ids)
    return lambda token_ids: str(decode(list(token_ids)))


def _tokenizer_eos(tokenizer: Any) -> set[int]:
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    if eos_token_id is None:
        return set()
    if isinstance(eos_token_id, (list, tuple, set)):
        return {int(token) for token in eos_token_id}
    return {int(eos_token_id)}
