from __future__ import annotations

import mlx.core as mx

from mlx_inference.runtime.decode import decode_step, select_tokens_from_logits
from mlx_inference.testing.fake import FakeCausalLM


def test_decode_step_passes_cache_and_evaluates_logits() -> None:
    model = FakeCausalLM(vocab_size=16)
    cache = {"request_id": "r1", "tokens": []}

    result = decode_step(
        model=model,
        input_ids=mx.array([[2, 3, 4]]),
        cache=cache,
        temperature=0.0,
        seed=1,
    )

    assert result.token_id == 5
    assert result.cache is cache
    assert cache["tokens"] == [2, 3, 4]
    assert model.call_count == 1
    assert result.logits is not None


def test_select_tokens_from_logits_uses_mlx_arrays() -> None:
    logits = mx.array(
        [
            [[0.0, 3.0, 1.0]],
            [[4.0, 2.0, 0.0]],
        ],
        dtype=mx.float32,
    )

    assert select_tokens_from_logits(logits, temperature=0.0) == [1, 0]
