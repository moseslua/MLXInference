from __future__ import annotations

import mlx.core as mx

from mlx_inference.runtime.decode import decode_step
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
