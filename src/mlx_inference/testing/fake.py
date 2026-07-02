from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import mlx.core as mx
import numpy as np


@dataclass(slots=True)
class FakeTokenizer:
    eos_token_id: int | None = None

    def encode(self, text: str) -> list[int]:
        if not text:
            return [1]
        return [max(1, (ord(character.lower()) - 95) % 100) for character in text]

    def decode(self, token_ids: list[int]) -> str:
        return " ".join(f"<{int(token_id)}>" for token_id in token_ids)


class FakeCausalLM:
    def __init__(self, *, vocab_size: int = 128) -> None:
        self.vocab_size = vocab_size
        self.call_count = 0
        self.batch_call_count = 0

    def __call__(self, input_ids: mx.array, cache: Any = None) -> mx.array:
        self.call_count += 1
        values = np.array(input_ids)
        if values.ndim != 2:
            raise ValueError("FakeCausalLM expects [batch, sequence] input ids")

        batch, seq_len = values.shape
        if batch > 1:
            self.batch_call_count += 1

        if isinstance(cache, dict) and isinstance(cache.get("tokens"), list):
            cache["tokens"].extend(int(token) for token in values.reshape(-1).tolist())

        logits = np.full((batch, seq_len, self.vocab_size), -1000.0, dtype=np.float32)
        for row_index, row in enumerate(values):
            next_token = int(row[-1] + 1) % self.vocab_size
            logits[row_index, -1, next_token] = 1000.0
        return mx.array(logits)
