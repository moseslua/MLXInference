from __future__ import annotations

from mlx_inference.runtime.generate import generate_greedy
from mlx_inference.testing.fake import FakeCausalLM, FakeTokenizer


def test_generate_greedy_owns_stepwise_model_calls() -> None:
    tokenizer = FakeTokenizer()
    model = FakeCausalLM(vocab_size=16)

    result = generate_greedy(
        model=model,
        tokenizer=tokenizer,
        prompt="abc",
        max_new_tokens=4,
        temperature=0.0,
        seed=7,
    )

    assert result.token_ids == [5, 6, 7, 8]
    assert result.text == "<5> <6> <7> <8>"
    assert model.call_count == 4
    assert result.timing.decode_steps == 4
    assert result.stop_reason == "max_tokens"


def test_generate_greedy_stops_on_eos() -> None:
    tokenizer = FakeTokenizer()
    model = FakeCausalLM(vocab_size=8)

    result = generate_greedy(
        model=model,
        tokenizer=tokenizer,
        input_ids=[6],
        max_new_tokens=5,
        eos_token_ids={7},
    )

    assert result.token_ids == [7]
    assert result.stop_reason == "eos"
