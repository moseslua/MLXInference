from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from mlx_inference.testing.fake import FakeCausalLM, FakeTokenizer


@dataclass(slots=True)
class LoadedModel:
    model: Any
    tokenizer: Any
    model_id: str
    config: dict[str, Any] = field(default_factory=dict)
    is_fake: bool = False


def load_model(model_id: str) -> LoadedModel:
    if model_id == "fake://deterministic":
        return LoadedModel(
            model=FakeCausalLM(),
            tokenizer=FakeTokenizer(),
            model_id=model_id,
            config={"model_type": "fake", "quantization": None},
            is_fake=True,
        )
    try:
        from mlx_lm import load
    except Exception as exc:
        raise RuntimeError("Unable to import mlx_lm. Install project dependencies with uv sync.") from exc

    try:
        loaded = load(model_id, return_config=True)
    except TypeError:
        model, tokenizer = load(model_id)
        return LoadedModel(model=model, tokenizer=tokenizer, model_id=model_id, config={}, is_fake=False)
    except Exception as exc:
        raise RuntimeError(f"Unable to load model '{model_id}': {exc}") from exc

    if len(loaded) == 3:
        model, tokenizer, config = loaded
    else:
        model, tokenizer = loaded[:2]
        config = {}
    return LoadedModel(model=model, tokenizer=tokenizer, model_id=model_id, config=dict(config), is_fake=False)


def validate_manifest(config: dict[str, Any]) -> None:
    if not isinstance(config, dict):
        raise ValueError("Model config must be a dictionary")
    if config and "model_type" not in config:
        raise ValueError("Model config is missing model_type")
