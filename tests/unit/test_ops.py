from __future__ import annotations

import numpy as np
import mlx.core as mx

from mlx_inference.runtime.ops import causal_attention, gated_mlp, rms_norm, rope


def as_np(value: mx.array) -> np.ndarray:
    return np.array(value)


def test_rms_norm_matches_formula() -> None:
    x = mx.array([[1.0, 2.0, 3.0]], dtype=mx.float32)
    weight = mx.array([0.5, 1.0, 1.5], dtype=mx.float32)

    actual = as_np(rms_norm(x, weight, eps=1e-6))
    expected = np.array([[1.0, 2.0, 3.0]]) * np.array([0.5, 1.0, 1.5])
    expected = expected / np.sqrt(np.mean(np.square([[1.0, 2.0, 3.0]]), axis=-1, keepdims=True) + 1e-6)

    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)


def test_rope_preserves_shape_and_rotates_pairs() -> None:
    x = mx.array([[[[1.0, 0.0, 0.0, 1.0]]]], dtype=mx.float32)
    rotated = rope(x, offset=1, base=10_000.0)

    assert rotated.shape == x.shape
    assert not np.allclose(as_np(rotated), as_np(x))


def test_causal_attention_masks_future_tokens() -> None:
    q = mx.array([[[[1.0, 0.0], [0.0, 1.0]]]], dtype=mx.float32)
    k = q
    v = mx.array([[[[1.0, 10.0], [100.0, 1000.0]]]], dtype=mx.float32)

    out = as_np(causal_attention(q, k, v))

    np.testing.assert_allclose(out[0, 0, 0], np.array([1.0, 10.0]), rtol=1e-5, atol=1e-5)
    assert out[0, 0, 1, 0] > 1.0


def test_gated_mlp_shape_and_values() -> None:
    x = mx.array([[1.0, -1.0]], dtype=mx.float32)
    gate = mx.array([[1.0, 0.5], [0.25, 1.0]], dtype=mx.float32)
    up = mx.array([[0.5, 1.0], [1.0, 0.5]], dtype=mx.float32)
    down = mx.array([[1.0, 0.0], [0.0, 1.0]], dtype=mx.float32)

    out = gated_mlp(x, gate, up, down)

    assert out.shape == (1, 2)
    assert np.isfinite(as_np(out)).all()
