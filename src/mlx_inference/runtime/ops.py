from __future__ import annotations

import math

import mlx.core as mx


def rms_norm(x: mx.array, weight: mx.array, *, eps: float = 1e-6) -> mx.array:
    variance = mx.mean(mx.square(x), axis=-1, keepdims=True)
    return x * mx.rsqrt(variance + eps) * weight


def rope(x: mx.array, *, offset: int = 0, base: float = 10_000.0) -> mx.array:
    dim = x.shape[-1]
    if dim % 2 != 0:
        raise ValueError("RoPE requires an even head dimension")

    seq_len = x.shape[-2]
    positions = mx.arange(offset, offset + seq_len, dtype=mx.float32)
    dims = mx.arange(0, dim, 2, dtype=mx.float32)
    inv_freq = mx.power(base, -dims / dim)
    angles = positions[:, None] * inv_freq[None, :]
    cos = mx.cos(angles)
    sin = mx.sin(angles)

    while len(cos.shape) < len(x.shape) - 1:
        cos = mx.expand_dims(cos, axis=0)
        sin = mx.expand_dims(sin, axis=0)

    even = x[..., 0::2]
    odd = x[..., 1::2]
    rotated_even = even * cos - odd * sin
    rotated_odd = even * sin + odd * cos
    return mx.stack([rotated_even, rotated_odd], axis=-1).reshape(x.shape)


def causal_attention(q: mx.array, k: mx.array, v: mx.array) -> mx.array:
    scale = 1.0 / math.sqrt(q.shape[-1])
    scores = mx.matmul(q, mx.swapaxes(k, -1, -2)) * scale
    seq_len = q.shape[-2]
    mask = mx.triu(mx.ones((seq_len, seq_len), dtype=mx.bool_), k=1)
    scores = mx.where(mask, mx.array(-1e9, dtype=scores.dtype), scores)
    weights = mx.softmax(scores, axis=-1)
    return mx.matmul(weights, v)


def gated_mlp(x: mx.array, gate_weight: mx.array, up_weight: mx.array, down_weight: mx.array) -> mx.array:
    gate = mx.matmul(x, gate_weight)
    up = mx.matmul(x, up_weight)
    hidden = _silu(gate) * up
    return mx.matmul(hidden, down_weight)


def _silu(x: mx.array) -> mx.array:
    return x * mx.sigmoid(x)
