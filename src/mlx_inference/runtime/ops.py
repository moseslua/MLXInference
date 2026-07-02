from __future__ import annotations

import math

import mlx.core as mx

# Module-level caches for static computations
_rope_cache: dict[tuple[int, float], mx.array] = {}
_causal_mask_cache: dict[int, mx.array] = {}


def rms_norm(x: mx.array, weight: mx.array, *, eps: float = 1e-6) -> mx.array:
    variance = mx.mean(mx.square(x), axis=-1, keepdims=True)
    return x * mx.rsqrt(variance + eps) * weight


def rope(x: mx.array, *, offset: int = 0, base: float = 10_000.0) -> mx.array:
    dim = x.shape[-1]
    if dim % 2 != 0:
        raise ValueError("RoPE requires an even head dimension")

    # Cache inv_freq computation per (dim, base)
    cache_key = (dim, base)
    if cache_key not in _rope_cache:
        dims = mx.arange(0, dim, 2, dtype=mx.float32)
        _rope_cache[cache_key] = mx.power(base, -dims / dim)
    inv_freq = _rope_cache[cache_key]

    seq_len = x.shape[-2]
    positions = mx.arange(offset, offset + seq_len, dtype=mx.float32)
    angles = positions[:, None] * inv_freq[None, :]
    cos = mx.cos(angles)
    sin = mx.sin(angles)

    # Direct broadcast instead of iterative expansion
    target_dims = len(x.shape) - 2
    if target_dims > 0:
        new_shape = (1,) * target_dims + cos.shape
        cos = mx.reshape(cos, new_shape)
        sin = mx.reshape(sin, new_shape)

    even = x[..., 0::2]
    odd = x[..., 1::2]
    rotated_even = even * cos - odd * sin
    rotated_odd = even * sin + odd * cos
    return mx.stack([rotated_even, rotated_odd], axis=-1).reshape(x.shape)


def causal_attention(q: mx.array, k: mx.array, v: mx.array) -> mx.array:
    scale = 1.0 / math.sqrt(q.shape[-1])
    scores = mx.matmul(q, mx.swapaxes(k, -1, -2)) * scale
    seq_len = q.shape[-2]

    # Cache causal mask per sequence length
    if seq_len not in _causal_mask_cache:
        _causal_mask_cache[seq_len] = mx.triu(mx.ones((seq_len, seq_len), dtype=mx.bool_), k=1)
    mask = _causal_mask_cache[seq_len]

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
