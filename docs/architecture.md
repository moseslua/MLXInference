# MLX Inference Runtime Architecture

This project treats MLX-LM as a model substrate and reference oracle. The runtime owns decode, cache policy, scheduling, timing boundaries, benchmark capture, CLI behavior, and evidence.

| CUDA concept | MLX reality | phase-1 treatment | deferred/dropped work |
| --- | --- | --- | --- |
| CUDA Graphs | MLX records lazy compute graphs and materializes with `mx.eval` | Record explicit `mx.eval` boundaries around decode and benchmark timing | Do not implement CUDA Graphs |
| VRAM/H2D memory pools | MLX uses unified memory on Apple Silicon | Avoid explicit host-to-device allocator design | Drop CUDA memory-pool porting |
| Paged KV cache | Prompt caches and model-native caches exist, but paged KV needs token-index to block mapping | Implement explicit cache taxonomy and bounded per-request policy | Future block/paged cache design only |
| Continuous batching | A small scheduler can batch active request token slices | Drive `active_requests -> token slice -> model forward -> per-request next tokens -> cache update` | No vLLM serving claim |
| Fused CUDA kernels | MLX primitives already dispatch optimized Metal operations | Test RMSNorm, RoPE, attention, and MLP formulas | Custom Metal kernels only after profiling |
| NCCL/tensor parallelism | Single-machine Apple Silicon runtime first | Keep out of scope | No NCCL or tensor-parallel implementation |
| Quantization | MLX-LM provides quantized model loading | Record quantization metadata in benchmarks | No custom quantized kernels |
| Speculative decoding | Requires draft model policy and acceptance tests | Out of scope | Defer until baseline decode is measured |

## Runtime Ownership

The owned loop is:

```text
x_t -> model(x_t, cache) -> logits -> logits processor -> argmax/sample -> x_{t+1}
```

Production runtime and CLI code must not call `mlx_lm.generate` or `mlx_lm.stream_generate`. Those helpers are reserved for reference tests and oracle scripts.

## Cache Taxonomy

1. MLX-LM native cache wrapper.
2. Local `mlx_inference` cache policy abstraction.
3. Prompt-cache persistence and reuse.
4. Future block/paged cache design.

Only the first three are phase-1 runtime claims.
