# Profiled Optimization Backlog

Optimization starts after correctness, evidence capture, and owned decode are green. Do not implement custom `mx.fast.metal_kernel` work before benchmark evidence points to a bottleneck.

1. Tighten `mx.eval` boundaries in decode timing.
2. Decouple model substrate from MLX-LM where loader evidence shows value.
3. Add prefill chunking for long prompts.
4. Improve real scheduler batching after mixed-request traces are stable.
5. Tune cache policy limits and prompt-cache reuse.
6. Measure quantized weights and possible KV-cache storage choices.
7. Design future block/paged KV cache with token-index to physical-block mapping.
8. Consider custom Metal kernels for RMSNorm, RoPE, attention, or MLP only after profile data shows MLX primitives are the limiting cost.

Every kept optimization needs a same-machine benchmark before and after the change plus unchanged behavior evidence.
