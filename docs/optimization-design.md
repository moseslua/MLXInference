# Runtime Optimization Notes

Date: 2026-07-02

This note records the optimization changes made after the runtime evidence baseline. The changes keep MLX-LM as the model loader/reference substrate and keep step-wise decode owned by `mlx_inference`.

## Scope

The pass focused on small, behavior-preserving runtime improvements:

- keep token selection in MLX until the final Python token id conversion
- avoid redundant synchronization in decode token selection
- cache model call-style inspection instead of probing with exceptions on every decode step
- split first-step prefill timing from later decode-step timing
- make per-step timing and scheduler trace collection optional
- bound scheduler traces and add O(1) pending-request lookup for cancellation
- include an attention-mask field in scheduler forward trace events
- cache RoPE inverse frequencies and causal masks by static shape parameters
- apply cache rotation consistently when loading prompt-cache files
- write prompt-cache JSON without pretty-print overhead

The pass intentionally did not add new dependencies or claim paged KV cache, custom Metal kernels, speculative decoding, or production serving behavior.

## Verification

The following checks cover the optimized paths:

```sh
uv run python -m pytest tests/unit tests/component -q
uv run python scripts/verify/assert_owned_decode.py --package src/mlx_inference --forbid mlx_lm.generate --forbid mlx_lm.stream_generate
uv run python scripts/verify/capture_scheduler_trace.py --model-id fake://deterministic --out /tmp/mlx-optimized-scheduler-trace.jsonl
uv run python scripts/verify/replay_scheduler_trace.py --trace /tmp/mlx-optimized-scheduler-trace.jsonl
uv run python scripts/verify/benchmark_runtime.py --model-id fake://deterministic --prompt-lens 4 --output-lens 3 --batch-sizes 1 --runs 2 --warmup-runs 1 --out /tmp/mlx-optimized-benchmark.json
```

Observed results in the implementation run:

- unit/component tests: 17 passed
- owned-decode static guard: passed
- scheduler trace replay: passed
- fake-model benchmark JSON: generated successfully

## Remaining Limits

Benchmark output from `fake://deterministic` is useful as a schema and plumbing check, not as a real model throughput claim. Real performance claims still need before/after measurements on the same machine with a real MLX-LM model.

Scheduler attention masks are recorded in trace events for observability. They are not yet passed into model forwards because MLX-LM model call signatures vary and the current scheduler contract only owns the input-id batch.

The cache policy remains a local token bookkeeping abstraction. It is not a paged KV cache and does not implement cross-request KV reuse.
