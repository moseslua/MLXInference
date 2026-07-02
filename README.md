# MLX Inference Runtime

A small Apple Silicon LLM inference runtime built on MLX.

This project uses MLX-LM as the model loader and reference oracle, but the `mlx_inference` package owns the runtime behavior: step-wise decoding, logits processing, cache policy, scheduler traces, timing boundaries, benchmark capture, CLI behavior, and completion evidence.

The core invariant is:

```text
MLX-LM = oracle/reference/model loader
mlx_inference = decode loop, cache policy, scheduler, benchmark harness, CLI, evidence gate
```

Production runtime and CLI code must not call `mlx_lm.generate` or `mlx_lm.stream_generate`.

## Status

The ULW execution run is complete.

- Goals complete: `11/11`
- Criteria passed: `33/33`
- Final unit/component run: `14 passed`
- Final owned-decode static check: passed
- Final scheduler replay: passed
- Final gate positive/negative checks: passed
- Code review: `APPROVE`
- Manual QA: `APPROVE`
- Gate review: `APPROVE`

Primary evidence lives under:

- `artifacts/verification/2026-07-02/`
- `artifacts/manual/2026-07-02/`
- `artifacts/benchmarks/2026-07-02/`
- `artifacts/reviews/2026-07-02/`
- `.omo/ulw-loop/mlx-inference-runtime/ledger.jsonl`

## What This Is

This is not a wrapper around `mlx_lm.generate`.

The owned generation path is:

```text
input token slice -> model(input, cache) -> logits -> logits processors
-> argmax/sample -> next token -> cache/state update -> stop check
```

The runtime is intentionally small. It is meant to show inference-runtime ownership on Apple Silicon before making larger serving or kernel claims.

## What This Is Not

This project does not claim:

- production serving readiness
- vLLM-equivalent continuous batching
- paged KV cache
- tensor parallelism
- speculative decoding
- CUDA Graphs
- NCCL
- CUDA memory-pool behavior
- custom Metal kernel speedups

The first implementation proves correctness, ownership boundaries, scheduler/cache shape, CLI behavior, benchmark capture, and evidence discipline. Performance optimization is deferred until profile data justifies it.

## Project Layout

```text
.
+-- pyproject.toml
+-- .python-version
+-- src/mlx_inference/
|   +-- cache/
|   |   +-- policy.py
|   +-- cli/
|   |   +-- __main__.py
|   |   +-- generate.py
|   +-- modeling/
|   |   +-- loader.py
|   +-- runtime/
|   |   +-- decode.py
|   |   +-- generate.py
|   |   +-- ops.py
|   +-- scheduler/
|   |   +-- core.py
|   +-- testing/
|       +-- fake.py
+-- tests/
|   +-- component/
|   +-- fixtures/
|   +-- unit/
+-- scripts/verify/
+-- docs/
+-- artifacts/
```

## Environment

The project is managed with `uv` and Python 3.12.

Python constraints:

```text
requires-python = ">=3.12,<3.13"
```

Runtime dependencies:

- `mlx`
- `mlx-lm`
- `numpy`

Development dependency:

- `pytest`

Set up the environment:

```sh
uv sync
```

Run an import smoke test:

```sh
uv run python -c "import sys, mlx, mlx_lm, mlx_inference; assert sys.version_info[:2] == (3, 12); print('ok')"
```

## CLI Usage

Show CLI help:

```sh
uv run python -m mlx_inference.cli --help
```

Run deterministic fake generation:

```sh
uv run python -m mlx_inference.cli.generate \
  --model fake://deterministic \
  --prompt abc \
  --max-new-tokens 3 \
  --temperature 0 \
  --seed 7 \
  --json
```

Expected shape:

```json
{"decode_steps": 3, "model_id": "fake://deterministic", "owned_decode": true, "stop_reason": "max_tokens", "text": "<5> <6> <7>", "token_ids": [5, 6, 7]}
```

Run with a real MLX-LM model id or local model path:

```sh
uv run python -m mlx_inference.cli.generate \
  --model mlx-community/Qwen2.5-0.5B-Instruct-4bit \
  --prompt "Write one sentence about Apple Silicon." \
  --max-new-tokens 32 \
  --temperature 0 \
  --seed 7 \
  --json
```

The real model command may require network access and Hugging Face permissions. The checked-in verification path uses `fake://deterministic` so tests and evidence do not depend on model downloads.

## Runtime Ownership

`src/mlx_inference/runtime/generate.py` owns the loop in `generate_greedy`.

Responsibilities:

- resolve prompt or explicit input token ids
- create per-step MLX arrays
- call `decode_step`
- pass and receive cache state
- select stop conditions
- accumulate generated token ids
- track decode timing
- expose `metadata["owned_decode"] = True`

`src/mlx_inference/runtime/decode.py` owns a single decode step.

Responsibilities:

- call the loaded model object directly
- extract logits and cache from tuple or dictionary outputs
- apply logits processors
- mark an `mx.eval` boundary
- select tokens by greedy argmax or temperature sampling
- return logits, selected token, cache, timing, and metadata

Static guard:

```sh
uv run python scripts/verify/assert_owned_decode.py \
  --package src/mlx_inference \
  --forbid mlx_lm.generate \
  --forbid mlx_lm.stream_generate
```

Expected output:

```text
owned decode invariant holds
```

## MLX-LM Boundary

`src/mlx_inference/modeling/loader.py` is the only production substrate bridge.

It uses:

```python
from mlx_lm import load
```

It does not use:

```python
mlx_lm.generate
mlx_lm.stream_generate
```

The loader supports:

- `fake://deterministic` for local deterministic tests
- MLX-LM model id or local model path loading
- optional returned config handling
- explicit runtime errors when `mlx_lm` import or model loading fails

## Cache Taxonomy

The project separates four cache concepts:

1. MLX-LM native cache wrapper.
2. Local `mlx_inference` cache policy abstraction.
3. Prompt-cache persistence and reuse.
4. Future block/paged cache design.

Only the first three are phase-1 runtime claims.

`src/mlx_inference/cache/policy.py` implements:

- per-request cache records
- request isolation
- token append
- reset and drop
- optional max-token bound
- optional rotating truncation
- prompt-cache save/load
- explicit rejection of paged-cache claims

Paged KV cache is not implemented. Calling `require_paged_cache()` raises `PagedCacheNotImplementedError` with the reason that paged KV requires token-index to physical-block mapping.

## Scheduler

`src/mlx_inference/scheduler/core.py` implements a small request scheduler over real batched model forwards.

The scheduler path is:

```text
active_requests -> token slice -> model forward -> per-request next tokens -> cache update
```

The trace records:

- `submit`
- `model_forward_batch`
- `next_token`
- `cache_update`
- `completed`

Captured scheduler trace example:

```json
{"active_requests": ["r1", "r2"], "event": "model_forward_batch", "token_slice": [[2], [5]]}
{"event": "next_token", "request_id": "r1", "token_id": 3}
{"cache_tokens": [2, 3], "event": "cache_update", "request_id": "r1", "token_id": 3}
```

Replay the trace:

```sh
uv run python scripts/verify/replay_scheduler_trace.py \
  --trace artifacts/verification/2026-07-02/scheduler-trace.jsonl
```

Expected output:

```text
scheduler trace replay ok
```

## Operator Tests

`src/mlx_inference/runtime/ops.py` contains MLX implementations used for operator-level tests:

- RMSNorm
- RoPE
- causal attention
- gated MLP

These are correctness tests and reference formulas, not custom Metal kernels.

## Verification Commands

Run unit and component tests:

```sh
uv run python -m pytest tests/unit tests/component -q
```

Run with JUnit output:

```sh
uv run python -m pytest tests/unit tests/component -q \
  --junitxml artifacts/verification/2026-07-02/unit-component-junit.xml
```

Run collect-only:

```sh
uv run python -m pytest --collect-only -q tests
```

Capture the environment:

```sh
uv run python scripts/verify/capture_env.py \
  --out artifacts/verification/2026-07-02/env.json
```

Capture a scheduler trace:

```sh
uv run python scripts/verify/capture_scheduler_trace.py \
  --model-id fake://deterministic \
  --out artifacts/verification/2026-07-02/scheduler-trace.jsonl
```

Replay a scheduler trace:

```sh
uv run python scripts/verify/replay_scheduler_trace.py \
  --trace artifacts/verification/2026-07-02/scheduler-trace.jsonl
```

Run the final gate:

```sh
uv run python scripts/verify/final_gate.py \
  --date 2026-07-02 \
  --out artifacts/verification/2026-07-02/final-report.md
```

Run the negative final-gate demo:

```sh
uv run python scripts/verify/final_gate.py \
  --date 2026-07-02 \
  --out artifacts/verification/2026-07-02/final-gate-negative.json \
  --require-missing-demo
```

That command is expected to exit nonzero because it intentionally requires a missing artifact.

## Benchmarking

Run the benchmark harness with the deterministic fake model:

```sh
uv run python scripts/verify/benchmark_runtime.py \
  --model-id fake://deterministic \
  --prompt-lens 4 \
  --output-lens 3 \
  --batch-sizes 1 \
  --runs 2 \
  --warmup-runs 1 \
  --out artifacts/benchmarks/2026-07-02/qwen25-05b-benchmark.json
```

The benchmark JSON records:

- model id
- prompt length
- output length
- batch size
- seed
- dtype
- device
- quantization
- prefill tokens per second
- decode tokens per second
- time to first token
- peak RSS when available
- machine metadata

Validate benchmark JSON:

```sh
uv run python scripts/verify/validate_benchmark_json.py \
  --input artifacts/benchmarks/2026-07-02/qwen25-05b-benchmark.json
```

Validate a malformed benchmark fixture is rejected:

```sh
uv run python scripts/verify/validate_benchmark_json.py \
  --input tests/fixtures/bad_benchmark.json \
  --expect-fail
```

## Evidence Artifacts

Important final artifacts:

```text
artifacts/verification/2026-07-02/final-unit-component-output.txt
artifacts/verification/2026-07-02/final-owned-decode.txt
artifacts/verification/2026-07-02/final-scheduler-replay.txt
artifacts/verification/2026-07-02/final-report.md
artifacts/verification/2026-07-02/final-gate-output.txt
artifacts/verification/2026-07-02/final-gate-negative.txt
artifacts/verification/2026-07-02/quality-gate.json
artifacts/manual/2026-07-02/cli/help.txt
artifacts/manual/2026-07-02/cli/generate-greedy.txt
artifacts/manual/2026-07-02/cli/bad-model.txt
artifacts/benchmarks/2026-07-02/qwen25-05b-benchmark.json
artifacts/benchmarks/2026-07-02/bad-benchmark-negative.txt
artifacts/reviews/2026-07-02/code-review.md
artifacts/reviews/2026-07-02/manual-qa.md
artifacts/reviews/2026-07-02/gate-review.md
```

ULW loop state:

```text
.omo/ulw-loop/mlx-inference-runtime/brief.md
.omo/ulw-loop/mlx-inference-runtime/goals.json
.omo/ulw-loop/mlx-inference-runtime/ledger.jsonl
```

Final status:

```text
11 goals complete
33 criteria pass
aggregate complete
```

## Architecture Notes

The CUDA-to-MLX adaptation is documented in `docs/architecture.md`.

Main mapping:

| CUDA concept | MLX reality | phase-1 treatment | deferred or dropped |
| --- | --- | --- | --- |
| CUDA Graphs | MLX lazy compute graphs materialized by `mx.eval` | explicit eval/timing boundaries | no CUDA Graph implementation |
| VRAM/H2D memory pools | Apple Silicon unified memory | avoid CUDA allocator design | no CUDA memory-pool port |
| Paged KV cache | prompt/native caches exist, but paged KV requires physical block mapping | explicit cache taxonomy and local policy | future design only |
| Continuous batching | small scheduler over token slices | real batched model forward trace | no vLLM serving claim |
| Fused CUDA kernels | MLX primitives dispatch optimized Metal operations | formula tests first | custom Metal only after profiling |
| NCCL/tensor parallelism | single-machine MLX runtime | out of scope | no NCCL or tensor parallelism |
| Quantization | MLX-LM can load quantized models | benchmark metadata records quantization | no custom quant kernels |
| Speculative decoding | requires draft policy and acceptance tests | out of scope | deferred |

## Optimization Backlog

Optimization starts after correctness and evidence.

Current order:

1. Tighten `mx.eval` timing boundaries.
2. Decouple from MLX-LM where loader evidence justifies it.
3. Add prefill chunking for long prompts.
4. Improve scheduler batching after mixed-request traces are stable.
5. Tune cache policy limits and prompt-cache reuse.
6. Measure quantized weights and possible KV storage choices.
7. Design future block/paged KV cache with token-index to physical-block mapping.
8. Consider custom Metal kernels only after profile data shows MLX primitives are the bottleneck.

Every optimization needs same-machine before/after benchmark evidence plus unchanged behavior evidence.

## Ambitious Next Steps

The current runtime proves the important boundary: MLX-LM loads and validates the model substrate, while `mlx_inference` owns the step-wise decode loop, cache policy, scheduler trace, CLI, benchmark capture, and evidence gate. The ambitious path is to grow that small runtime into a serious Apple Silicon inference system without blurring those boundaries or borrowing claims from CUDA/vLLM systems before the behavior exists.

The next phase should be evidence-led. Each item below needs a failing-first or baseline artifact, a minimal implementation, a same-machine benchmark when performance is involved, and a regression proof that the owned decode invariant still holds.

### 1. Real-Model Parity Matrix

Expand beyond `fake://deterministic` by running a small, controlled model matrix through both MLX-LM oracle paths and `mlx_inference` candidate paths.

Start with:

- one small instruct model
- one base causal model
- one quantized model
- one local model path
- one malformed or unavailable model id

For each model, capture:

- tokenizer output for fixed prompts
- greedy decode token ids at `temperature=0`
- stop reason and EOS behavior
- first-token latency
- decode tokens per second
- loaded config metadata
- failure mode when the model cannot be loaded

The goal is not broad model support as a claim. The goal is to make model compatibility visible, reproducible, and falsifiable.

### 2. Prefill and Decode Separation

The current loop is intentionally simple. A stronger runtime should explicitly split prefill from decode.

Target shape:

```text
prompt tokens -> prefill(model, prompt, cache) -> cache
last prompt token -> decode_step(model, token, cache) -> next token
```

This unlocks clearer timing and better long-prompt behavior:

- prefill latency
- time to first token
- steady-state decode latency
- cache growth by request
- prompt length scaling
- output length scaling

Acceptance requires benchmark JSON with separate prefill and decode fields, plus a trace proving the runtime does not reprocess the whole prompt after the first generated token.

### 3. Scheduler v2: From Trace Proof to Runtime Policy

The current scheduler proves the right path:

```text
active_requests -> token slice -> model forward -> per-request next tokens -> cache update
```

The ambitious next scheduler should become a real policy surface.

Add:

- dynamic admission from a pending queue
- cancellation cleanup
- per-request token budgets
- fairness between short and long requests
- max active request limits
- deterministic trace replay
- scheduler metrics
- explicit stop reasons per request
- cache cleanup after completion

Do not call this vLLM-style continuous batching until mixed-length, mixed-arrival requests share real model-forward steps and the trace proves it.

### 4. Cache v2: Prompt Reuse Before Paged KV

The cache roadmap should stay taxonomized.

Near-term work:

- persist prompt token history
- reload prompt cache by request id
- validate stale-cache rejection
- add cache-size accounting
- expose cache reset and drop through CLI or script surfaces
- measure cache reuse against a no-cache baseline

Paged KV remains a future design until the runtime has:

- token index to physical block mapping
- block table ownership
- allocation and free lists
- reuse policy
- eviction policy
- multi-request isolation tests
- adversarial stale-block tests

Until those exist, the README and docs should continue to say "future block/paged cache design", not "paged KV cache".

### 5. Streaming CLI and Local Server

Once the decode loop and scheduler are stable, add user-facing streaming.

CLI target:

```sh
uv run python -m mlx_inference.cli.generate \
  --model fake://deterministic \
  --prompt abc \
  --max-new-tokens 32 \
  --stream
```

Server target:

- local-only HTTP endpoint
- JSON request body
- streaming response option
- cancellation path
- health endpoint
- model metadata endpoint
- no production-serving claim

The server should call the same owned runtime APIs as the CLI. It must not become a separate path that bypasses decode, cache, scheduler, or evidence checks.

### 6. Benchmark Matrix and Regression Gates

Turn the benchmark harness into a matrix runner.

Dimensions:

- model id
- prompt length
- output length
- batch size
- quantization
- seed
- scheduler on/off
- cache policy on/off
- prefill chunk size
- hardware and OS metadata

Every benchmark JSON should include enough metadata to prevent false comparisons across machines. Regression checks should compare only same-machine, same-model, same-config runs.

Useful derived reports:

- time to first token by prompt length
- decode tokens per second by output length
- throughput by batch size
- memory use by prompt length
- scheduler overhead by active request count
- cache reuse win/loss summary

### 7. Profiling Before Metal Kernels

Custom Metal kernels are a later phase, not a starting point.

Before writing any `mx.fast.metal_kernel`, capture:

- operator-level timing
- end-to-end decode timing
- MLX primitive baseline
- input shapes
- dtype
- model id
- hardware metadata
- profiler output path

Only consider custom kernels when the profile shows a stable bottleneck in a kernel-shaped operation.

Possible candidates:

- RMSNorm
- RoPE
- attention score path
- attention value path
- gated MLP
- sampling/logits post-processing

Each custom kernel needs:

- formula parity test
- dtype tolerance table
- shape coverage
- fallback implementation
- benchmark before and after
- rollback decision if it loses or complicates the runtime without measurable gain

### 8. Quantization and Memory Experiments

MLX-LM already supports quantized model loading. The runtime should record and compare quantization behavior before implementing custom quantized kernels.

Measure:

- model load time
- first token latency
- decode throughput
- memory footprint
- quality of deterministic token parity where applicable
- cache memory growth

Potential experiments:

- different MLX-LM quantized model variants
- KV cache dtype choices
- prompt-cache persistence size
- quantized weight metadata reporting
- memory pressure behavior on long prompts

Claims must stay narrow: "measured this model/config on this machine", not general quantization conclusions.

### 9. API Hardening and Types

The runtime should become harder to misuse.

Add:

- explicit request and generation config objects
- structured error types
- typed scheduler events
- typed benchmark payloads
- JSON schema for artifacts
- stricter CLI validation
- stable public exports
- compatibility tests for artifact schema changes

The goal is to make behavior reviewable and integration-safe without hiding the simple runtime model under unnecessary abstraction.

### 10. CI and Release Hygiene

The project is not currently a git repository. If it becomes one, the next serious step is release hygiene.

Add:

- lint or formatting check
- unit/component test job
- owned-decode static guard job
- final-gate smoke job
- artifact schema validation job
- optional real-model parity job behind credentials/cache
- README command drift check

Release artifacts should include:

- package version
- environment capture
- benchmark JSON
- final gate report
- reviewer reports
- known limits for that version

### 11. Observability and Trace Design

The scheduler trace is already useful. Make it a first-class runtime artifact.

Extend traces with:

- request lifecycle events
- queue wait time
- model forward start/end
- decode step duration
- selected token
- stop reason
- cache token count
- cache policy decision
- cancellation cleanup
- error event

Trace replay should validate invariants, not just parse JSONL.

Examples:

- every `next_token` must follow a `model_forward_batch`
- every completed request must have a stop reason
- every generated token must have a cache update
- cancelled requests must not continue generating
- over-budget requests must be rejected before model forward

### 12. Documentation as Evidence

Keep documentation tied to artifacts.

Every major README claim should point to one of:

- source path
- test path
- script path
- artifact path
- ledger entry
- reviewer report

Avoid unsupported language like "production ready", "paged KV", "continuous batching", or "optimized Metal kernels" until the implementation and artifacts prove it.

The strongest portfolio signal is not that the runtime is huge. It is that the runtime makes precise claims, owns the critical loop, and carries evidence for each claim.

## Development Rules

Keep these invariants intact:

- Do not call `mlx_lm.generate` or `mlx_lm.stream_generate` from production runtime or CLI code.
- Keep MLX-LM as loader/oracle/model substrate.
- Keep decode ownership in `mlx_inference.runtime`.
- Keep scheduler claims tied to real model-forward traces.
- Keep cache claims taxonomized.
- Do not claim paged KV cache until token-index to physical-block mapping exists.
- Do not add custom Metal kernels before profiling justifies them.
- Do not report performance claims without JSON benchmark evidence.

Recommended pre-completion check:

```sh
uv run python -m pytest tests/unit tests/component -q
uv run python scripts/verify/assert_owned_decode.py --package src/mlx_inference --forbid mlx_lm.generate --forbid mlx_lm.stream_generate
uv run python scripts/verify/replay_scheduler_trace.py --trace artifacts/verification/2026-07-02/scheduler-trace.jsonl
uv run python scripts/verify/final_gate.py --date 2026-07-02 --out artifacts/verification/2026-07-02/final-report.md
```

## Known Limits

- Real-model verification depends on local network/model availability and Hugging Face access.
- The reproducible evidence path uses `fake://deterministic`.
- Native MLX-LM cache semantics are not equivalent to paged KV cache.
- Scheduler batching is a small runtime proof, not a production serving scheduler.
- Benchmark numbers are same-machine evidence only and should not be compared across hardware without controls.
- The project is not initialized as a git repository.
