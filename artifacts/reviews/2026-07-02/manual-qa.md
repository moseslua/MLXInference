# Manual QA Report - MLX Inference Runtime

Date: 2026-07-02
Scope: `/Users/moses/MLXInference`
Evidence directory: `artifacts/reviews/2026-07-02/manual-qa-evidence/`

## Summary

Final manual QA re-inspected the requested verification, manual, benchmark, brief, goal, and ledger artifacts, then reran bounded CLI/data-surface scenarios for the required runtime surfaces.

Verdict: APPROVE

## Inspected Artifacts

- `artifacts/verification/2026-07-02/` - present with non-empty env, import, collect-only, scheduler, gate, unit/component, parity, and negative evidence artifacts.
- `artifacts/manual/2026-07-02/` - present with non-empty CLI help, deterministic generation, bad model, and docs artifacts.
- `artifacts/benchmarks/2026-07-02/` - present with non-empty benchmark JSON and benchmark output.
- `.omo/ulw-loop/mlx-inference-runtime/brief.md` - present, non-empty.
- `.omo/ulw-loop/mlx-inference-runtime/goals.json` - present, non-empty.
- `.omo/ulw-loop/mlx-inference-runtime/ledger.jsonl` - present, non-empty.

Note: repo-root `goals.json` and `ledger.jsonl` do not exist. The active goal and ledger files are under `.omo/ulw-loop/mlx-inference-runtime/`, matching the brief path and embedded goal metadata.

## Commands Run

- `find artifacts/verification/2026-07-02 artifacts/manual/2026-07-02 artifacts/benchmarks/2026-07-02 .omo/ulw-loop/mlx-inference-runtime -maxdepth 3 -type f -print`
- `wc -c .omo/ulw-loop/mlx-inference-runtime/brief.md .omo/ulw-loop/mlx-inference-runtime/goals.json .omo/ulw-loop/mlx-inference-runtime/ledger.jsonl ...`
- `uv run python -m mlx_inference.cli --help`
- `uv run python -m mlx_inference.cli.generate --model fake://deterministic --prompt abc --max-new-tokens 3 --temperature 0 --seed 7 --json`
- `uv run python -m mlx_inference.cli.generate --model bad-id --prompt hi`
- `uv run python scripts/verify/capture_env.py --out artifacts/reviews/2026-07-02/manual-qa-evidence/env.json`
- `uv run python -c "import mlx, mlx_lm, mlx_inference; print('import smoke ok')"`
- `uv run pytest --collect-only`
- `uv run python scripts/verify/benchmark_runtime.py --model-id fake://deterministic --prompt-lens 4 --output-lens 3 --batch-sizes 1 --runs 2 --warmup-runs 1 --out artifacts/reviews/2026-07-02/manual-qa-evidence/benchmark.json`
- `uv run python scripts/verify/validate_benchmark_json.py --input tests/fixtures/bad_benchmark.json --expect-fail`
- `uv run python scripts/verify/capture_scheduler_trace.py --model-id fake://deterministic --out artifacts/reviews/2026-07-02/manual-qa-evidence/scheduler-trace.jsonl`
- `uv run python scripts/verify/replay_scheduler_trace.py --trace artifacts/reviews/2026-07-02/manual-qa-evidence/scheduler-trace.jsonl`
- `uv run python scripts/verify/final_gate.py --date 2026-07-02 --out artifacts/reviews/2026-07-02/manual-qa-evidence/final-gate-negative.json --require-missing-demo`
- `uv run python scripts/verify/final_gate.py --date 2026-07-02 --out artifacts/reviews/2026-07-02/manual-qa-evidence/final-gate-positive.json`

## manualQa

### surfaceEvidence

| scenario id | criterion reference | surface | exact invocation | verdict | artifactRefs |
| --- | --- | --- | --- | --- | --- |
| S01 | CLI help | CLI stdout | `uv run python -m mlx_inference.cli --help` | PASS | A01 |
| S02 | fake deterministic generation | CLI JSON stdout | `uv run python -m mlx_inference.cli.generate --model fake://deterministic --prompt abc --max-new-tokens 3 --temperature 0 --seed 7 --json` | PASS | A02 |
| S03 | bad model failure | CLI stderr/exit | `uv run python -m mlx_inference.cli.generate --model bad-id --prompt hi` | PASS | A03 |
| S04 | env/import | data/CLI | `uv run python scripts/verify/capture_env.py --out artifacts/reviews/2026-07-02/manual-qa-evidence/env.json`; `uv run python -c "import mlx, mlx_lm, mlx_inference; print('import smoke ok')"` | PASS | A04, A05 |
| S05 | collect-only | pytest collect surface | `uv run pytest --collect-only` | PASS | A06 |
| S06 | benchmark JSON | data/CLI | `uv run python scripts/verify/benchmark_runtime.py --model-id fake://deterministic --prompt-lens 4 --output-lens 3 --batch-sizes 1 --runs 2 --warmup-runs 1 --out artifacts/reviews/2026-07-02/manual-qa-evidence/benchmark.json` | PASS | A07, A08 |
| S07 | scheduler trace/replay | data/CLI | `uv run python scripts/verify/capture_scheduler_trace.py --model-id fake://deterministic --out artifacts/reviews/2026-07-02/manual-qa-evidence/scheduler-trace.jsonl`; `uv run python scripts/verify/replay_scheduler_trace.py --trace artifacts/reviews/2026-07-02/manual-qa-evidence/scheduler-trace.jsonl` | PASS | A10, A11 |
| S08 | final gate positive | data/CLI | `uv run python scripts/verify/final_gate.py --date 2026-07-02 --out artifacts/reviews/2026-07-02/manual-qa-evidence/final-gate-positive.json` | PASS | A14, A15 |

### adversarialCases

| scenario id | criterion reference | adversarial class | expected behavior | verdict | artifactRefs |
| --- | --- | --- | --- | --- | --- |
| A-S03 | bad model failure | invalid model id | CLI exits nonzero and reports actionable model-load failure | PASS | A03 |
| A-S09 | bad benchmark rejection | malformed benchmark schema | validator accepts `--expect-fail`, reports missing required benchmark keys, and exits 0 | PASS | A09 |
| A-S10 | final gate negative | intentionally missing required evidence | final gate exits nonzero and records `status: failed` plus missing artifact path | PASS | A12, A13 |

### artifactRefs

| id | kind | description | path |
| --- | --- | --- | --- |
| A01 | CLI transcript | CLI help output includes `generate` and `exit_code=0` | `artifacts/reviews/2026-07-02/manual-qa-evidence/cli-help.txt` |
| A02 | JSON output | fake deterministic generation output with owned decode and token ids `[5, 6, 7]` | `artifacts/reviews/2026-07-02/manual-qa-evidence/fake-generate.json` |
| A03 | CLI stderr | bad model failure includes `Unable to load model` and nonzero exit recorded separately | `artifacts/reviews/2026-07-02/manual-qa-evidence/bad-model.stderr` |
| A04 | JSON output | captured Python/runtime environment with Python 3.12 | `artifacts/reviews/2026-07-02/manual-qa-evidence/env.json` |
| A05 | CLI transcript | import smoke for `mlx`, `mlx_lm`, and `mlx_inference` | `artifacts/reviews/2026-07-02/manual-qa-evidence/import-smoke.txt` |
| A06 | pytest transcript | collect-only output with collected tests and `exit_code=0` | `artifacts/reviews/2026-07-02/manual-qa-evidence/pytest-collect-only.txt` |
| A07 | JSON output | fresh benchmark JSON with positive throughput fields | `artifacts/reviews/2026-07-02/manual-qa-evidence/benchmark.json` |
| A08 | CLI transcript | benchmark command output and `exit_code=0` | `artifacts/reviews/2026-07-02/manual-qa-evidence/benchmark-output.txt` |
| A09 | CLI transcript | malformed benchmark fixture rejection via `--expect-fail` | `artifacts/reviews/2026-07-02/manual-qa-evidence/bad-benchmark-rejection.txt` |
| A10 | JSONL trace | scheduler trace containing submit, model forward batch, next token, cache update, and completed events | `artifacts/reviews/2026-07-02/manual-qa-evidence/scheduler-trace.jsonl` |
| A11 | CLI transcript | scheduler replay output with `scheduler trace replay ok` | `artifacts/reviews/2026-07-02/manual-qa-evidence/scheduler-replay.txt` |
| A12 | JSON output | final gate negative result with `status: failed` and missing evidence | `artifacts/reviews/2026-07-02/manual-qa-evidence/final-gate-negative.json` |
| A13 | CLI transcript | final gate negative stderr with missing evidence and nonzero exit | `artifacts/reviews/2026-07-02/manual-qa-evidence/final-gate-negative.txt` |
| A14 | JSON output | final gate positive result with `status: passed` | `artifacts/reviews/2026-07-02/manual-qa-evidence/final-gate-positive.json` |
| A15 | CLI transcript | final gate positive command output and `exit_code=0` | `artifacts/reviews/2026-07-02/manual-qa-evidence/final-gate-positive.txt` |

## Evidence Gaps

- Existing artifact `artifacts/benchmarks/2026-07-02/bad-benchmark-negative.txt` is non-empty, but it captures argparse rejection for missing `--input`, not malformed benchmark schema rejection. Fresh QA evidence `A09` covers the required bad benchmark rejection against `tests/fixtures/bad_benchmark.json`.
- `artifacts/reviews/2026-07-02/manual-qa-evidence/bad-model.stdout` and `fake-generate.stderr` are zero-byte sidecar files from command redirection; they are not used as PASS evidence.
- No HTTP, browser, tmux, or desktop GUI surface exists for this runtime scope; CLI/data channels are the faithful surfaces.

APPROVE
