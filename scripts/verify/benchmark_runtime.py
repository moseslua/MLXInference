from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path
from statistics import mean
from time import perf_counter

from mlx_inference.modeling.loader import load_model
from mlx_inference.runtime.generate import generate_greedy


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--prompt-lens", nargs="+", type=int, required=True)
    parser.add_argument("--output-lens", nargs="+", type=int, required=True)
    parser.add_argument("--batch-sizes", nargs="+", type=int, required=True)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    loaded = load_model(args.model_id)
    rows: list[dict[str, object]] = []
    for prompt_len in args.prompt_lens:
        for output_len in args.output_lens:
            for batch_size in args.batch_sizes:
                prompt = "x" * prompt_len
                for _ in range(args.warmup_runs):
                    generate_greedy(
                        model=loaded.model,
                        tokenizer=loaded.tokenizer,
                        prompt=prompt,
                        max_new_tokens=output_len,
                        temperature=0.0,
                        seed=0,
                    )
                durations: list[float] = []
                for run_index in range(args.runs):
                    started = perf_counter()
                    result = generate_greedy(
                        model=loaded.model,
                        tokenizer=loaded.tokenizer,
                        prompt=prompt,
                        max_new_tokens=output_len,
                        temperature=0.0,
                        seed=run_index,
                    )
                    durations.append(perf_counter() - started)
                elapsed = mean(durations)
                rows.append(
                    {
                        "model_id": args.model_id,
                        "quantization": loaded.config.get("quantization"),
                        "prompt_len": prompt_len,
                        "output_len": output_len,
                        "batch_size": batch_size,
                        "seed": 0,
                        "dtype": "unknown",
                        "device": "mlx",
                        "prefill_tok_s": float(prompt_len / max(elapsed, 1e-9)),
                        "decode_tok_s": float(output_len / max(elapsed, 1e-9)),
                        "ttft_ms": float(result.timing.step_ms[0] if result.timing.step_ms else 0.0),
                        "peak_rss_mb": None,
                    }
                )

    payload = {
        "machine": {
            "system": platform.system(),
            "machine": platform.machine(),
            "release": platform.release(),
        },
        "runs": rows,
    }
    validate_payload(payload)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True))
    print(out)
    return 0


def validate_payload(payload: dict[str, object]) -> None:
    runs = payload.get("runs")
    if not isinstance(runs, list) or not runs:
        raise ValueError("Benchmark JSON must contain non-empty runs")
    required = {
        "model_id",
        "quantization",
        "prompt_len",
        "output_len",
        "batch_size",
        "seed",
        "dtype",
        "device",
        "prefill_tok_s",
        "decode_tok_s",
        "ttft_ms",
        "peak_rss_mb",
    }
    for index, row in enumerate(runs):
        if not isinstance(row, dict):
            raise ValueError(f"Benchmark run {index} is not an object")
        missing = sorted(required - set(row))
        if missing:
            raise ValueError(f"Benchmark run {index} missing keys: {missing}")


if __name__ == "__main__":
    raise SystemExit(main())
