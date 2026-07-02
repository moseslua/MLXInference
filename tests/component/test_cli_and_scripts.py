from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def run_command(args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(args, check=False, text=True, capture_output=True)


def test_cli_help_and_fake_generation() -> None:
    help_result = run_command([sys.executable, "-m", "mlx_inference.cli", "--help"])
    assert help_result.returncode == 0
    assert "generate" in help_result.stdout

    gen_result = run_command(
        [
            sys.executable,
            "-m",
            "mlx_inference.cli.generate",
            "--model",
            "fake://deterministic",
            "--prompt",
            "abc",
            "--max-new-tokens",
            "3",
            "--temperature",
            "0",
            "--seed",
            "7",
            "--json",
        ]
    )
    assert gen_result.returncode == 0, gen_result.stderr
    payload = json.loads(gen_result.stdout)
    assert payload["token_ids"] == [5, 6, 7]
    assert payload["owned_decode"] is True


def test_bad_model_error_is_actionable() -> None:
    result = run_command(
        [
            sys.executable,
            "-m",
            "mlx_inference.cli.generate",
            "--model",
            "bad-id",
            "--prompt",
            "hi",
        ]
    )
    assert result.returncode != 0
    assert "Unable to load model" in result.stderr


def test_env_capture_and_benchmark_schema(tmp_path: Path) -> None:
    env_path = tmp_path / "env.json"
    bench_path = tmp_path / "bench.json"

    env_result = run_command([sys.executable, "scripts/verify/capture_env.py", "--out", str(env_path)])
    assert env_result.returncode == 0, env_result.stderr
    assert json.loads(env_path.read_text())["python"]["major_minor"] == [3, 12]

    bench_result = run_command(
        [
            sys.executable,
            "scripts/verify/benchmark_runtime.py",
            "--model-id",
            "fake://deterministic",
            "--prompt-lens",
            "4",
            "--output-lens",
            "3",
            "--batch-sizes",
            "1",
            "--runs",
            "2",
            "--warmup-runs",
            "1",
            "--out",
            str(bench_path),
        ]
    )
    assert bench_result.returncode == 0, bench_result.stderr
    benchmark = json.loads(bench_path.read_text())
    assert benchmark["runs"][0]["decode_tok_s"] > 0
    assert benchmark["runs"][0]["model_id"] == "fake://deterministic"
