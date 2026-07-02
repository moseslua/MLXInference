from __future__ import annotations

from pathlib import Path


FORBIDDEN = ("mlx_lm.generate", "mlx_lm.stream_generate", "stream_generate(")


def test_runtime_and_cli_do_not_call_mlx_lm_generation_helpers() -> None:
    roots = [Path("src/mlx_inference/runtime"), Path("src/mlx_inference/cli")]
    offenders: list[str] = []
    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob("*.py"):
            text = path.read_text()
            for token in FORBIDDEN:
                if token in text:
                    offenders.append(f"{path}:{token}")

    assert offenders == []
