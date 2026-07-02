from __future__ import annotations

import argparse
import json
import sys

from mlx_inference.modeling.loader import load_model
from mlx_inference.runtime.generate import generate_greedy


def add_generate_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", required=True, help="MLX-LM model id/path or fake://deterministic")
    parser.add_argument("--prompt", required=True, help="Prompt text")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--json", action="store_true", dest="as_json")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m mlx_inference.cli.generate")
    add_generate_arguments(parser)
    return parser


def run_generate(args: argparse.Namespace) -> int:
    try:
        loaded = load_model(args.model)
        result = generate_greedy(
            model=loaded.model,
            tokenizer=loaded.tokenizer,
            prompt=args.prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            seed=args.seed,
        )
    except Exception as exc:
        print(f"Unable to load model or generate output: {exc}", file=sys.stderr)
        return 2

    if args.as_json:
        print(
            json.dumps(
                {
                    "model_id": args.model,
                    "text": result.text,
                    "token_ids": result.token_ids,
                    "stop_reason": result.stop_reason,
                    "owned_decode": result.metadata["owned_decode"],
                    "decode_steps": result.timing.decode_steps,
                },
                indent=None,
                sort_keys=True,
            )
        )
    else:
        print(result.text)
    return 0


def main(argv: list[str] | None = None) -> int:
    return run_generate(build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
