from __future__ import annotations

import argparse

from mlx_inference.cli.generate import run_generate


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="mlx-inference", description="MLX inference runtime CLI")
    subparsers = parser.add_subparsers(dest="command")
    generate_parser = subparsers.add_parser("generate", help="Generate text with the owned decode loop")
    from mlx_inference.cli.generate import add_generate_arguments

    add_generate_arguments(generate_parser)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "generate":
        return run_generate(args)
    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
