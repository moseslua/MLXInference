from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from benchmark_runtime import validate_payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--expect-fail", action="store_true")
    args = parser.parse_args()

    try:
        validate_payload(json.loads(Path(args.input).read_text()))
    except Exception as exc:
        if args.expect_fail:
            print(f"expected failure: {exc}")
            return 0
        print(str(exc), file=sys.stderr)
        return 1

    if args.expect_fail:
        print("expected validation failure, but input passed", file=sys.stderr)
        return 1
    print("ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
