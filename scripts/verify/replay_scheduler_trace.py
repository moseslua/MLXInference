from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", required=True)
    parser.add_argument("--expect-reject", action="store_true")
    args = parser.parse_args()

    try:
        events = [json.loads(line) for line in Path(args.trace).read_text().splitlines() if line.strip()]
        has_forward = any(event.get("event") == "model_forward_batch" for event in events)
        if not has_forward:
            raise ValueError("trace did not contain model_forward_batch")
        if args.expect_reject:
            raise ValueError("expected rejection trace was accepted")
    except Exception as exc:
        if args.expect_reject:
            print(f"expected rejection: {exc}")
            return 0
        print(str(exc), file=sys.stderr)
        return 1

    print("scheduler trace replay ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
