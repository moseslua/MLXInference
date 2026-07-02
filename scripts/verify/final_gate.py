from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REQUIRED_RELATIVE = [
    "env.json",
    "unit-component-junit.xml",
    "owned-decode.txt",
    "scheduler-trace.jsonl",
]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--artifact-root", default="artifacts")
    parser.add_argument("--require-missing-demo", action="store_true")
    args = parser.parse_args()

    root = Path(args.artifact_root)
    verification = root / "verification" / args.date
    manual_cli = root / "manual" / args.date / "cli" / "generate-greedy.txt"
    benchmark = root / "benchmarks" / args.date / "qwen25-05b-benchmark.json"
    required = [verification / name for name in REQUIRED_RELATIVE] + [manual_cli, benchmark]
    if args.require_missing_demo:
        required.append(verification / "intentionally-missing.txt")
    missing = [path for path in required if not path.exists() or path.stat().st_size == 0]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "date": args.date,
        "status": "failed" if missing else "passed",
        "missing": [str(path) for path in missing],
        "checked": [str(path) for path in required],
    }
    out.write_text(json.dumps(report, indent=2, sort_keys=True))
    if missing:
        print(f"missing evidence: {[str(path) for path in missing]}", file=sys.stderr)
        return 1
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
