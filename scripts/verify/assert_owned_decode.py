from __future__ import annotations

import argparse
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package", required=True)
    parser.add_argument("--forbid", action="append", default=[])
    args = parser.parse_args()

    offenders: list[str] = []
    root = Path(args.package)
    for path in list((root / "runtime").rglob("*.py")) + list((root / "cli").rglob("*.py")):
        text = path.read_text()
        for forbidden in args.forbid:
            if forbidden in text:
                offenders.append(f"{path}:{forbidden}")

    if offenders:
        print("\n".join(offenders))
        return 1
    print("owned decode invariant holds")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
