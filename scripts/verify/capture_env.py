from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from importlib import metadata
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    payload = {
        "python": {
            "version": platform.python_version(),
            "major_minor": [sys.version_info.major, sys.version_info.minor],
            "executable": sys.executable,
        },
        "platform": {
            "system": platform.system(),
            "machine": platform.machine(),
            "release": platform.release(),
        },
        "packages": {
            "mlx": _version("mlx"),
            "mlx-lm": _version("mlx-lm"),
            "mlx-inference": _version("mlx-inference"),
        },
        "sw_vers": _run(["sw_vers", "-productVersion"]),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True))
    print(out)
    return 0


def _version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def _run(command: list[str]) -> str | None:
    try:
        result = subprocess.run(command, check=False, text=True, capture_output=True)
    except FileNotFoundError:
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip()


if __name__ == "__main__":
    raise SystemExit(main())
