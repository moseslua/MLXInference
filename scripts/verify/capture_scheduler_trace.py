from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from mlx_inference.modeling.loader import load_model
from mlx_inference.scheduler.core import Scheduler, SchedulerRequest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    loaded = load_model(args.model_id)
    scheduler = Scheduler(max_active=2, token_budget=16)
    scheduler.submit(SchedulerRequest(request_id="r1", input_ids=[2], max_new_tokens=2))
    scheduler.submit(SchedulerRequest(request_id="r2", input_ids=[5], max_new_tokens=2))
    results = scheduler.run(loaded.model)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as handle:
        for entry in scheduler.trace:
            handle.write(json.dumps(entry, sort_keys=True) + "\n")
        handle.write(json.dumps({"event": "completed", "results": [asdict(result) for result in results]}, sort_keys=True) + "\n")
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
