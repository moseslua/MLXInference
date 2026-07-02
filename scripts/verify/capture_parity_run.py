from __future__ import annotations

import argparse
import json
from pathlib import Path

from mlx_inference.modeling.loader import load_model
from mlx_inference.runtime.generate import generate_greedy


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--prompt-file", required=True)
    parser.add_argument("--reference", default="mlx-lm")
    parser.add_argument("--candidate", default="mlx_inference")
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    prompt = Path(args.prompt_file).read_text()
    loaded = load_model(args.model_id)
    result = generate_greedy(
        model=loaded.model,
        tokenizer=loaded.tokenizer,
        prompt=prompt,
        max_new_tokens=8,
        temperature=0.0,
        seed=0,
    )
    payload = {
        "model_id": args.model_id,
        "reference": args.reference,
        "candidate": args.candidate,
        "candidate_text": result.text,
        "candidate_token_ids": result.token_ids,
        "owned_decode": result.metadata["owned_decode"],
    }
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "parity.json").write_text(json.dumps(payload, indent=2, sort_keys=True))
    print(out_dir / "parity.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
