"""Export item embeddings from a SASRec checkpoint."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

SASREC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SASREC_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data_utils import DEFAULT_DATA_ROOT, dataset_paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT))
    parser.add_argument("--train-sample", type=int)
    parser.add_argument("--test-sample", type=int)
    parser.add_argument("--checkpoint")
    parser.add_argument("--output")
    args = parser.parse_args()

    paths = dataset_paths(
        args.dataset,
        args.data_root,
        train_sample=args.train_sample,
        test_sample=args.test_sample,
    )
    checkpoint = (
        Path(args.checkpoint).expanduser().resolve()
        if args.checkpoint
        else paths.directory / "sasrec_best.pt"
    )
    output = (
        Path(args.output).expanduser().resolve()
        if args.output
        else paths.item_embeddings
    )
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    try:
        state = payload.get("model_state", payload)
        embeddings = state["item_emb.weight"]
    except (KeyError, TypeError) as exc:
        raise ValueError(f"Unsupported SASRec checkpoint: {checkpoint}") from exc
    torch.save(embeddings, output)
    print(f"Saved item embeddings with shape {tuple(embeddings.shape)} to {output}")


if __name__ == "__main__":
    main()
