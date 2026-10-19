"""Evaluate a prediction JSON file with exact-title HR and NDCG metrics."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def normalize_title(value: str) -> str:
    value = str(value).strip()
    if len(value) >= 2 and value[0] == value[-1] == '"':
        value = value[1:-1]
    return value.strip()


def evaluate_records(records: list[dict], top_k: list[int]) -> dict[str, float]:
    if not records:
        raise ValueError("Prediction file contains no records")
    ranks = []
    for record in records:
        target = normalize_title(record["output"])
        predictions = [normalize_title(value) for value in record["predict"]]
        try:
            ranks.append(predictions.index(target))
        except ValueError:
            ranks.append(math.inf)

    metrics: dict[str, float] = {}
    count = len(ranks)
    for cutoff in top_k:
        hits = [rank for rank in ranks if rank < cutoff]
        metrics[f"NDCG@{cutoff}"] = sum(
            1.0 / math.log2(rank + 2) for rank in hits
        ) / count
        metrics[f"HR@{cutoff}"] = len(hits) / count
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--predictions",
        required=True,
        help="Prediction JSON produced by inference.py",
    )
    parser.add_argument("--top-k", default="1,5,10")
    parser.add_argument("--output", help="Defaults to <predictions>.metrics.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    predictions = Path(args.predictions).expanduser().resolve()
    with predictions.open("r", encoding="utf-8") as handle:
        records = json.load(handle)
    available = min(len(record["predict"]) for record in records)
    requested = sorted({int(value) for value in args.top_k.split(",")})
    cutoffs = [value for value in requested if 0 < value <= available]
    if not cutoffs:
        raise ValueError(
            f"No requested cutoff is available; each record has {available} predictions"
        )
    metrics = evaluate_records(records, cutoffs)
    output = (
        Path(args.output).expanduser().resolve()
        if args.output
        else predictions.with_suffix(".metrics.json")
    )
    with output.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
    print(json.dumps(metrics, indent=2))
    print(f"Saved metrics to {output}")


if __name__ == "__main__":
    main()
