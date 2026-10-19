"""Build a sparse item co-occurrence artifact from a dataset training split."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from itertools import combinations
from pathlib import Path

from data_utils import DEFAULT_DATA_ROOT, dataset_paths, load_interactions


def build_cooccurrence(train_path: Path) -> Counter[tuple[int, int]]:
    counts: Counter[tuple[int, int]] = Counter()
    frame = load_interactions(train_path)
    for sequence in frame["item_ids"]:
        for left, right in combinations(sorted(set(sequence)), 2):
            counts[(left, right)] += 1
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT))
    parser.add_argument("--train-sample", type=int)
    parser.add_argument("--output", help="Defaults to data/<dataset>/cooc_matrix.json")
    args = parser.parse_args()

    paths = dataset_paths(
        args.dataset,
        args.data_root,
        train_sample=args.train_sample,
    )
    output = Path(args.output).expanduser().resolve() if args.output else paths.cooccurrence
    output.parent.mkdir(parents=True, exist_ok=True)

    counts = build_cooccurrence(paths.train)
    payload = {
        "format": "coo",
        "source": paths.train.name,
        "pairs": [[left, right, count] for (left, right), count in sorted(counts.items())],
    }
    with output.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, separators=(",", ":"))
    print(f"Saved {len(counts):,} co-occurring item pairs to {output}")


if __name__ == "__main__":
    main()
