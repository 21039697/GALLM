"""Validate released datasets and artifacts without loading an LLM."""

from __future__ import annotations

import argparse
import ast
import json
import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data_utils import (
    DEFAULT_DATA_ROOT,
    dataset_paths,
    load_catalog,
    load_interactions,
    validate_dataset,
)


def check_artifacts(paths) -> dict:
    stats = validate_dataset(paths)
    catalog = load_catalog(paths.catalog)
    weights = torch.load(paths.item_embeddings, map_location="cpu", weights_only=True)
    if not isinstance(weights, torch.Tensor) or weights.ndim != 2:
        raise ValueError("item_embeddings.pth must contain a two-dimensional tensor")
    if not weights.is_floating_point() or weights.shape[1] == 0:
        raise ValueError("Item embeddings must have a positive width and floating dtype")
    if not torch.isfinite(weights).all():
        raise ValueError("Item embeddings contain non-finite values")
    frames = [load_interactions(paths.train), load_interactions(paths.test)]
    used_ids = {item for frame in frames for seq in frame.item_ids for item in seq}
    missing = sorted(item for item in used_ids if item >= weights.shape[0])
    if missing:
        raise ValueError(f"Interaction IDs have no embedding rows: {missing[:10]}")

    with paths.cooccurrence.open(encoding="utf-8") as handle:
        graph = json.load(handle)
    if not isinstance(graph, dict):
        raise ValueError("Co-occurrence artifact must be a JSON object")
    if graph.get("format") == "coo":
        records = graph["pairs"]
        if graph.get("source") != paths.train.name:
            raise ValueError("Co-occurrence source does not match the selected training CSV")
    else:
        records = [(*ast.literal_eval(key), count) for key, count in graph.items()]
    seen = set()
    for left, right, count in records:
        if type(left) is not int or type(right) is not int:
            raise ValueError("Co-occurrence IDs must be integers")
        if left == right or left not in catalog or right not in catalog:
            raise ValueError(f"Invalid co-occurrence pair: {(left, right)}")
        if not isinstance(count, (int, float)) or not math.isfinite(count) or count <= 0:
            raise ValueError(f"Invalid co-occurrence count: {count}")
        pair = tuple(sorted((left, right)))
        if pair in seen:
            raise ValueError(f"Duplicate co-occurrence pair: {pair}")
        seen.add(pair)
    return {
        **stats,
        "embedding_shape": list(weights.shape),
        "cooccurrence_pairs": len(seen),
        "catalog_only_ids_without_embeddings": sum(
            item >= weights.shape[0] for item in catalog
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", help="Default: check every dataset directory")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--train-sample", type=int)
    parser.add_argument("--test-sample", type=int)
    args = parser.parse_args()
    names = [args.dataset] if args.dataset else sorted(
        entry.name for entry in args.data_root.iterdir() if entry.is_dir()
    )
    if not names:
        parser.error("No dataset directories found")
    failures = 0
    for name in names:
        try:
            paths = dataset_paths(name, args.data_root, args.train_sample, args.test_sample)
            print(json.dumps({"dataset": name, "status": "ok", **check_artifacts(paths)}))
        except (OSError, ValueError, TypeError, KeyError, SyntaxError, RuntimeError) as exc:
            failures += 1
            print(json.dumps({"dataset": name, "status": "error", "error": str(exc)}))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
