"""Shared dataset loading and validation utilities."""

from __future__ import annotations

import ast
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_ROOT = PROJECT_ROOT / "data"


@dataclass(frozen=True)
class DatasetPaths:
    """All files belonging to one dataset."""

    name: str
    directory: Path
    train: Path
    test: Path
    catalog: Path
    item_embeddings: Path
    cooccurrence: Path


def _resolve_file(directory: Path, split: str, sample_size: int | None) -> Path:
    if sample_size is not None:
        path = directory / f"{split}_{sample_size}.csv"
        if not path.is_file():
            raise FileNotFoundError(f"Missing {split} file: {path}")
        return path

    matches = sorted(directory.glob(f"{split}_*.csv"))
    if len(matches) != 1:
        found = ", ".join(path.name for path in matches) or "none"
        raise FileNotFoundError(
            f"Expected exactly one {split}_*.csv in {directory}; found: {found}. "
            f"Pass --{split}-sample to select a file explicitly."
        )
    return matches[0]


def dataset_paths(
    dataset: str,
    data_root: str | Path = DEFAULT_DATA_ROOT,
    train_sample: int | None = None,
    test_sample: int | None = None,
) -> DatasetPaths:
    root = Path(data_root).expanduser().resolve()
    directory = root / dataset
    if not directory.is_dir():
        raise FileNotFoundError(
            f"Dataset directory does not exist: {directory}. "
            "Create data/<dataset>/ and add the three required source files."
        )
    catalog = directory / "id2name4Rec.json"
    if not catalog.is_file():
        raise FileNotFoundError(f"Missing item catalog: {catalog}")
    return DatasetPaths(
        name=dataset,
        directory=directory,
        train=_resolve_file(directory, "train", train_sample),
        test=_resolve_file(directory, "test", test_sample),
        catalog=catalog,
        item_embeddings=directory / "item_embeddings.pth",
        cooccurrence=directory / "cooc_matrix.json",
    )


def parse_item_ids(value: object) -> list[int]:
    if isinstance(value, str):
        value = ast.literal_eval(value)
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"item_ids must be a list, got {type(value).__name__}")
    item_ids = [int(item_id) for item_id in value if int(item_id) != 0]
    if len(item_ids) < 2:
        raise ValueError("Every interaction sequence must contain at least two non-padding items")
    if any(item_id < 0 for item_id in item_ids):
        raise ValueError("Item IDs must be non-negative integers")
    return item_ids


def load_interactions(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    frame = pd.read_csv(path)
    required = {"user_id", "item_ids"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {', '.join(sorted(missing))}")
    frame = frame.copy()
    frame["user_id"] = frame["user_id"].astype(int)
    try:
        frame["item_ids"] = frame["item_ids"].map(parse_item_ids)
    except Exception as exc:
        raise ValueError(f"Invalid item_ids in {path}: {exc}") from exc
    return frame


def load_catalog(path: str | Path) -> dict[int, str]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    catalog = {int(item_id): str(title) for item_id, title in raw.items()}
    if 0 in catalog:
        raise ValueError(f"{path} uses reserved padding item ID 0")
    if not catalog:
        raise ValueError(f"Item catalog is empty: {path}")
    return catalog


def validate_dataset(paths: DatasetPaths) -> dict[str, int]:
    train = load_interactions(paths.train)
    test = load_interactions(paths.test)
    catalog = load_catalog(paths.catalog)
    interaction_items = {
        item_id
        for sequence in pd.concat([train["item_ids"], test["item_ids"]])
        for item_id in sequence
    }
    missing_titles = interaction_items.difference(catalog)
    if missing_titles:
        examples = sorted(missing_titles)[:10]
        raise ValueError(
            f"{len(missing_titles)} interaction item IDs are absent from {paths.catalog}; "
            f"examples: {examples}"
        )
    return {
        "train_sequences": len(train),
        "test_sequences": len(test),
        "catalog_items": len(catalog),
        "max_item_id": max(interaction_items | set(catalog)),
    }


def iter_history_target(
    frame: pd.DataFrame, catalog: dict[int, str]
) -> Iterable[tuple[int, list[int], int]]:
    for row in frame.itertuples(index=False):
        valid_ids = [item_id for item_id in row.item_ids if item_id in catalog]
        if len(valid_ids) >= 2:
            yield int(row.user_id), valid_ids[:-1], valid_ids[-1]
