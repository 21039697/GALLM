"""Train the original full-sequence SASRec and export its item embeddings."""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

SASREC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SASREC_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data_utils import DEFAULT_DATA_ROOT, dataset_paths, load_interactions, validate_dataset
from SASRec.dataloader import NextItemDataset
from SASRec.model import SASRec, SASRecConfig


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def evaluate(
    model: SASRec,
    loader: DataLoader,
    sample_count: int,
    device: torch.device,
    top_k: tuple[int, ...] = (1, 5, 10, 20),
) -> dict[str, float]:
    ndcg_totals = torch.zeros(len(top_k))
    hit_totals = torch.zeros(len(top_k))
    with torch.no_grad():
        for sequences, targets in loader:
            sequences = sequences.to(device)
            targets = targets.to(device)
            ranking_scores = -model(sequences)
            ranks = ranking_scores.argsort(dim=-1).argsort(dim=-1)
            target_ranks = ranks[
                torch.arange(ranks.shape[0], device=device), targets
            ]
            for index, cutoff in enumerate(top_k):
                hits = target_ranks.lt(cutoff)
                hit_totals[index] += hits.sum().item()
                ndcg_totals[index] += (
                    1 / torch.log(target_ranks[hits] + 2)
                ).sum().item()

    ndcg = ndcg_totals / (sample_count * (1.0 / math.log(2)))
    hit_rate = hit_totals / sample_count
    metrics = {
        f"NDCG@{cutoff}": round(ndcg[index].item(), 4)
        for index, cutoff in enumerate(top_k)
    }
    metrics.update(
        {
            f"HR@{cutoff}": round(hit_rate[index].item(), 4)
            for index, cutoff in enumerate(top_k)
        }
    )
    metrics["default"] = metrics["NDCG@10"]
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT))
    parser.add_argument("--train-sample", type=int)
    parser.add_argument("--test-sample", type=int)
    parser.add_argument("--batch-size", "--batchsize", type=int, default=256)
    parser.add_argument("--learning-rate", "--lr", type=float, default=0.001)
    parser.add_argument("--epochs", "--num-epochs", "--num_epochs", type=int, default=201)
    parser.add_argument("--dropout", "--dropout-rate", type=float, default=0.5)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--hidden-size", "--hidden-units", type=int, default=64)
    parser.add_argument("--max-length", "--maxlen", type=int, default=50)
    parser.add_argument("--num-layers", "--num-blocks", type=int, default=2)
    parser.add_argument("--num-heads", type=int, default=1)
    parser.add_argument("--train-sequence-length", type=int, default=11)
    parser.add_argument("--test-sequence-length", type=int, default=13)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--cuda", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--state-dict-path")
    parser.add_argument("--output-dir")
    parser.add_argument("--max-train-rows", type=int)
    parser.add_argument("--max-test-rows", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    paths = dataset_paths(
        args.dataset,
        args.data_root,
        train_sample=args.train_sample,
        test_sample=args.test_sample,
    )
    stats = validate_dataset(paths)
    train_frame = load_interactions(paths.train)
    test_frame = load_interactions(paths.test)
    # Keep an embedding for every catalog ID, including unseen catalog items.
    item_count = stats["max_item_id"]
    if args.max_train_rows:
        train_frame = train_frame.head(args.max_train_rows)
    if args.max_test_rows:
        test_frame = test_frame.head(args.max_test_rows)

    device = torch.device(
        f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu"
    )
    config = SASRecConfig(
        item_count=item_count,
        hidden_size=args.hidden_size,
        max_length=args.max_length,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        dropout=args.dropout,
    )
    if max(args.train_sequence_length, args.test_sequence_length) > args.max_length:
        raise ValueError("Train and test sequence lengths cannot exceed --max-length")
    model = SASRec(config).to(device)
    if args.state_dict_path:
        payload = torch.load(
            Path(args.state_dict_path).expanduser().resolve(),
            map_location=device,
            weights_only=True,
        )
        state = payload.get("model_state", payload)
        model.load_state_dict(state)

    train_dataset = NextItemDataset(train_frame, args.train_sequence_length)
    test_dataset = NextItemDataset(test_frame, args.test_sequence_length)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else SASREC_DIR
        / "log"
        / "SASRec_full"
        / args.dataset
        / datetime.now().strftime("%m-%d-%Hh%Mm%Ss")
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    patience_count = 0
    best_ndcg = 0.0
    best_state = None
    best_metrics = None
    stopped_epoch = 0
    start_time = time.time()
    for epoch in range(args.epochs):
        print("=" * 35)
        print(f"Start Training Epoch {epoch}")
        model.train()
        for batch in tqdm(train_loader):
            histories, targets = (
                batch[0].to(device),
                batch[1].to(device),
            )
            loss = model.full_sequence_loss(histories, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        metrics = evaluate(model, test_loader, len(test_dataset), device)
        print("Test\n", metrics)
        stopped_epoch = epoch
        if metrics["NDCG@10"] > best_ndcg:
            best_ndcg = metrics["NDCG@10"]
            best_metrics = metrics
            patience_count = 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            patience_count += 1
            print(f"Patience{patience_count}/{args.patience}")
            if patience_count >= args.patience:
                break

    if best_state is None:
        raise RuntimeError("SASRec did not produce an NDCG@10 value above zero")
    model.load_state_dict(best_state)
    torch.save(best_state, output_dir / "best_model.pth")
    torch.save(model.checkpoint(), paths.directory / "sasrec_best.pt")
    torch.save(model.item_emb.weight.data.cpu(), paths.item_embeddings)
    final_metrics = evaluate(model, test_loader, len(test_dataset), device)
    metadata = {
        "implementation": "original_full_sequence_sasrec",
        "config": vars(args),
        "stopped_epoch": stopped_epoch,
        "best_metrics": best_metrics,
        "final_metrics": final_metrics,
        "elapsed_seconds": time.time() - start_time,
        "checkpoint": str(output_dir / "best_model.pth"),
    }
    with (paths.directory / "sasrec_metrics.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(metadata, handle, indent=2)
    print("The Final Test Metric for Model is following:")
    print(final_metrics)
    print(f"Best checkpoint: {output_dir / 'best_model.pth'}")
    print(f"Item embeddings: {paths.item_embeddings}")
    print("Training Done!")


if __name__ == "__main__":
    main()
