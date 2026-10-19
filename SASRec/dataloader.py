"""Sequence dataset matching the original SASRec truncation behavior."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


class NextItemDataset(Dataset):
    def __init__(self, frame, max_length: int) -> None:
        self.sequences = frame["item_ids"].tolist()
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        item_ids = self.sequences[index].copy()
        if len(item_ids) > self.max_length:
            item_ids = item_ids[-self.max_length :]
        item_ids = torch.tensor(item_ids, dtype=torch.long)
        sequence, target = item_ids[:-1], item_ids[-1]
        sequence = F.pad(
            sequence,
            (self.max_length - sequence.shape[0], 0),
            mode="constant",
            value=0,
        )
        return sequence, target
