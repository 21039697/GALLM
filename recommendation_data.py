"""Prompt construction shared by all recommendation modes."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.utils.data import Dataset

from data_utils import iter_history_target
from utils import get_prompt


PROMPT_TEMPLATE = (
    "Below is an instruction that describes a task, paired with an input that "
    "provides further context. Write a response that appropriately completes "
    "the request. \n\n### Instruction:\n{instruction}\n\n### Input:\n{history_prefix}"
)
RESPONSE_HEADER = "\n\n### Response:\n"
ITEM_TOKEN = "[HistoryEmb]"


def load_examples(frame, catalog: dict[int, str], dataset: str) -> list[dict]:
    instruction, history_prefix = get_prompt(dataset)
    examples = []
    for user_id, history_ids, target_id in iter_history_target(frame, catalog):
        examples.append(
            {
                "user_id": user_id,
                "instruction": instruction,
                "history_prefix": history_prefix,
                "history_ids": history_ids,
                "history_titles": [catalog[item_id] for item_id in history_ids],
                "target_id": target_id,
                "target_title": catalog[target_id],
            }
        )
    return examples


def encode_prompt(
    example: dict,
    tokenizer,
    include_target: bool,
    include_item_embeddings: bool = True,
) -> dict[str, list[int]]:
    prefix = PROMPT_TEMPLATE.format(
        instruction=example["instruction"],
        history_prefix=example["history_prefix"],
    )
    encoded = tokenizer(prefix, add_special_tokens=not include_item_embeddings)
    input_ids = list(encoded["input_ids"])
    attention_mask = list(encoded["attention_mask"])
    cf_item_ids = [0] * len(input_ids)
    text_item_ids = [0] * len(input_ids)

    history = list(zip(example["history_titles"], example["history_ids"]))
    for history_index, (title, item_id) in enumerate(history):
        title_ids = tokenizer.encode(f'"{title}"', add_special_tokens=False)
        input_ids.extend(title_ids)
        attention_mask.extend([1] * len(title_ids))
        cf_item_ids.extend([0] * len(title_ids))
        text_item_ids.extend([item_id] * len(title_ids))

        if include_item_embeddings:
            marker_ids = tokenizer.encode(ITEM_TOKEN, add_special_tokens=False)
            input_ids.extend(marker_ids)
            attention_mask.extend([1] * len(marker_ids))
            cf_item_ids.extend([item_id] * len(marker_ids))
            text_item_ids.extend([item_id] * len(marker_ids))

        if include_item_embeddings or history_index + 1 < len(history):
            separator_ids = tokenizer.encode(", ", add_special_tokens=False)
            input_ids.extend(separator_ids)
            attention_mask.extend([1] * len(separator_ids))
            cf_item_ids.extend([0] * len(separator_ids))
            text_item_ids.extend([0] * len(separator_ids))

    response_header = RESPONSE_HEADER if include_item_embeddings else "\n " + RESPONSE_HEADER
    response_header_ids = tokenizer.encode(response_header, add_special_tokens=False)
    input_ids.extend(response_header_ids)
    attention_mask.extend([1] * len(response_header_ids))
    cf_item_ids.extend([0] * len(response_header_ids))
    text_item_ids.extend([0] * len(response_header_ids))
    prompt_length = len(input_ids)

    labels = [-100] * prompt_length
    if include_target:
        target_ids = tokenizer.encode(
            f'"{example["target_title"]}"', add_special_tokens=False
        ) + [tokenizer.eos_token_id]
        input_ids.extend(target_ids)
        attention_mask.extend([1] * len(target_ids))
        cf_item_ids.extend([0] * len(target_ids))
        text_item_ids.extend([0] * len(target_ids))
        labels.extend(target_ids)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "cf_item_ids": cf_item_ids,
        "text_item_ids": text_item_ids,
        "labels": labels,
    }


class RecommendationTrainingDataset(Dataset):
    def __init__(
        self,
        examples: list[dict],
        tokenizer,
        max_length: int | None,
        include_item_embeddings: bool = True,
    ) -> None:
        self.records = []
        for example in examples:
            encoded = encode_prompt(
                example,
                tokenizer,
                include_target=True,
                include_item_embeddings=include_item_embeddings,
            )
            if max_length is None or len(encoded["input_ids"]) <= max_length:
                self.records.append(encoded)
        if not self.records:
            raise ValueError(
                "No training examples fit --max-length; increase it or shorten histories"
            )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict:
        return self.records[index]


@dataclass
class LeftPadCollator:
    tokenizer: object
    pad_to_multiple_of: int | None = 8

    def _pad(self, values: list[list[int]], padding_value: int) -> torch.Tensor:
        max_length = max(map(len, values))
        if self.pad_to_multiple_of:
            multiple = self.pad_to_multiple_of
            max_length = ((max_length + multiple - 1) // multiple) * multiple
        output = torch.full(
            (len(values), max_length), padding_value, dtype=torch.long
        )
        for row, sequence in enumerate(values):
            output[row, -len(sequence) :] = torch.tensor(sequence, dtype=torch.long)
        return output

    def __call__(self, features: list[dict]) -> dict[str, torch.Tensor]:
        return {
            "input_ids": self._pad(
                [feature["input_ids"] for feature in features],
                self.tokenizer.pad_token_id,
            ),
            "attention_mask": self._pad(
                [feature["attention_mask"] for feature in features], 0
            ),
            "cf_item_ids": self._pad(
                [feature["cf_item_ids"] for feature in features], 0
            ),
            "text_item_ids": self._pad(
                [feature["text_item_ids"] for feature in features], 0
            ),
            "labels": self._pad([feature["labels"] for feature in features], -100),
        }
