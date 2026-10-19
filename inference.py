"""Run multi-GPU inference for BigRec, LLaRA, or GALLM."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("USE_TF", "0")

import torch
import transformers
from accelerate import Accelerator
from accelerate.utils import gather_object
from peft import PeftModel
from transformers import AutoTokenizer, GenerationConfig

from data_utils import (
    DEFAULT_DATA_ROOT,
    dataset_paths,
    load_catalog,
    load_interactions,
)
from recommendation_data import (
    ITEM_TOKEN,
    RESPONSE_HEADER,
    encode_prompt,
    load_examples,
)
from model.gallm import GALLMRecommender, VALID_MODES


class TokenTrie:
    def __init__(self, sequences: list[list[int]]) -> None:
        self.root: dict = {}
        for sequence in sequences:
            node = self.root
            for token in sequence:
                node = node.setdefault(int(token), {})

    def next_tokens(self, prefix: list[int]) -> list[int]:
        node = self.root
        for token in prefix:
            node = node.get(int(token))
            if node is None:
                return []
        return list(node)


def left_pad(records: list[dict], tokenizer, device: torch.device) -> dict[str, torch.Tensor]:
    keys = ("input_ids", "attention_mask", "cf_item_ids", "text_item_ids")
    padding = {
        "input_ids": tokenizer.pad_token_id,
        "attention_mask": 0,
        "cf_item_ids": 0,
        "text_item_ids": 0,
    }
    max_length = max(len(record["input_ids"]) for record in records)
    result = {}
    for key in keys:
        tensor = torch.full(
            (len(records), max_length), padding[key], dtype=torch.long
        )
        for row, record in enumerate(records):
            values = torch.tensor(record[key], dtype=torch.long)
            tensor[row, -len(values) :] = values
        result[key] = tensor.to(device)
    return result


def find_last_subsequence(values: list[int], pattern: list[int]) -> int | None:
    for index in range(len(values) - len(pattern), -1, -1):
        if values[index : index + len(pattern)] == pattern:
            return index + len(pattern)
    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=sorted(VALID_MODES), default="gallm")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT))
    parser.add_argument("--train-sample", type=int)
    parser.add_argument("--test-sample", type=int)
    parser.add_argument("--output")
    parser.add_argument(
        "--batch-size",
        type=int,
        help="Defaults to 8 for GALLM and 4 for BigRec/LLaRA",
    )
    parser.add_argument("--num-beams", type=int, default=10)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--max-test-rows", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--lora-target-modules", default="q_proj,v_proj")
    quantization = parser.add_mutually_exclusive_group()
    quantization.add_argument("--use-8bit", dest="use_8bit", action="store_true")
    quantization.add_argument("--no-8bit", dest="use_8bit", action="store_false")
    parser.set_defaults(use_8bit=False)
    parser.add_argument("--unconstrained", action="store_true")
    parser.add_argument(
        "--standard-cbs",
        action="store_true",
        help="Apply title constraints after log-softmax instead of using custom CBS",
    )
    compilation = parser.add_mutually_exclusive_group()
    compilation.add_argument(
        "--compile-model",
        dest="compile_model",
        action="store_true",
        help="Compile the merged inference model (legacy default)",
    )
    compilation.add_argument(
        "--no-compile",
        dest="compile_model",
        action="store_false",
        help="Disable torch.compile while retaining the static KV cache",
    )
    parser.set_defaults(compile_model=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.batch_size is None:
        args.batch_size = 8 if args.mode == "gallm" else 4
    if args.top_k > args.num_beams:
        raise ValueError("--top-k cannot exceed --num-beams")
    transformers.set_seed(args.seed)
    accelerator = Accelerator()
    checkpoint = Path(args.checkpoint).expanduser().resolve()
    if not (checkpoint / "adapter_config.json").is_file():
        raise FileNotFoundError(f"LoRA adapter not found in {checkpoint}")
    metadata_path = checkpoint / "model_metadata.json"
    if metadata_path.is_file():
        with metadata_path.open("r", encoding="utf-8") as handle:
            saved_mode = json.load(handle).get("mode")
        if saved_mode != args.mode:
            raise ValueError(
                f"Checkpoint mode is {saved_mode}, but --mode is {args.mode}"
            )

    paths = dataset_paths(
        args.dataset,
        args.data_root,
        train_sample=args.train_sample,
        test_sample=args.test_sample,
    )
    required_artifacts = []
    if args.mode in {"llara", "gallm"}:
        required_artifacts.append(paths.item_embeddings)
    if args.mode == "gallm":
        required_artifacts.append(paths.cooccurrence)
    for artifact in required_artifacts:
        if not artifact.is_file():
            raise FileNotFoundError(
                f"Missing artifact required by {args.mode}: {artifact}"
            )

    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    if tokenizer.pad_token != "<pad>":
        tokenizer.add_special_tokens({"pad_token": "<pad>"})
    if (
        args.mode in {"llara", "gallm"}
        and ITEM_TOKEN not in tokenizer.additional_special_tokens
    ):
        tokenizer.add_special_tokens({"additional_special_tokens": [ITEM_TOKEN]})
    tokenizer.padding_side = "left"

    model = GALLMRecommender(
        mode=args.mode,
        base_model=args.base_model,
        tokenizer=tokenizer,
        item_embeddings_path=(
            paths.item_embeddings if args.mode in {"llara", "gallm"} else None
        ),
        cooccurrence_path=paths.cooccurrence if args.mode == "gallm" else None,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        lora_target_modules=[
            name.strip() for name in args.lora_target_modules.split(",") if name.strip()
        ],
        use_8bit=args.use_8bit,
        prepare_for_training=False,
        attach_lora=False,
        device_map=(
            {"": accelerator.local_process_index}
            if torch.cuda.is_available()
            else None
        ),
        use_custom_cbs=not args.standard_cbs and not args.unconstrained,
    )
    model.llama_model = PeftModel.from_pretrained(
        model.llama_model,
        checkpoint,
        torch_dtype=torch.bfloat16,
    )
    model.load_components(checkpoint)
    model.llama_model = model.llama_model.merge_and_unload()
    model.refresh_attention_hooks()

    device = accelerator.device
    if not args.use_8bit:
        model.to(device)
    else:
        device = next(model.llama_model.parameters()).device
    model.llama_model.config.use_cache = True
    model.llama_model.generation_config.cache_implementation = "static"
    model.llama_model.generation_config.disable_compile = True
    if args.compile_model:
        model.llama_model.compile(
            mode="reduce-overhead",
            fullgraph=True,
        )
    model.eval()

    catalog = load_catalog(paths.catalog)
    test_frame = load_interactions(paths.test)
    if args.max_test_rows:
        test_frame = test_frame.head(args.max_test_rows)
    examples = load_examples(test_frame, catalog, args.dataset)
    examples = [example | {"_index": index} for index, example in enumerate(examples)]

    title_sequences = [
        tokenizer.encode(f'"{title}"', add_special_tokens=False)
        + [tokenizer.eos_token_id]
        for title in catalog.values()
    ]
    trie = TokenTrie(title_sequences)
    response_header = tokenizer.encode(RESPONSE_HEADER, add_special_tokens=False)

    def allowed_tokens(batch_id: int, input_ids: torch.Tensor) -> list[int]:
        values = input_ids.tolist()
        start = find_last_subsequence(values, response_header)
        prefix = values[start:] if start is not None else []
        return trie.next_tokens(prefix) or [tokenizer.eos_token_id]

    local_predictions: list[dict] = []
    with accelerator.split_between_processes(examples) as local_examples:
        for start in range(0, len(local_examples), args.batch_size):
            batch_examples = local_examples[start : start + args.batch_size]
            encoded = [
                encode_prompt(
                    example,
                    tokenizer,
                    include_target=False,
                    include_item_embeddings=args.mode in {"llara", "gallm"},
                )
                for example in batch_examples
            ]
            tensors = left_pad(encoded, tokenizer, device)
            generation_kwargs = {
                "input_ids": tensors["input_ids"],
                "attention_mask": tensors["attention_mask"],
            }
            if model.uses_item_embeddings:
                generation_kwargs["cf_item_ids"] = tensors["cf_item_ids"]
                generation_kwargs["rec_embedding"] = model.prepare_item_embeddings(
                    tensors["cf_item_ids"]
                )
            if model.uses_graph_bias:
                model.attention_bias_hooks.biases = model.build_attention_biases(
                    tensors["cf_item_ids"],
                    tensors["text_item_ids"],
                    tensors["attention_mask"],
                )
            with torch.inference_mode():
                try:
                    generated = model.llama_model.generate(
                        **generation_kwargs,
                        generation_config=GenerationConfig(
                            eos_token_id=tokenizer.eos_token_id,
                            pad_token_id=tokenizer.pad_token_id,
                            num_beams=args.num_beams,
                            num_return_sequences=args.top_k,
                            max_new_tokens=args.max_new_tokens,
                            cache_implementation="static",
                            disable_compile=True,
                            return_dict_in_generate=True,
                            output_scores=True,
                        ),
                        prefix_allowed_tokens_fn=None
                        if args.unconstrained
                        else allowed_tokens,
                    )
                finally:
                    model.attention_bias_hooks.biases = None

            prompt_length = tensors["input_ids"].shape[1]
            decoded = tokenizer.batch_decode(
                generated.sequences[:, prompt_length:], skip_special_tokens=True
            )
            score_values = (
                generated.sequences_scores.detach().float().cpu().tolist()
                if generated.sequences_scores is not None
                else [0.0] * len(decoded)
            )
            for row, example in enumerate(batch_examples):
                low = row * args.top_k
                high = low + args.top_k
                local_predictions.append(
                    {
                        "_index": example["_index"],
                        "user_id": example["user_id"],
                        "history_item_ids": example["history_ids"],
                        "target_item_id": example["target_id"],
                        "output": f'"{example["target_title"]}"',
                        "predict": decoded[low:high],
                        "scores": score_values[low:high],
                    }
                )

    predictions = gather_object(local_predictions)
    accelerator.wait_for_everyone()
    decoding_name = (
        "beam"
        if args.unconstrained
        else "standard_cbs"
        if args.standard_cbs
        else "custom_cbs"
    )
    output = (
        Path(args.output).expanduser().resolve()
        if args.output
        else checkpoint
        / f"predict_{args.dataset}_{len(examples)}_{args.mode}_{decoding_name}.json"
    )
    if accelerator.is_main_process:
        predictions.sort(key=lambda record: record["_index"])
        for record in predictions:
            record.pop("_index", None)
        with output.open("w", encoding="utf-8") as handle:
            json.dump(predictions, handle, indent=2, ensure_ascii=False)
        print(f"Saved {len(predictions):,} predictions to {output}")


if __name__ == "__main__":
    main()
