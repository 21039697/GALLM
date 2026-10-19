"""Train BigRec, LLaRA, or GALLM on one dataset under data/."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path

os.environ.setdefault("USE_TF", "0")

import torch
import transformers
from transformers import AutoTokenizer, Trainer

from data_utils import (
    DEFAULT_DATA_ROOT,
    dataset_paths,
    load_catalog,
    load_interactions,
    validate_dataset,
)
from recommendation_data import (
    ITEM_TOKEN,
    RecommendationTrainingDataset,
    LeftPadCollator,
    load_examples,
)
from model.gallm import GALLMRecommender, VALID_MODES


class ConstantLearningRateScheduler:
    """Keep BigRec's legacy learning rate unchanged for every optimizer step."""

    def __init__(self, optimizer: torch.optim.Optimizer) -> None:
        self.optimizer = optimizer

    def step(self) -> None:
        pass

    def state_dict(self) -> dict:
        return {}

    def load_state_dict(self, state_dict: dict) -> None:
        pass

    def get_last_lr(self) -> list[float]:
        return [group["lr"] for group in self.optimizer.param_groups]


class RecommendationTrainer(Trainer):
    def create_scheduler(self, num_training_steps: int, optimizer=None):
        if self.model.mode != "bigrec":
            return super().create_scheduler(num_training_steps, optimizer)
        if optimizer is None:
            optimizer = self.optimizer
        self.lr_scheduler = ConstantLearningRateScheduler(optimizer)
        return self.lr_scheduler

    def _save(self, output_dir: str | None = None, state_dict=None) -> None:
        output = Path(output_dir or self.args.output_dir)
        output.mkdir(parents=True, exist_ok=True)
        self.model.llama_model.save_pretrained(
            output, safe_serialization=True, save_embedding_layers=False
        )
        torch.save(self.model.component_state_dict(), output / "model_components.pt")
        with (output / "model_metadata.json").open("w", encoding="utf-8") as handle:
            json.dump({"mode": self.model.mode}, handle, indent=2)
        processing_class = getattr(self, "processing_class", None)
        if processing_class is not None:
            processing_class.save_pretrained(output)
        torch.save(self.args, output / "training_args.bin")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=sorted(VALID_MODES), default="gallm")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT))
    parser.add_argument("--train-sample", type=int)
    parser.add_argument("--test-sample", type=int)
    parser.add_argument("--output-dir")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Global effective batch size across all GPUs and accumulation steps",
    )
    parser.add_argument(
        "--micro-batch-size",
        type=int,
        default=8,
        help="Per-GPU batch size for one forward/backward pass",
    )
    parser.add_argument("--epochs", type=float, default=5)
    parser.add_argument("--lora-learning-rate", type=float, default=1e-4)
    parser.add_argument(
        "--item-learning-rate",
        type=float,
        help="Defaults to --lora-learning-rate",
    )
    parser.add_argument(
        "--bias-learning-rate",
        type=float,
        help="Defaults to 50 times --lora-learning-rate",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        help="Defaults to 0.01 for GALLM and 0 for BigRec/LLaRA",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        help="Optional maximum token length; the legacy default keeps every row",
    )
    parser.add_argument("--max-train-rows", type=int)
    parser.add_argument("--warmup-steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--lora-target-modules", default="q_proj,v_proj")
    quantization = parser.add_mutually_exclusive_group()
    quantization.add_argument("--use-8bit", dest="use_8bit", action="store_true")
    quantization.add_argument("--no-8bit", dest="use_8bit", action="store_false")
    parser.set_defaults(use_8bit=True)
    checkpointing = parser.add_mutually_exclusive_group()
    checkpointing.add_argument(
        "--gradient-checkpointing",
        dest="gradient_checkpointing",
        action="store_true",
    )
    checkpointing.add_argument(
        "--no-gradient-checkpointing",
        dest="gradient_checkpointing",
        action="store_false",
    )
    parser.set_defaults(gradient_checkpointing=None)
    parser.add_argument("--report-to", default="none")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    transformers.set_seed(args.seed)
    paths = dataset_paths(
        args.dataset,
        args.data_root,
        train_sample=args.train_sample,
        test_sample=args.test_sample,
    )
    stats = validate_dataset(paths)
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

    world_size = max(1, int(os.environ.get("WORLD_SIZE", "1")))
    if world_size > 1 and not args.output_dir:
        raise ValueError("Multi-GPU training requires an explicit --output-dir")
    run_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else Path("outputs").resolve()
        / args.dataset
        / args.mode
        / datetime.now().strftime("%Y%m%d-%H%M%S")
    )
    run_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token != "<pad>":
        tokenizer.add_special_tokens({"pad_token": "<pad>"})
    if args.mode in {"llara", "gallm"}:
        tokenizer.add_special_tokens({"additional_special_tokens": [ITEM_TOKEN]})
    tokenizer.padding_side = "left"

    catalog = load_catalog(paths.catalog)
    train_frame = load_interactions(paths.train)
    if args.max_train_rows:
        train_frame = train_frame.head(args.max_train_rows)
    examples = load_examples(train_frame, catalog, args.dataset)
    train_dataset = RecommendationTrainingDataset(
        examples,
        tokenizer,
        args.max_length,
        include_item_embeddings=args.mode in {"llara", "gallm"},
    )
    dropped = len(examples) - len(train_dataset)

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
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
        prepare_for_training=True,
        device_map={"": local_rank} if torch.cuda.is_available() else None,
    )
    if args.use_8bit:
        model.move_auxiliary_to(torch.device(f"cuda:{local_rank}"))
    model.llama_model.print_trainable_parameters()

    distributed_micro_batch = args.micro_batch_size * world_size
    if args.batch_size < distributed_micro_batch:
        raise ValueError(
            "--batch-size must be at least --micro-batch-size multiplied by WORLD_SIZE"
        )
    if args.batch_size % distributed_micro_batch != 0:
        raise ValueError(
            "--batch-size must be divisible by --micro-batch-size multiplied by WORLD_SIZE"
        )
    accumulation = args.batch_size // distributed_micro_batch
    item_learning_rate = (
        args.item_learning_rate
        if args.item_learning_rate is not None
        else args.lora_learning_rate
    )
    bias_learning_rate = (
        args.bias_learning_rate
        if args.bias_learning_rate is not None
        else args.lora_learning_rate * 50.0
    )
    weight_decay = (
        args.weight_decay
        if args.weight_decay is not None
        else (0.01 if args.mode == "gallm" else 0.0)
    )
    gradient_checkpointing = (
        args.gradient_checkpointing
        if args.gradient_checkpointing is not None
        else args.mode == "gallm"
    )
    optimizer_groups = [
        {
            "params": [
                parameter
                for parameter in model.llama_model.parameters()
                if parameter.requires_grad
            ],
            "lr": args.lora_learning_rate,
            "name": "lora",
        }
    ]
    if model.uses_item_embeddings:
        optimizer_groups.append(
            {
                "params": list(model.item_embeddings.parameters())
                + list(model.item_projector.parameters()),
                "lr": item_learning_rate,
                "name": "item",
            }
        )
    if model.uses_graph_bias:
        optimizer_groups.append(
            {
                "params": list(model.text_text_bias.parameters())
                + list(model.item_text_bias.parameters())
                + list(model.item_item_bias.parameters()),
                "lr": bias_learning_rate,
                "name": "graph_bias",
            }
        )
    optimizer = torch.optim.AdamW(
        optimizer_groups,
        weight_decay=weight_decay,
    )
    if int(os.environ.get("RANK", "0")) == 0:
        group_summary = ", ".join(
            f"{group['name']}={group['lr']:g}" for group in optimizer_groups
        )
        print(
            f"Global batch={args.batch_size}, per-GPU micro batch="
            f"{args.micro_batch_size}, GPUs={world_size}, "
            f"gradient accumulation={accumulation}"
        )
        scheduler_name = "constant" if args.mode == "bigrec" else "linear"
        print(f"Optimizer learning rates: {group_summary}")
        print(
            f"Scheduler={scheduler_name}, weight decay={weight_decay:g}, "
            f"gradient checkpointing={gradient_checkpointing}"
        )

    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    use_tf32 = (
        torch.cuda.is_available()
        and torch.cuda.get_device_capability()[0] >= 8
    )
    training_args = transformers.TrainingArguments(
        output_dir=str(run_dir),
        per_device_train_batch_size=args.micro_batch_size,
        gradient_accumulation_steps=accumulation,
        num_train_epochs=args.epochs,
        learning_rate=args.lora_learning_rate,
        weight_decay=weight_decay,
        warmup_steps=args.warmup_steps,
        bf16=use_bf16,
        fp16=torch.cuda.is_available() and not use_bf16,
        tf32=use_tf32,
        gradient_checkpointing=gradient_checkpointing,
        gradient_checkpointing_kwargs=(
            {"use_reentrant": False} if gradient_checkpointing else None
        ),
        ddp_find_unused_parameters=(
            True
            if args.mode == "gallm"
            else (False if world_size > 1 else None)
        ),
        logging_strategy="steps",
        logging_steps=0.1,
        save_strategy="steps",
        save_steps=1.0 / args.epochs,
        save_on_each_node=False,
        log_on_each_node=False,
        remove_unused_columns=False,
        report_to=args.report_to,
        seed=args.seed,
        data_seed=args.seed,
    )
    metadata = vars(args) | {
        "dataset_files": {
            "train": str(paths.train),
            "test": str(paths.test),
            "catalog": str(paths.catalog),
            "item_embeddings": str(paths.item_embeddings),
            "cooccurrence": str(paths.cooccurrence),
        },
        "dataset_stats": stats,
        "training_examples": len(train_dataset),
        "dropped_over_max_length": dropped,
        "effective_learning_rates": {
            group["name"]: group["lr"] for group in optimizer_groups
        },
        "effective_weight_decay": weight_decay,
        "learning_rate_scheduler": (
            "constant" if args.mode == "bigrec" else "linear"
        ),
        "effective_gradient_checkpointing": gradient_checkpointing,
    }
    if int(os.environ.get("RANK", "0")) == 0:
        with (run_dir / "run_config.json").open("w", encoding="utf-8") as handle:
            json.dump(metadata, handle, indent=2)

    trainer = RecommendationTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=LeftPadCollator(tokenizer),
        processing_class=tokenizer,
        optimizers=(optimizer, None),
    )
    trainer.train()
    final_dir = run_dir / "final"
    trainer.save_model(str(final_dir))
    print(f"Training complete. Final checkpoint: {final_dir}")


if __name__ == "__main__":
    main()
