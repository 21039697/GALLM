# Advanced usage

Start with the [README](../README.md) for installation and the main workflow, and [data preparation](data.md) for artifact regeneration.

## Model and optimizer defaults

| Setting | BigRec | LLaRA | GALLM |
| --- | --- | --- | --- |
| LoRA learning rate | `1e-4` | `1e-4` | `1e-4` |
| Item embeddings/projector learning rate | — | LoRA rate | LoRA rate |
| Graph bias learning rate | — | — | LoRA rate × 50 |
| Weight decay | `0` | `0` | `0.01` |
| Schedule | Constant | Linear | Linear |
| Gradient checkpointing | Off | Off | On |

Training defaults: five epochs, global batch size 64, per-GPU micro batch size 8, 20 warmup steps for the linear schedule, seed 42, and LoRA rank 8 / alpha 16 / dropout 0.05 on `q_proj,v_proj`. Training loads the base model in 8-bit by default; add `--no-8bit` to disable this. Inference defaults to non-quantized loading; `--use-8bit` opts in when bitsandbytes is installed.

Use `--lora-learning-rate`, `--item-learning-rate`, `--bias-learning-rate`, and `--weight-decay` to override optimizer settings. Derived rates follow the LoRA rate unless explicitly overridden. Use `--gradient-checkpointing` or `--no-gradient-checkpointing` to override checkpointing.

LLaRA and GALLM insert `[HistoryEmb]` after every history title and replace its token embedding with a projected SASRec embedding. GALLM adds per-layer Item–Item, Item–Text, and Text–Text attention biases. Item–Item counts use fixed buckets: 0–2, 3–5, 6–12, 13–30, and greater than 30. The paper describes equal-frequency binning; this implementation uses these fixed boundaries.

## Checkpoints

If `--output-dir` is omitted, training creates `outputs/<dataset>/<mode>/<timestamp>/`. Checkpoints and `final/` contain the LoRA adapter, tokenizer, `model_components.pt`, and `model_metadata.json`. Inference requires both `--base-model` and `--checkpoint`, with a matching mode.

## Multiple GPUs

Use `CUDA_VISIBLE_DEVICES` to select GPU IDs, and set the process count to the number of selected GPUs. For example, `CUDA_VISIBLE_DEVICES=0,2` uses GPUs 0 and 2, which appear as local devices 0 and 1 inside the processes.

Global batch size must be divisible by `micro_batch_size × GPU_count`. With two GPUs and the defaults, gradient accumulation is 4. Supply a shared output directory:

```bash
CUDA_VISIBLE_DEVICES=0,2 torchrun --standalone --nproc_per_node=2 train.py \
  --dataset Clothing --mode gallm \
  --base-model meta-llama/Llama-3.2-3B \
  --output-dir outputs/Clothing/gallm/run-01 \
  --batch-size 64 --micro-batch-size 8 --epochs 5

CUDA_VISIBLE_DEVICES=0,2 accelerate launch --multi_gpu --num_processes 2 inference.py \
  --dataset Clothing --mode gallm \
  --base-model meta-llama/Llama-3.2-3B \
  --checkpoint outputs/Clothing/gallm/run-01/final \
  --output outputs/Clothing/gallm/run-01/predictions.json
```

Each inference process handles a disjoint test subset. The main process gathers predictions and restores CSV order before writing JSON.

## Decoding

Inference defaults to 10 beams and 10 recommendations. `--top-k` cannot exceed `--num-beams`. Per-process batch size defaults to 8 for GALLM and 4 for BigRec/LLaRA; override with `--batch-size`.

The default constrained beam search applies catalog-title constraints before log-softmax, using `model/constrained_decoding.py` for BigRec and `model/graph_decoding.py` for LLaRA/GALLM. `--standard-cbs` applies constraints after log-softmax; `--unconstrained` disables title constraints.

Inference merges the LoRA adapter, uses a static KV cache, and compiles the forward path. Add `--no-compile` when TorchInductor is unavailable. This retains the static cache.

Without `--output`, predictions are written to `<checkpoint>/predict_<dataset>_<actual_test_rows>_<mode>_<decoding>.json`. Each record contains an `output` target title and a ranked `predict` list. Evaluation writes a neighboring `.metrics.json` file by default.

## Quick checks

```bash
python scripts/check_data.py
python -m unittest discover -s tests -v
```

The unit tests use a tiny randomly initialized Llama on CPU, requiring no model download. They check graph gradient flow, embedding injection, constrained decoding, and the evaluation metric calculation. Full paper-scale training requires the chosen pretrained checkpoint and GPU resources.
