"""BigRec, LLaRA, and GALLM recommendation model variants."""

from __future__ import annotations

import ast
import json
import os
from itertools import combinations
from pathlib import Path
from typing import Any

os.environ.setdefault("USE_TF", "0")

import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import BitsAndBytesConfig, LlamaForCausalLM


class MLPProjector(nn.Module):
    def __init__(self, input_size: int, output_size: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, output_size),
            nn.GELU(),
            nn.Linear(output_size, output_size),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.layers(inputs)


class GraphLlamaForCausalLM(LlamaForCausalLM):
    """Llama that replaces history marker embeddings during the prefill pass."""

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        past_key_values=None,
        inputs_embeds=None,
        labels=None,
        cf_item_ids=None,
        rec_embedding=None,
        **kwargs,
    ):
        if inputs_embeds is None and input_ids is not None:
            inputs_embeds = self.model.embed_tokens(input_ids)
            if (
                cf_item_ids is not None
                and rec_embedding is not None
                and cf_item_ids.shape[:2] == inputs_embeds.shape[:2]
            ):
                marker_mask = cf_item_ids.ne(0).unsqueeze(-1)
                inputs_embeds = torch.where(
                    marker_mask,
                    rec_embedding.to(device=inputs_embeds.device, dtype=inputs_embeds.dtype),
                    inputs_embeds,
                )
        return super().forward(
            input_ids=None if inputs_embeds is not None else input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            **kwargs,
        )


class AttentionBiasHooks:
    """Inject a per-layer additive bias into Llama attention masks."""

    def __init__(self) -> None:
        self.biases: list[torch.Tensor | None] | None = None
        self.handles: list[Any] = []

    def register(self, layers: nn.ModuleList) -> None:
        self.remove()
        for layer_index, layer in enumerate(layers):
            handle = layer.self_attn.register_forward_pre_hook(
                self._hook(layer_index), with_kwargs=True
            )
            self.handles.append(handle)

    def remove(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def _hook(self, layer_index: int):
        def add_bias(module, args, kwargs):
            if self.biases is None or layer_index >= len(self.biases):
                return args, kwargs
            layer_bias = self.biases[layer_index]
            attention_mask = kwargs.get("attention_mask")
            if layer_bias is None or attention_mask is None:
                return args, kwargs

            if attention_mask.shape[0] != layer_bias.shape[0]:
                if attention_mask.shape[0] % layer_bias.shape[0] != 0:
                    return args, kwargs
                repeats = attention_mask.shape[0] // layer_bias.shape[0]
                layer_bias = layer_bias.repeat_interleave(repeats, dim=0)

            query_length, key_length = attention_mask.shape[-2:]
            bias_length = layer_bias.shape[-1]
            if query_length != bias_length or key_length < bias_length:
                return args, kwargs
            if key_length > bias_length:
                layer_bias = F.pad(layer_bias, (key_length - bias_length, 0, 0, 0))
            kwargs["attention_mask"] = attention_mask + layer_bias.to(
                device=attention_mask.device, dtype=attention_mask.dtype
            )
            return args, kwargs

        return add_bias


def load_cooccurrence(path: str | Path) -> dict[tuple[int, int], float]:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    pairs: dict[tuple[int, int], float] = {}
    if isinstance(payload, dict) and payload.get("format") == "coo":
        records = payload.get("pairs", [])
        for left, right, count in records:
            left, right = sorted((int(left), int(right)))
            pairs[(left, right)] = float(count)
        return pairs

    if isinstance(payload, dict):
        for key, count in payload.items():
            parsed = ast.literal_eval(key)
            if not isinstance(parsed, tuple) or len(parsed) != 2:
                raise ValueError(f"Invalid legacy co-occurrence key: {key}")
            left, right = sorted((int(parsed[0]), int(parsed[1])))
            pairs[(left, right)] = float(count)
        return pairs
    raise ValueError(f"Unsupported co-occurrence format in {path}")


VALID_MODES = {"bigrec", "llara", "gallm"}


class GALLMRecommender(nn.Module):
    """A shared wrapper for the BigRec, LLaRA, and GALLM modes."""

    accepts_loss_kwargs = False

    def __init__(
        self,
        *,
        mode: str,
        base_model: str,
        tokenizer,
        item_embeddings_path: str | Path | None = None,
        cooccurrence_path: str | Path | None = None,
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
        lora_target_modules: list[str] | None = None,
        use_8bit: bool = False,
        prepare_for_training: bool = True,
        attach_lora: bool = True,
        device_map: str | dict | None = None,
        use_custom_cbs: bool = False,
    ) -> None:
        super().__init__()
        mode = mode.lower()
        if mode not in VALID_MODES:
            raise ValueError(f"mode must be one of {sorted(VALID_MODES)}, got {mode}")
        self.mode = mode
        self.uses_item_embeddings = mode in {"llara", "gallm"}
        self.uses_graph_bias = mode == "gallm"
        lora_target_modules = lora_target_modules or ["q_proj", "v_proj"]
        quantization = (
            BitsAndBytesConfig(load_in_8bit=True) if use_8bit else None
        )
        load_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        model_class = GraphLlamaForCausalLM
        if use_custom_cbs:
            if self.uses_item_embeddings:
                from model.graph_decoding import (
                    CustomLlamaForCausalLM as CustomCBSModel,
                )
            else:
                from model.constrained_decoding import (
                    CustomLlamaForCausalLM as CustomCBSModel,
                )
            model_class = CustomCBSModel
        # SDPA may elide the causal mask for unpadded batches. Graph hooks
        # require an explicit mask so relation biases are never skipped.
        attention_kwargs = {"attn_implementation": "eager"} if self.uses_graph_bias else {}
        self.llama_model = model_class.from_pretrained(
            base_model,
            quantization_config=quantization,
            torch_dtype=load_dtype,
            device_map=device_map,
            **attention_kwargs,
        )
        self.llama_model.resize_token_embeddings(len(tokenizer))
        self.llama_model.config.pad_token_id = tokenizer.pad_token_id
        self.llama_model.config.use_cache = False

        if prepare_for_training:
            preparation_kwargs = {}
            if mode == "gallm":
                preparation_kwargs = {
                    "use_gradient_checkpointing": True,
                    "gradient_checkpointing_kwargs": {"use_reentrant": False},
                }
            self.llama_model = prepare_model_for_kbit_training(
                self.llama_model,
                **preparation_kwargs,
            )
        if attach_lora:
            self.llama_model = get_peft_model(
                self.llama_model,
                LoraConfig(
                    r=lora_r,
                    lora_alpha=lora_alpha,
                    target_modules=lora_target_modules,
                    lora_dropout=lora_dropout,
                    bias="none",
                    task_type="CAUSAL_LM",
                ),
            )
        self.quantization_method = getattr(
            self.llama_model, "quantization_method", None
        )
        self.hf_device_map = getattr(self.llama_model, "hf_device_map", None)

        decoder = self._decoder()
        hidden_size = decoder.config.hidden_size
        self.num_layers = decoder.config.num_hidden_layers
        self.num_buckets = 5

        self.attention_bias_hooks = AttentionBiasHooks()
        self.cooccurrence: dict[tuple[int, int], float] = {}

        if self.uses_item_embeddings:
            if item_embeddings_path is None:
                raise ValueError(f"{mode} mode requires item_embeddings_path")
            item_weights = torch.load(
                Path(item_embeddings_path), map_location="cpu", weights_only=True
            )
            if not isinstance(item_weights, torch.Tensor) or item_weights.ndim != 2:
                raise ValueError(
                    "item_embeddings.pth must contain one two-dimensional tensor"
                )
            item_weights = item_weights.to(dtype=load_dtype)
            self.item_embeddings = nn.Embedding.from_pretrained(
                item_weights, freeze=False, padding_idx=0
            )
            self.item_projector = MLPProjector(
                item_weights.shape[1], hidden_size
            ).to(dtype=load_dtype)

        if self.uses_graph_bias:
            if cooccurrence_path is None:
                raise ValueError("gallm mode requires cooccurrence_path")
            self.text_text_bias = nn.ParameterList(
                [nn.Parameter(torch.tensor(0.5)) for _ in range(self.num_layers)]
            )
            self.item_text_bias = nn.ParameterList(
                [nn.Parameter(torch.tensor(0.5)) for _ in range(self.num_layers)]
            )
            self.item_item_bias = nn.ParameterList(
                [
                    nn.Parameter(torch.full((self.num_buckets,), 0.5))
                    for _ in range(self.num_layers)
                ]
            )
            self.cooccurrence = load_cooccurrence(cooccurrence_path)
            self.attention_bias_hooks.register(decoder.layers)

    def _decoder(self):
        current = self.llama_model
        visited: set[int] = set()
        while id(current) not in visited:
            visited.add(id(current))
            if hasattr(current, "layers") and hasattr(current, "config"):
                return current
            if hasattr(current, "base_model"):
                current = current.base_model
            elif hasattr(current, "model"):
                current = current.model
            else:
                break
        raise TypeError("The selected base model is not a supported Llama architecture")

    def refresh_attention_hooks(self) -> None:
        if self.uses_graph_bias:
            self.attention_bias_hooks.register(self._decoder().layers)

    def move_auxiliary_to(self, device: torch.device) -> None:
        """Move non-backbone trainable components for quantized DDP training."""
        if self.uses_item_embeddings:
            self.item_embeddings.to(device)
            self.item_projector.to(device)
        if self.uses_graph_bias:
            self.text_text_bias.to(device)
            self.item_text_bias.to(device)
            self.item_item_bias.to(device)

    def prepare_item_embeddings(self, item_ids: torch.Tensor) -> torch.Tensor:
        if not self.uses_item_embeddings:
            raise RuntimeError("BigRec mode does not use item embeddings")
        if self.item_embeddings.weight.device != item_ids.device:
            self.item_embeddings.to(item_ids.device)
            self.item_projector.to(item_ids.device)
        return self.item_projector(self.item_embeddings(item_ids))

    def build_attention_biases(
        self,
        cf_item_ids: torch.Tensor,
        text_item_ids: torch.Tensor,
        attention_mask: torch.Tensor | None,
    ) -> list[torch.Tensor]:
        if not self.uses_graph_bias:
            raise RuntimeError("Only GALLM mode builds graph attention biases")
        if cf_item_ids.shape != text_item_ids.shape:
            raise ValueError("cf_item_ids and text_item_ids must have identical shapes")
        device = cf_item_ids.device
        valid = (
            attention_mask.to(device=device, dtype=torch.bool)
            if attention_mask is not None
            else torch.ones_like(cf_item_ids, dtype=torch.bool)
        )
        item_tokens = cf_item_ids.ne(0) & valid
        title_tokens = text_item_ids.ne(0) & cf_item_ids.eq(0) & valid
        text_tokens = cf_item_ids.eq(0) & valid

        item_query, item_key = item_tokens.unsqueeze(2), item_tokens.unsqueeze(1)
        title_query, title_key = title_tokens.unsqueeze(2), title_tokens.unsqueeze(1)
        text_text_mask = text_tokens.unsqueeze(2) & text_tokens.unsqueeze(1)

        position_items = torch.where(item_tokens, cf_item_ids, text_item_ids)
        same_item = position_items.unsqueeze(2).eq(position_items.unsqueeze(1))
        item_text_mask = (
            (item_query & title_key) | (title_query & item_key)
        ) & same_item

        counts = torch.zeros(
            (*cf_item_ids.shape, cf_item_ids.shape[1]),
            dtype=torch.float32,
            device=device,
        )
        for batch_index in range(cf_item_ids.shape[0]):
            positions: dict[int, list[int]] = {}
            for position in torch.nonzero(item_tokens[batch_index], as_tuple=True)[0].tolist():
                item_id = int(cf_item_ids[batch_index, position])
                positions.setdefault(item_id, []).append(position)
            for left, right in combinations(sorted(positions), 2):
                count = self.cooccurrence.get((left, right))
                if count is None:
                    continue
                for left_position in positions[left]:
                    for right_position in positions[right]:
                        counts[batch_index, left_position, right_position] = count
                        counts[batch_index, right_position, left_position] = count

        buckets = (
            counts.ge(3).long()
            + counts.ge(6).long()
            + counts.ge(13).long()
            + counts.gt(30).long()
        )
        item_item_mask = (
            item_query
            & item_key
            & cf_item_ids.unsqueeze(2).ne(cf_item_ids.unsqueeze(1))
        )

        layer_biases = []
        for layer_index in range(self.num_layers):
            bias = (
                text_text_mask.float() * self.text_text_bias[layer_index].to(device)
                + item_text_mask.float() * self.item_text_bias[layer_index].to(device)
                + item_item_mask.float()
                * self.item_item_bias[layer_index].to(device)[buckets]
            )
            layer_biases.append(bias.unsqueeze(1))
        return layer_biases

    def forward(
        self,
        input_ids,
        attention_mask,
        labels=None,
        cf_item_ids=None,
        text_item_ids=None,
        **kwargs,
    ):
        kwargs.pop("num_items_in_batch", None)
        rec_embedding = (
            self.prepare_item_embeddings(cf_item_ids)
            if self.uses_item_embeddings
            else None
        )
        self.attention_bias_hooks.biases = (
            self.build_attention_biases(cf_item_ids, text_item_ids, attention_mask)
            if self.uses_graph_bias
            else None
        )
        model_kwargs = dict(kwargs)
        if self.uses_item_embeddings:
            model_kwargs["cf_item_ids"] = cf_item_ids
            model_kwargs["rec_embedding"] = rec_embedding
        try:
            outputs = self.llama_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                **model_kwargs,
            )
        finally:
            self.attention_bias_hooks.biases = None

        loss = None
        if labels is not None:
            shift_logits = outputs.logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous().to(shift_logits.device)
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.shape[-1]),
                shift_labels.view(-1),
                ignore_index=-100,
            )
        return {"loss": loss, "logits": outputs.logits}

    def gradient_checkpointing_enable(
        self, gradient_checkpointing_kwargs: dict | None = None
    ) -> None:
        self.llama_model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs=gradient_checkpointing_kwargs
        )

    def gradient_checkpointing_disable(self) -> None:
        self.llama_model.gradient_checkpointing_disable()

    def component_state_dict(self) -> dict[str, torch.Tensor]:
        return {
            key: value
            for key, value in self.state_dict().items()
            if not key.startswith("llama_model.")
        }

    def load_components(self, checkpoint_path: str | Path) -> None:
        path = Path(checkpoint_path) / "model_components.pt"
        if not path.is_file():
            raise FileNotFoundError(f"Missing model component checkpoint: {path}")
        incompatible = self.load_state_dict(
            torch.load(path, map_location="cpu", weights_only=True), strict=False
        )
        unexpected = [
            key for key in incompatible.unexpected_keys if not key.startswith("llama_model.")
        ]
        if unexpected:
            raise RuntimeError(f"Unexpected model component keys: {unexpected}")
