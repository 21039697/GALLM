"""Graph-aware custom CBS model with history embedding replacement."""

from __future__ import annotations

import torch

from model.constrained_decoding import CustomLlamaForCausalLM as CustomCBSLlama


class CustomLlamaForCausalLM(CustomCBSLlama):
    """Add LLaRA/GALLM item embedding replacement to custom beam search."""

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        cache_position=None,
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
                    rec_embedding.to(
                        device=inputs_embeds.device,
                        dtype=inputs_embeds.dtype,
                    ),
                    inputs_embeds,
                )

        return super().forward(
            input_ids=None if inputs_embeds is not None else input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
            **kwargs,
        )
