"""Llama beam search that applies constraints before log-softmax."""

from __future__ import annotations

from typing import Union

import torch
from torch import nn
from transformers import LlamaForCausalLM
from transformers.generation.beam_search import BeamScorer
from transformers.generation.configuration_utils import GenerationConfig
from transformers.generation.logits_process import LogitsProcessorList
from transformers.generation.stopping_criteria import StoppingCriteriaList
from transformers.generation.utils import (
    GenerateBeamDecoderOnlyOutput,
    GenerateBeamEncoderDecoderOutput,
)


GenerateBeamOutput = Union[
    GenerateBeamDecoderOnlyOutput,
    GenerateBeamEncoderDecoderOutput,
]


class CustomLlamaForCausalLM(LlamaForCausalLM):
    """Preserve the original constrained-before-softmax beam-search behavior."""

    def _beam_search(
        self,
        input_ids: torch.LongTensor,
        beam_scorer: BeamScorer,
        logits_processor: LogitsProcessorList,
        stopping_criteria: StoppingCriteriaList,
        generation_config: GenerationConfig,
        synced_gpus: bool,
        **model_kwargs,
    ) -> GenerateBeamOutput | torch.LongTensor:
        pad_token_id = generation_config._pad_token_tensor
        eos_token_id = generation_config._eos_token_tensor
        output_attentions = generation_config.output_attentions
        output_hidden_states = generation_config.output_hidden_states
        output_scores = generation_config.output_scores
        output_logits = generation_config.output_logits
        return_dict_in_generate = generation_config.return_dict_in_generate
        sequential = generation_config.low_memory
        do_sample = generation_config.do_sample

        batch_size = len(beam_scorer._beam_hyps)
        num_beams = beam_scorer.num_beams
        batch_beam_size, current_length = input_ids.shape
        model_kwargs = self._get_initial_cache_position(input_ids, model_kwargs)
        if num_beams * batch_size != batch_beam_size:
            raise ValueError(
                f"Expected input batch {num_beams * batch_size}, "
                f"got {batch_beam_size}"
            )

        scores = () if return_dict_in_generate and output_scores else None
        raw_logits = () if return_dict_in_generate and output_logits else None
        beam_indices = (
            tuple(() for _ in range(batch_beam_size))
            if return_dict_in_generate and output_scores
            else None
        )
        decoder_attentions = (
            () if return_dict_in_generate and output_attentions else None
        )
        cross_attentions = (
            () if return_dict_in_generate and output_attentions else None
        )
        decoder_hidden_states = (
            () if return_dict_in_generate and output_hidden_states else None
        )
        if return_dict_in_generate and self.config.is_encoder_decoder:
            encoder_attentions = (
                model_kwargs["encoder_outputs"].get("attentions")
                if output_attentions
                else None
            )
            encoder_hidden_states = (
                model_kwargs["encoder_outputs"].get("hidden_states")
                if output_hidden_states
                else None
            )

        beam_scores = torch.zeros(
            (batch_size, num_beams),
            dtype=torch.float,
            device=input_ids.device,
        )
        beam_scores[:, 1:] = -1e9
        beam_scores = beam_scores.view(batch_size * num_beams)
        peer_finished = False
        decoder_prompt_length = input_ids.shape[-1]

        while self._has_unfinished_sequences(
            peer_finished,
            synced_gpus,
            device=input_ids.device,
        ):
            model_inputs = self.prepare_inputs_for_generation(
                input_ids, **model_kwargs
            )
            if output_attentions:
                model_inputs["output_attentions"] = True
            if output_hidden_states:
                model_inputs["output_hidden_states"] = True

            if sequential:
                raise RuntimeError(
                    "Custom CBS does not support low_memory beam search. "
                    "Set generation_config.low_memory=False."
                )
            else:
                outputs = self(**model_inputs, return_dict=True)

            model_kwargs = self._update_model_kwargs_for_generation(
                outputs,
                model_kwargs,
                is_encoder_decoder=self.config.is_encoder_decoder,
            )
            if synced_gpus and peer_finished:
                current_length += 1
                continue

            next_token_logits = outputs.logits[:, -1, :].clone().float()
            next_token_logits = next_token_logits.to(input_ids.device)

            # This ordering is the intentional difference from standard CBS.
            processed_logits = logits_processor(input_ids, next_token_logits)
            processed_scores = nn.functional.log_softmax(
                processed_logits, dim=-1
            )
            next_token_scores = processed_scores + beam_scores[:, None].expand_as(
                processed_scores
            )

            if return_dict_in_generate:
                if output_scores:
                    scores += (processed_scores,)
                if output_logits:
                    raw_logits += (next_token_logits,)
                if output_attentions:
                    decoder_attentions += (
                        (outputs.decoder_attentions,)
                        if self.config.is_encoder_decoder
                        else (outputs.attentions,)
                    )
                    if self.config.is_encoder_decoder:
                        cross_attentions += (outputs.cross_attentions,)
                if output_hidden_states:
                    decoder_hidden_states += (
                        (outputs.decoder_hidden_states,)
                        if self.config.is_encoder_decoder
                        else (outputs.hidden_states,)
                    )

            vocab_size = next_token_scores.shape[-1]
            next_token_scores = next_token_scores.view(
                batch_size, num_beams * vocab_size
            )
            eos_count = eos_token_id.shape[0] if eos_token_id is not None else 0
            tokens_to_keep = max(2, 1 + eos_count) * num_beams
            if do_sample:
                probabilities = nn.functional.softmax(
                    next_token_scores, dim=-1
                )
                next_tokens = torch.multinomial(
                    probabilities, num_samples=tokens_to_keep
                )
                next_token_scores = torch.gather(
                    next_token_scores, -1, next_tokens
                )
                next_token_scores, sorted_indices = torch.sort(
                    next_token_scores, descending=True, dim=1
                )
                next_tokens = torch.gather(
                    next_tokens, -1, sorted_indices
                )
            else:
                next_token_scores, next_tokens = torch.topk(
                    next_token_scores,
                    tokens_to_keep,
                    dim=1,
                    largest=True,
                    sorted=True,
                )

            next_indices = torch.div(
                next_tokens, vocab_size, rounding_mode="floor"
            )
            next_tokens = next_tokens % vocab_size
            beam_outputs = beam_scorer.process(
                input_ids,
                next_token_scores,
                next_tokens,
                next_indices,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
                beam_indices=beam_indices,
                decoder_prompt_len=decoder_prompt_length,
            )
            beam_scores = beam_outputs["next_beam_scores"]
            beam_next_tokens = beam_outputs["next_beam_tokens"]
            beam_index = beam_outputs["next_beam_indices"]
            input_ids = torch.cat(
                [
                    input_ids[beam_index, :],
                    beam_next_tokens.unsqueeze(-1),
                ],
                dim=-1,
            )
            del outputs

            if model_kwargs.get("past_key_values") is not None:
                model_kwargs["past_key_values"] = self._temporary_reorder_cache(
                    model_kwargs["past_key_values"], beam_index
                )
            if return_dict_in_generate and output_scores:
                beam_indices = tuple(
                    beam_indices[beam_index[index]] + (beam_index[index],)
                    for index in range(len(beam_indices))
                )
            current_length += 1
            if beam_scorer.is_done or all(
                stopping_criteria(input_ids, scores)
            ):
                peer_finished = True

        sequence_outputs = beam_scorer.finalize(
            input_ids,
            beam_scores,
            next_tokens,
            next_indices,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            max_length=stopping_criteria.max_length,
            beam_indices=beam_indices,
            decoder_prompt_len=decoder_prompt_length,
        )
        if not return_dict_in_generate:
            return sequence_outputs["sequences"]
        if not output_scores:
            sequence_outputs["sequence_scores"] = None

        if self.config.is_encoder_decoder:
            return GenerateBeamEncoderDecoderOutput(
                sequences=sequence_outputs["sequences"],
                sequences_scores=sequence_outputs["sequence_scores"],
                scores=scores,
                logits=raw_logits,
                beam_indices=sequence_outputs["beam_indices"],
                encoder_attentions=encoder_attentions,
                encoder_hidden_states=encoder_hidden_states,
                decoder_attentions=decoder_attentions,
                cross_attentions=cross_attentions,
                decoder_hidden_states=decoder_hidden_states,
                past_key_values=model_kwargs.get("past_key_values"),
            )
        return GenerateBeamDecoderOnlyOutput(
            sequences=sequence_outputs["sequences"],
            sequences_scores=sequence_outputs["sequence_scores"],
            scores=scores,
            logits=raw_logits,
            beam_indices=sequence_outputs["beam_indices"],
            attentions=decoder_attentions,
            hidden_states=decoder_hidden_states,
            past_key_values=model_kwargs.get("past_key_values"),
        )
