"""Full-sequence SASRec model used by the original project."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
import torch.nn as nn


@dataclass
class SASRecConfig:
    item_count: int
    hidden_size: int = 64
    max_length: int = 50
    num_layers: int = 2
    num_heads: int = 1
    dropout: float = 0.5


class PointWiseFeedForward(nn.Module):
    def __init__(self, hidden_size: int, dropout: float) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(hidden_size, hidden_size, kernel_size=1)
        self.dropout1 = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(hidden_size, hidden_size, kernel_size=1)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        outputs = self.conv1(inputs.transpose(-1, -2))
        outputs = self.dropout1(outputs)
        outputs = self.relu(outputs)
        outputs = self.conv2(outputs)
        outputs = self.dropout2(outputs).transpose(-1, -2)
        return outputs + inputs


class SASRec(nn.Module):
    def __init__(self, config: SASRecConfig) -> None:
        super().__init__()
        self.config = config
        self.item_num = config.item_count
        self.item_emb = nn.Embedding(
            config.item_count + 1, config.hidden_size, padding_idx=0
        )
        self.pos_emb = nn.Embedding(config.max_length, config.hidden_size)
        self.emb_dropout = nn.Dropout(config.dropout)
        self.attention_layernorms = nn.ModuleList()
        self.attention_layers = nn.ModuleList()
        self.forward_layernorms = nn.ModuleList()
        self.forward_layers = nn.ModuleList()
        self.last_layernorm = nn.LayerNorm(config.hidden_size, eps=1e-8)

        for _ in range(config.num_layers):
            self.attention_layernorms.append(
                nn.LayerNorm(config.hidden_size, eps=1e-8)
            )
            self.attention_layers.append(
                nn.MultiheadAttention(
                    config.hidden_size,
                    config.num_heads,
                    config.dropout,
                )
            )
            self.forward_layernorms.append(
                nn.LayerNorm(config.hidden_size, eps=1e-8)
            )
            self.forward_layers.append(
                PointWiseFeedForward(config.hidden_size, config.dropout)
            )
            # This placement intentionally matches the original implementation.
            self.apply(self._initialize)

    @staticmethod
    def _initialize(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def log2feats(
        self, log_seqs: torch.Tensor, return_attention: bool = False
    ):
        device = log_seqs.device
        seqs = self.item_emb(log_seqs)
        seqs *= self.item_emb.embedding_dim**0.5
        positions = torch.arange(
            log_seqs.shape[1], dtype=torch.long, device=device
        ).unsqueeze(0)
        positions = positions.repeat(log_seqs.shape[0], 1)
        seqs += self.pos_emb(positions)
        seqs = self.emb_dropout(seqs)

        timeline_mask = log_seqs.eq(0)
        seqs *= ~timeline_mask.unsqueeze(-1)
        sequence_length = seqs.shape[1]
        attention_mask = ~torch.tril(
            torch.ones(
                (sequence_length, sequence_length),
                dtype=torch.bool,
                device=device,
            )
        )

        attention_weights = None
        for layer_index in range(len(self.attention_layers)):
            seqs = seqs.transpose(0, 1)
            queries = self.attention_layernorms[layer_index](seqs)
            attention_outputs, attention_weights = self.attention_layers[
                layer_index
            ](
                queries,
                seqs,
                seqs,
                attn_mask=attention_mask,
            )
            seqs = queries + attention_outputs
            seqs = seqs.transpose(0, 1)
            seqs = self.forward_layernorms[layer_index](seqs)
            seqs = self.forward_layers[layer_index](seqs)
            seqs *= ~timeline_mask.unsqueeze(-1)

        log_feats = self.last_layernorm(seqs)
        if return_attention:
            return log_feats, attention_weights[:, -1, :]
        return log_feats

    def forward(self, log_seqs: torch.Tensor) -> torch.Tensor:
        final_features = self.log2feats(log_seqs)[:, -1, :]
        return final_features @ self.item_emb.weight.T

    def full_sequence_loss(
        self,
        histories: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        sequence_with_target = torch.cat([histories, targets.unsqueeze(1)], dim=1)
        sequences, positives, negatives = sample_full_sequence_negatives(
            sequence_with_target, self.item_num
        )
        log_feats = self.log2feats(sequences)
        positive_logits = (log_feats * self.item_emb(positives)).sum(dim=-1)
        negative_logits = (log_feats * self.item_emb(negatives)).sum(dim=-1)
        valid = positives.ne(0)
        criterion = nn.BCEWithLogitsLoss()
        loss = criterion(
            positive_logits[valid],
            torch.ones_like(positive_logits[valid]),
        )
        loss += criterion(
            negative_logits[valid],
            torch.zeros_like(negative_logits[valid]),
        )
        return loss

    def checkpoint(self) -> dict:
        return {"config": asdict(self.config), "model_state": self.state_dict()}


def sample_full_sequence_negatives(
    sequence_with_target: torch.Tensor,
    item_count: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Match the original full SASRec negative-sampling procedure."""
    sequences = sequence_with_target[:, :-1]
    positives = sequence_with_target[:, 1:]
    probabilities = torch.ones(
        (sequences.shape[0], item_count + 1),
        dtype=torch.float,
        device=sequence_with_target.device,
    )
    batch_indices = torch.arange(
        sequence_with_target.shape[0], device=sequence_with_target.device
    ).view(-1, 1)
    probabilities[batch_indices, sequence_with_target] = 0
    probabilities[:, 0] = 0
    negatives = torch.multinomial(
        probabilities,
        sequences.shape[1],
        replacement=False,
    )
    return sequences.long(), positives.long(), negatives.long()
