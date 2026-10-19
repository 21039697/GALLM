"""CPU regression checks with a local tiny Llama; no downloads required."""

import os
os.environ.setdefault("USE_TF", "0")

import json
import math
import tempfile
import unittest
from pathlib import Path

import torch
from transformers import LlamaConfig, LlamaForCausalLM

from evaluate_batch_match import evaluate_records
from model.gallm import GALLMRecommender, GraphLlamaForCausalLM
from model.constrained_decoding import CustomLlamaForCausalLM
from model.graph_decoding import CustomLlamaForCausalLM as GraphCBS


def tiny_config():
    return LlamaConfig(
        vocab_size=32, hidden_size=16, intermediate_size=32,
        num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2,
        max_position_embeddings=64, pad_token_id=0, bos_token_id=1,
        eos_token_id=2, attention_dropout=0.0,
    )


class TokenizerStub:
    pad_token_id = 0

    def __len__(self):
        return 32


class PipelineTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)

    def test_exact_title_metrics(self):
        records = [
            {"output": '"A"', "predict": [" A ", "B"]},
            {"output": "B", "predict": ["A", "B"]},
            {"output": "C", "predict": ["A", "B"]},
        ]
        metrics = evaluate_records(records, [1, 2])
        self.assertAlmostEqual(metrics["HR@1"], 1 / 3)
        self.assertAlmostEqual(metrics["HR@2"], 2 / 3)
        self.assertAlmostEqual(metrics["NDCG@2"], (1 + 1 / math.log2(3)) / 3)

    def test_item_embedding_replacement(self):
        model = GraphLlamaForCausalLM(tiny_config()).eval()
        ids = torch.tensor([[1, 5, 6, 7]])
        item_ids = torch.tensor([[0, 0, 1, 0]])
        replacement = torch.randn(1, 4, 16)
        with torch.no_grad():
            actual = model(input_ids=ids, cf_item_ids=item_ids, rec_embedding=replacement)
            embedded = model.model.embed_tokens(ids)
            embedded[:, 2] = replacement[:, 2]
            expected = model(inputs_embeds=embedded)
        torch.testing.assert_close(actual.logits, expected.logits)

    def test_graph_backward_and_checkpoint_components(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            LlamaForCausalLM(tiny_config()).save_pretrained(root / "base")
            torch.save(torch.randn(4, 8), root / "items.pth")
            (root / "cooc.json").write_text(json.dumps({"(1, 2)": 4}))
            model = GALLMRecommender(
                mode="gallm", base_model=str(root / "base"),
                tokenizer=TokenizerStub(), item_embeddings_path=root / "items.pth",
                cooccurrence_path=root / "cooc.json", use_8bit=False,
                prepare_for_training=False,
            ).cpu()
            ids = torch.tensor([[1, 5, 6, 7, 8, 9]])
            cf = torch.tensor([[0, 0, 1, 0, 2, 0]])
            text = torch.tensor([[0, 1, 1, 2, 2, 0]])
            result = model(
                input_ids=ids, attention_mask=torch.ones_like(ids),
                cf_item_ids=cf, text_item_ids=text, labels=ids,
            )
            self.assertTrue(torch.isfinite(result["loss"]))
            result["loss"].backward()
            for group in (model.item_item_bias, model.item_text_bias, model.text_text_bias):
                self.assertIsNotNone(group[0].grad)
                self.assertGreater(group[0].grad.abs().sum().item(), 0)
            self.assertGreater(model.item_projector.layers[0].weight.grad.abs().sum(), 0)
            state = {key: value.detach().clone() for key, value in model.component_state_dict().items()}
            torch.save(state, root / "model_components.pt")
            with torch.no_grad():
                model.item_item_bias[0].fill_(9)
            model.load_components(root)
            torch.testing.assert_close(model.item_item_bias[0], state["item_item_bias.0"])

    def test_catalog_constrained_beam_search(self):
        for model_class in (CustomLlamaForCausalLM, GraphCBS):
            with self.subTest(model=model_class.__module__):
                model = model_class(tiny_config()).eval()
                # Only one title [5, 6] is legal; both beams must follow it.
                def allowed(batch_id, input_ids):
                    suffix = input_ids.tolist()[2:]
                    return [5] if not suffix else ([6] if len(suffix) == 1 else [2])
                result = model.generate(
                    torch.tensor([[1, 3]]), attention_mask=torch.ones(1, 2, dtype=torch.long),
                    num_beams=2, max_new_tokens=3, prefix_allowed_tokens_fn=allowed,
                )
                self.assertEqual(result[0].tolist(), [1, 3, 5, 6, 2])


if __name__ == "__main__":
    unittest.main()
