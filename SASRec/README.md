# SASRec preprocessing

`main.py` trains and evaluates the original full-sequence SASRec using the root-level
`data/<dataset>/` directory. It writes these generated files back to that
dataset directory:

- `sasrec_best.pt`
- `sasrec_metrics.json`
- `item_embeddings.pth`

Example:

```bash
python SASRec/main.py --dataset Clothing
```

The original defaults are preserved:

- 201 maximum epochs;
- batch size 256;
- Adam with learning rate 0.001 and weight decay 0;
- hidden size 64, two blocks, one attention head, dropout 0.5;
- full-sequence positive/negative BCE loss;
- evaluation after every epoch;
- strict NDCG@10 checkpoint improvement and early stopping after 10
  consecutive non-improving test evaluations;
- training and test sequence lengths of 11 and 13, respectively.

The test split is evaluated after every epoch and is used for checkpoint
selection to preserve the original implementation. It is not an untouched
final test set in this SASRec stage.

To re-export embeddings from an existing checkpoint:

```bash
python SASRec/export_item_embeddings.py --dataset Clothing
```
