# Data and artifacts

## Included datasets

`Toy`, `Clothing`, and `Amazon_Books` are processed Amazon review datasets. `Movielens` is processed MovieLens-10M. The original data sources referenced by the paper are [Amazon Review Data](https://nijianmo.github.io/amazon/index.html) and [MovieLens-10M](https://grouplens.org/datasets/movielens/10m/). Consult the original providers for their data terms.

The released files are processed samples, not the full original corpora. Test row counts may be smaller than the `5000` filename label. No raw-to-processed conversion script or validation split is included.

## Schema

Place a dataset in `data/<dataset>/` with these files:

| File | Contents |
| --- | --- |
| `train_<N>.csv` | Training interaction sequences |
| `test_<M>.csv` | Test interaction sequences |
| `id2name4Rec.json` | JSON object mapping string item IDs to titles |
| `item_embeddings.pth` | SASRec tensor indexed by item ID, including padding row 0 |
| `cooc_matrix.json` | Global item co-occurrence counts from training data |

Required CSV columns are `user_id` and `item_ids`. Extra columns are ignored:

```csv
user_id,item_ids
1,"[12, 45, 91]"
```

Here `[12, 45]` is the history and `91` is the target. Every non-padding item must appear in the catalog. IDs must be positive integers; `0` is reserved for padding. Select files explicitly with `--train-sample N --test-sample M` if multiple files match a split.

## Validate the release

```bash
python scripts/check_data.py
python scripts/check_data.py --dataset Movielens
```

Checks cover CSV schemas, catalog coverage, embedding dimensions and interaction-ID coverage, and co-occurrence records. Missing or malformed artifacts cause a nonzero exit. Catalog items absent from the interaction samples may lack pretrained embeddings; the checker reports this separately. The supplied MovieLens embeddings cover all released interactions but omit 12 catalog-only IDs. Re-training SASRec allocates rows through the maximum catalog ID.

The loader accepts both legacy `"(left, right)": count` co-occurrence objects and sparse COO objects with `format`, `source`, and `pairs` fields. Counts include each unordered pair at most once per training sequence, including its training target. Test sequences are never used to build co-occurrence counts.

## Regenerate artifacts

The supplied artifacts allow starting directly with GALLM training. To replace them:

```bash
python SASRec/main.py --dataset Clothing
python build_cooccurrence.py --dataset Clothing
```

SASRec writes `sasrec_best.pt`, `sasrec_metrics.json`, and `item_embeddings.pth` into the dataset directory. Training logs and an additional checkpoint are written under `SASRec/log/`. See [SASRec defaults](../SASRec/README.md). Re-export embeddings with:

```bash
python SASRec/export_item_embeddings.py --dataset Clothing
```

Regeneration overwrites the corresponding supplied artifact. Use separate data copies or the scripts' output arguments to retain both versions. The original configuration and provenance of supplied embeddings cannot be recovered from the tensors alone.

## Evaluation protocol

The paper describes a chronological 8:1:1 train/validation/test split. This release provides only sampled training and test CSVs. The existing SASRec routine evaluates the test split after each epoch, chooses the best test NDCG@10, and stops after 10 non-improving epochs. This behavior is retained and must be accounted for when interpreting results. Validation-based selection requires a validation split and a corresponding training change.

LLM evaluation measures exact title HR and NDCG over generated candidates. Duplicate item titles are indistinguishable under this evaluation. The release does not establish that supplied artifacts and defaults exactly reproduce every paper result.
