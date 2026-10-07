# Data and generated artifacts

This repository intentionally does not redistribute the complete Guardian corpus or the largest generated artifacts.

## Source corpus

The notebook uses a Guardian article dataset covering January 2016 through June 2022. In the original project workspace the raw file was named:

```text
guardian_articles.parquet
```

The notebook describes the source as a Kaggle dataset, but the exact Kaggle URL is not recorded in the submitted notebook/report files available here. Add the original dataset citation/link here if it can be recovered.

## Large or private artifacts not committed

The original workflow referenced files such as:

```text
articles_clean.csv
articles_clean_with_both_lemma_columns.parquet
articles_clean_with_lemmas_and_zs_scores.parquet
articles_scored.parquet
articles_classified_csv
embeddings_full.npy
articles_metadata_small.parquet
chunk_embeddings_small.npy
chunked_df_small_light.parquet
chunked_df_small_bm25.parquet
bm25_index_small.pkl
guardian_word2vec.model
guardian_electra_tokenizer/
guardian_electra_small_3ep_1619/
guardian_electra_mnli/
```

These are generated from the source corpus and/or trained models and should be recreated with the scripts in `scripts/`.

## Annotation data

Two original annotation workbooks are intentionally not included in this public-ready repository bundle:

```text
manual_annotation_60.xlsx
manual_annotation_8000.xlsx
```

Both contain full Guardian article bodies. Keep them in private storage unless redistribution of the underlying article text is permitted.

`annotations/manual_annotation_logistic_training_set.csv` is retained because it is a small legacy label file referenced by the notebook. The final classifier in the notebook uses the 8,000-article annotation workbook instead.

## Bias probe

The synthetic 4,500-pair nationality bias-probe input is included under `bias_probe/`, while row-level predictions and group-level metrics are under `../results/bias/`.
