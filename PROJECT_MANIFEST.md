# Project manifest

This file maps the final notebook stages to the standalone scripts and main artifacts recovered from the original project folder.

| Stage | Standalone code | HPC launcher | Main input | Main output |
|---|---|---|---|---|
| Lemmatization | `scripts/preprocessing/lemmatize_guardian.py` | `hpc/preprocessing/lemmatize_guardian.sh` | `articles_clean.csv` | `articles_clean_with_both_lemma_columns.parquet` |
| Zero-shot topic scores | `scripts/classification/zero_shot_guardian.py` | `hpc/classification/run_zero_shot.sh` | lemmatized articles | `articles_clean_with_lemmas_and_zs_scores.parquet` |
| Word2Vec lexicon expansion | `scripts/classification/guardian_w2v.py` | `hpc/classification/guardian_w2v.sh` | sentence-level lemma corpus | `guardian_word2vec.model` |
| Article embeddings | `scripts/classification/embed_articles_bge.py` | `hpc/classification/run_embed_articles.sh` | scored article CSV | `embeddings_full.npy`, metadata parquet files |
| ELECTRA tokenizer | `scripts/bias/train_tokenizer.py` | — | Guardian corpus | `guardian_electra_tokenizer/` |
| ELECTRA pretraining | `scripts/bias/pretrain_electra.py` | `hpc/bias/run_pretrain.slurm` | 2016–2019 Guardian corpus | generator/discriminator checkpoints |
| MultiNLI fine-tuning | `scripts/bias/finetune_mnli.py` | `hpc/bias/run_finetune_mnli.slurm` | pretrained discriminator + MultiNLI | `guardian_electra_mnli/` |
| Bias probe evaluation | `scripts/bias/evaluate_bias_probe.py` | — | fine-tuned NLI model + 4,500 probe pairs | prediction and group-metric CSVs |
| RAG chunking + dense embeddings | `scripts/rag/chunks_embedding_full_pipeline.py` | `hpc/rag/chunks_embedding_full_pipeline.sh` | cleaned Guardian corpus | chunk parquet + embedding matrix |
| BM25 assets | `scripts/rag/bm25_assets_parallel.py` | `hpc/rag/bm25_assets.sh` | chunk parquet | BM25-enriched parquet + pickle index |

## Recovered but intentionally not committed

The following source files were supplied during repository reconstruction but are not copied into this public-ready tree because they contain full Guardian article bodies:

- `manual_annotation_60.xlsx`
- `manual_annotation_8000.xlsx`

The full raw corpus, large generated matrices/indexes, and model checkpoints were not uploaded to this reconstruction session and are also intentionally excluded from Git.

## Reconstruction notes

For a public-ready repository, cluster account identifiers and the personal absolute MultiNLI path found in the recovered files were replaced with portable placeholders/CLI arguments. The scientific hyperparameters were left unchanged.

Two recovered-file version inconsistencies are preserved rather than silently rewritten:

1. The notebook documents the Word2Vec output as `word2vec_model.model`, while the recovered `guardian_w2v.py` writes `guardian_word2vec.model`.
2. The recovered bias-probe evaluator computes 0.5/0.7 threshold metrics for neutral and entailment probabilities, while the supplied group-metrics CSV has a different threshold-column layout. This suggests the saved CSV may have been produced by a nearby script revision.

These differences do not change the high-level project description, but they should be resolved if strict end-to-end rerun parity is required.
