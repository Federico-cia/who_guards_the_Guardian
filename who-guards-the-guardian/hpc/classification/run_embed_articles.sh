#!/bin/bash
#SBATCH --job-name=guardian_bge_embed
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --partition=stud
#SBATCH --qos=stud
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --mem=24G
#SBATCH --time=08:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

set -euo pipefail

module load /software/modules/miniconda3

eval "$(conda shell.bash hook)"
conda activate guardian_embed

PROJECT_DIR="$HOME/nlp_guardian_embedding"
INPUT_CSV="$PROJECT_DIR/data/articles_clean_with_lexicon_and_zs_scores.csv"
OUTPUT_DIR="$PROJECT_DIR/output_bge_base"

mkdir -p "$OUTPUT_DIR"
cd "$PROJECT_DIR"

python embed_articles_bge.py \
  --input_csv "$INPUT_CSV" \
  --output_dir "$OUTPUT_DIR" \
  --model_name "BAAI/bge-base-en-v1.5" \
  --title_col "webTitle" \
  --body_col "bodyContent" \
  --max_chars 4000 \
  --batch_size 128 \
  --save_every 5000 \
  --normalize_embeddings

ls -lh "$OUTPUT_DIR/embeddings_full.npy" "$OUTPUT_DIR/articles_metadata_small.parquet"

conda deactivate
module unload /software/modules/miniconda3
