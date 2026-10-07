#!/bin/bash
#SBATCH --job-name=guardian_chunk_pipeline
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --partition=stud
#SBATCH --qos=stud
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=out/%x_%j.out
#SBATCH --error=err/%x_%j.err

set -euo pipefail

PROJECT_DIR="$HOME/generation/chunks_embedding"
INPUT_PARQUET="$PROJECT_DIR/articles_clean_with_both_lemma_columns.parquet"
OUTPUT_NPY="$PROJECT_DIR/chunk_embeddings_small.npy"
OUTPUT_LIGHT_PARQUET="$PROJECT_DIR/chunked_df_small_light.parquet"

mkdir -p "$PROJECT_DIR/out" "$PROJECT_DIR/err"
cd "$PROJECT_DIR"

module load /software/modules/miniconda3
eval "$(conda shell.bash hook)"
conda activate guardian_embed

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export TOKENIZERS_PARALLELISM=false

python chunks_embedding_full_pipeline.py \
  --input_parquet "$INPUT_PARQUET" \
  --output_npy "$OUTPUT_NPY" \
  --output_light_parquet "$OUTPUT_LIGHT_PARQUET" \
  --model_name "BAAI/bge-base-en-v1.5" \
  --max_tokens 512 \
  --overlap_ratio 0.15 \
  --batch_size 128

conda deactivate
module unload /software/modules/miniconda3
