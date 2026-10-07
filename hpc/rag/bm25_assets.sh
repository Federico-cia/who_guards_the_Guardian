#!/bin/bash
#SBATCH --job-name="bm25_assets"
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --partition=stud
#SBATCH --cpus-per-task=8
#SBATCH --output=out/%x_%j.out
#SBATCH --error=err/%x_%j.err
#SBATCH --time=30:00

module load /software/modules/miniconda3
eval "$(conda shell.bash hook)"

conda activate ragbm25

mkdir -p out err

python bm25_assets_parallel.py \
  --input chunked_df_small.parquet \
  --output_parquet chunked_df_small_bm25.parquet \
  --output_bm25 bm25_index_small.pkl \
  --workers 8 \
  --batch_size 2000

conda deactivate
module unload /software/modules/miniconda3