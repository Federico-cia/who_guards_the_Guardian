#!/bin/bash
#SBATCH --job-name=electra-mnli
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --partition=stud
#SBATCH --qos=stud

set -e

mkdir -p logs

module purge

source ~/venvs/guardian_electra/bin/activate

export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4
export NCCL_DEBUG=WARN

python finetune_mnli.py \
  --model_dir ./guardian_electra_small_3ep_1619/discriminator \
  --output_dir ./guardian_electra_mnli \
  --max_seq_length 256 \
  --num_train_epochs 3 \
  --per_device_train_batch_size 64 \
  --per_device_eval_batch_size 64 \
  --learning_rate 2e-5 \
  --weight_decay 0.01 \
  --warmup_ratio 0.06 \
  --num_proc 4 \
  --mnli_train_file "$HOME/data/mnli_train.parquet" \
  --mnli_validation_matched_file "$HOME/data/mnli_validation_matched.parquet" \
  --mnli_validation_mismatched_file "$HOME/data/mnli_validation_mismatched.parquet"
