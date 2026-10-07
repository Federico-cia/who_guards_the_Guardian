#!/bin/bash
#SBATCH --job-name=guardian-electra-small
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --partition=stud
#SBATCH --qos=stud
set -e

mkdir -p logs

module purge
# Adapt these to your university HPC.
# Examples:
# module load cuda/12.1
# module load python/3.10
source ~/venvs/guardian_electra/bin/activate

export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4
export NCCL_DEBUG=WARN

torchrun \
  --nproc_per_node=1 \
  pretrain_electra.py \
  --train_file ~/data/articles_16_19.parquet \
  --text_column bodyContent \
  --tokenizer_dir ./guardian_electra_tokenizer \
  --output_dir ./guardian_electra_small_3ep_1619 \
  --max_seq_length 256 \
  --num_train_epochs 3 \
  --per_device_train_batch_size 64 \
  --per_device_eval_batch_size 64 \
  --gradient_accumulation_steps 1 \
  --learning_rate 5e-4 \
  --weight_decay 0.01 \
  --warmup_ratio 0.06 \
  --mlm_probability 0.15 \
  --generator_loss_weight 1.0 \
  --discriminator_loss_weight 50.0 \
  --validation_split 0.02 \
  --num_proc 4 \
  --dataloader_num_workers 4 \
  --logging_steps 100 \
  --save_every_epoch
