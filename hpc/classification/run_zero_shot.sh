#!/bin/bash
#SBATCH --job-name="guard_zs"
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --partition=stud
#SBATCH --qos=stud
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --output=out/%x_%j.out
#SBATCH --error=err/%x_%j.err
#SBATCH --time=720

module load /software/modules/miniconda3
eval "$(conda shell.bash hook)"
conda activate guardlemma

mkdir -p out err

python -u zero_shot_guardian.py

conda deactivate
module unload /software/modules/miniconda3
