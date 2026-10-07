#!/bin/bash
#SBATCH --job-name="guardian_w2v"
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --partition=stud
#SBATCH --qos=stud
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --output=out/%x_%j.out
#SBATCH --error=err/%x_%j.err

set -euo pipefail

mkdir -p out err

module load /software/modules/miniconda3
eval "$(conda shell.bash hook)"
conda activate base

python guardian_w2v.py

conda deactivate
module unload /software/modules/miniconda3
