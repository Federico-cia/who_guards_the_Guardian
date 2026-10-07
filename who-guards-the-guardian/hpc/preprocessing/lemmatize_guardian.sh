#!/bin/bash
#SBATCH --job-name=lemmatize_guardian
#SBATCH --output=lemmatize_guardian.out
#SBATCH --error=lemmatize_guardian.err
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

module load python

python lemmatize_guardian.py