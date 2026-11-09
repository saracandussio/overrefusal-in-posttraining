#!/bin/bash
#SBATCH -p lovelace
#SBATCH -A ai-lab
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH -o slurm_outputs/%x-%j.out
cd ~/overrefusal-in-posttraining
source .overenv/bin/activate
export PYTHONUNBUFFERED=1
python scripts/intervene_ab.py "$@" 2>&1 | grep --line-buffered -v "HTTP Request"
