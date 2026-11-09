#!/bin/bash
#SBATCH -p lovelace
#SBATCH -A ai-lab
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH -o slurm_outputs/%x-%j.out
cd ~/overrefusal-in-posttraining
source .overenv/bin/activate
export PYTHONUNBUFFERED=1
python scripts/loso_perception.py --family "$1" --layers 16 19 24 26 31 2>&1 | grep --line-buffered -v "HTTP Request"
