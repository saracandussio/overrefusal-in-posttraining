#!/bin/bash
#SBATCH --job-name=judge-api
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=48:00:00
#SBATCH --output=slurm_outputs/%x-%j.out

# Secondo giudice via API (ORFEO), stessa griglia GA/PD, in un file a parte:
# results/<famiglia>/judges/<tag>.csv. Riprende da dove si era fermato.
#   sbatch jobs/judge_api.sh olmo2 deepseek deepseek-ai/DeepSeek-V4-Flash-0731

FAMILY=${1:?famiglia}; TAG=${2:?tag}; MODEL=${3:?modello}
source .overenv/bin/activate
set -euo pipefail
export PYTHONUNBUFFERED=1

python scripts/judge.py --family "$FAMILY" --backend api --model "$MODEL" --tag "$TAG" \
    --checkpoints sft__none dpo__none final__none
python scripts/judge_agreement.py --family "$FAMILY" --tag "$TAG"
