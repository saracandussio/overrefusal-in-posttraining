#!/bin/bash
#SBATCH --job-name=redo-base
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=slurm_outputs/%x-%j.out
# #SBATCH --partition=...        # la partizione GPU del cluster

# Rifà da zero il base di una famiglia con la cornice "User: ...\nAssistant:":
# generazione, giudice, estrazione. Uso: sbatch jobs/redo_base.sh olmo2
# Le righe vecchie restano in raw_results.csv finché le nuove non sono pronte
# (e in git); su HF le shard vecchie restano nella storia del repo.

FAMILY=${1:?uso: sbatch jobs/redo_base.sh olmo2}
source .overenv/bin/activate
set -euo pipefail
export HF_HOME=/share/ai-lab/scandussio/hf_cache
export PYTHONUNBUFFERED=1
step() { echo; echo "=== $(date '+%H:%M') $* ==="; }

step "generazione (GPU)"
python scripts/generate.py --family "$FAMILY" --checkpoints base__none --redo

step "giudice (API, serve JUDGE_API_KEY nel .env)"
python scripts/judge.py --family "$FAMILY" --checkpoints base__none

step "comportamento aggiornato"
python scripts/behavior.py --family "$FAMILY"

step "estrazione (GPU), sostituisce le shard del base su HF"
python scripts/extract.py --family "$FAMILY" --checkpoints base__none --replace

step "fine"
