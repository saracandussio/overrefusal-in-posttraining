#!/bin/bash
#SBATCH --job-name=redo-base
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=slurm_outputs/%x-%j.out

# Rigenera ed estrae il base di una famiglia con la cornice "User: ...\nAssistant:".
# Il giudice NON è qui: parte dopo, con jobs/judge_gptoss_local.sh FAMIGLIA base.
# La GPU si chiede al lancio (una qualsiasi da 24 GB in su):
#   sbatch -p <partizione> --gres=gpu:1 jobs/redo_base.sh olmo2

FAMILY=${1:?uso: sbatch jobs/redo_base.sh olmo2}
source .overenv/bin/activate
set -euo pipefail
export HF_HOME=/share/ai-lab/scandussio/hf_cache
export HF_TOKEN=${HF_TOKEN:-$(cat ~/.cache/huggingface/token 2>/dev/null || cat ~/.hf_token 2>/dev/null || true)}
[ -n "$HF_TOKEN" ] || { echo "manca il token HF"; exit 1; }
export PYTHONUNBUFFERED=1
step() { echo; echo "=== $(date '+%H:%M') $* ==="; }
python -c "import torch; print('GPU:', torch.cuda.get_device_name(0))"

step "generazione"
python scripts/generate.py --family "$FAMILY" --checkpoints base__none --redo

step "estrazione, sostituisce le shard del base su HF"
python scripts/extract.py --family "$FAMILY" --checkpoints base__none --replace

step "fine: ora il giudice (jobs/judge_gptoss_local.sh $FAMILY base)"
