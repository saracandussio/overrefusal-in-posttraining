#!/bin/bash
#SBATCH --job-name=nosys
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=slurm_outputs/%x-%j.out

# Run senza prompt di sistema: genera ed estrae UN checkpoint post-trained.
# Scrive solo in results_nosys/ e nelle cartelle <stage>__nosys su HF;
# i risultati originali non vengono toccati (gli script rifiutano il resto).
# Prima, una volta sola: OVERREFUSAL_VARIANT=nosys python scripts/seed_variant.py --family olmo3
#   sbatch -p lovelace --gres=gpu:1 jobs/nosys.sh olmo3 sft__nosys
#   sbatch -p lovelace --gres=gpu:1 jobs/nosys.sh olmo3 dpo__nosys
#   sbatch -p lovelace --gres=gpu:1 jobs/nosys.sh olmo3 final__nosys

FAMILY=${1:?famiglia}; CKPT=${2:?checkpoint, es. sft__nosys}
source .overenv/bin/activate
set -euo pipefail
export OVERREFUSAL_VARIANT=nosys
export HF_HOME=/share/ai-lab/scandussio/hf_cache
export HF_TOKEN=${HF_TOKEN:-$(cat ~/.cache/huggingface/token 2>/dev/null || cat ~/.hf_token 2>/dev/null || true)}
[ -n "$HF_TOKEN" ] || { echo "manca il token HF"; exit 1; }
export PYTHONUNBUFFERED=1
step() { echo; echo "=== $(date '+%H:%M') $* ===" ; }
python -c "import torch; print('GPU:', torch.cuda.get_device_name(0))"
[ -f "results_nosys/$FAMILY/raw_results.csv" ] || { echo "manca il seed: scripts/seed_variant.py"; exit 1; }

step "generazione $CKPT"
python scripts/generate.py --family "$FAMILY" --checkpoints "$CKPT"

step "estrazione $CKPT"
python scripts/extract.py --family "$FAMILY" --checkpoints "$CKPT"

step "fine: ora il giudice, con OVERREFUSAL_VARIANT=nosys"
