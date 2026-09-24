#!/bin/bash
#SBATCH --job-name=overref
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=slurm_outputs/%x-%j.out
# #SBATCH --partition=...        # la partizione CPU del cluster, se serve

FAMILY=${1:?uso: sbatch jobs/run_family.sh olmo2}

source .overenv/bin/activate
set -euo pipefail                  # controllo stretto solo dopo l'attivazione

export HF_HOME=/share/ai-lab/scandussio/hf_cache
export PYTHONUNBUFFERED=1          # log in tempo reale

step() { echo; echo "=== $(date '+%H:%M') $* ==="; }
python -c "import overrefusal, sys; print('env ok:', sys.executable)"

step "axis"
python scripts/axis.py --family "$FAMILY" --per-source --bootstrap 300
python scripts/fig_axis.py --family "$FAMILY"

step "drift"
python scripts/drift.py --family "$FAMILY" --positions pre_gen first_gen

step "scatter"
for pos in pre_gen first_gen; do
  python scripts/fig_scatter.py --family "$FAMILY" --position "$pos" --layers 8 19 26 31
done

step "probe (veloce, senza leave-one-source-out)"
python scripts/probe.py --family "$FAMILY" --positions pre_gen first_gen --no-loso

step "fine"
