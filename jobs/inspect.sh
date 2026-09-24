#!/bin/bash
#SBATCH --job-name=inspect
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=slurm_outputs/%x-%j.out

# Controlli prompt per prompt su una cella: sbatch jobs/inspect.sh olmo2 16 pre_gen
source .overenv/bin/activate
set -euo pipefail
export HF_HOME=/share/ai-lab/scandussio/hf_cache
export PYTHONUNBUFFERED=1
FAMILY=${1:?famiglia}; LAYER=${2:-16}; POS=${3:-pre_gen}

python scripts/inspect_cell.py --family "$FAMILY" --checkpoint sft__none --layer "$LAYER" \
    --position "$POS" --compare dpo__none
python scripts/inspect_cell.py --family "$FAMILY" --checkpoint dpo__none --layer "$LAYER" \
    --position "$POS"
