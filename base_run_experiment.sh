#!/bin/bash
#SBATCH --no-requeue
#SBATCH --job-name="overrefusal-base-rerun"
#SBATCH --partition=Main
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=4:00:00
#SBATCH --mem=100G
#SBATCH --cpus-per-task=8
#SBATCH --output=slurm_outputs/exp_%A_%a.out
#SBATCH --array=0
#SBATCH --export=ALL

echo "=== Job $SLURM_ARRAY_TASK_ID — checkpoint: base | system-prompt: none ==="
echo "=== Node: $SLURMD_NODENAME | Start: $(date) ==="

# ---------------------------------------------------------------------------
# Ambiente
# ---------------------------------------------------------------------------
module load python/3.11.7-gcc-13.2.0-b7gwkjx
cd /u/scandussio/overrefusal-in-posttraining || exit 1
source .overenv/bin/activate

export HF_HOME=/share/ai-lab/scandussio/hf_cache
export HF_TOKEN=$(cat ~/.hf_token)
if [[ -z "$HF_TOKEN" ]]; then
    echo "ERROR: HF_TOKEN vuoto — controlla ~/.hf_token"
    exit 1
fi

export TMPDIR="/scratch/scandussio/tmp/overrefusal_${SLURM_ARRAY_TASK_ID}"
mkdir -p "$TMPDIR"

# ---------------------------------------------------------------------------
# Run — tutti i dataset, solo base × none
# ---------------------------------------------------------------------------
python run_experiment.py \
    --config config_olmo2 \
    --checkpoints base \
    --system-prompts none

EXIT_CODE=$?
rm -rf "$TMPDIR"
echo "=== Done: base × none | Exit: $EXIT_CODE | End: $(date) ==="
exit $EXIT_CODE
