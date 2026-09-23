#!/bin/bash
#SBATCH --no-requeue
#SBATCH --job-name="overrefusal-exp"
#SBATCH --partition=Main
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=4:00:00
#SBATCH --mem=100G
#SBATCH --cpus-per-task=8
#SBATCH --output=slurm_outputs/exp_%A_%a.out
#SBATCH --array=0-7
#SBATCH --export=ALL

# ---------------------------------------------------------------------------
# Combinazioni checkpoint × system-prompt
# Devono corrispondere alle chiavi in OLMO_CHECKPOINTS e SYSTEM_PROMPTS
# nel tuo config_olmo2.py — modifica qui se hai valori diversi.
# ---------------------------------------------------------------------------
CHECKPOINTS=(
    "base"
    "base"
    "sft"
    "sft"
    "dpo"
    "dpo"
    "final"
    "final"
)

SYSTEM_PROMPTS=(
    "none"
    "mistral_safety"
    "none"
    "mistral_safety"
    "none"
    "mistral_safety"
    "none"
    "mistral_safety"
)

CHECKPOINT="${CHECKPOINTS[$SLURM_ARRAY_TASK_ID]}"
SYSPROMPT="${SYSTEM_PROMPTS[$SLURM_ARRAY_TASK_ID]}"

if [[ -z "$CHECKPOINT" || -z "$SYSPROMPT" ]]; then
    echo "ERROR: combinazione non definita per SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID"
    exit 1
fi

echo "=== Job $SLURM_ARRAY_TASK_ID — checkpoint: $CHECKPOINT | system-prompt: $SYSPROMPT ==="
echo "=== Node: $SLURMD_NODENAME | Start: $(date) ==="

# ---------------------------------------------------------------------------
# Stagger: evita di martellare HuggingFace / il filesystem tutti insieme
# ---------------------------------------------------------------------------
STAGGER=$(( SLURM_ARRAY_TASK_ID * 60 ))
if [[ $STAGGER -gt 0 ]]; then
    echo "=== Stagger ${STAGGER}s ==="
    sleep "$STAGGER"
fi

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

# Tmp dir dedicata per job (evita collisioni su nodi condivisi)
export TMPDIR="/scratch/scandussio/tmp/overrefusal_${SLURM_ARRAY_TASK_ID}"
mkdir -p "$TMPDIR"

# ---------------------------------------------------------------------------
# Run — tutti i dataset, un solo checkpoint e system-prompt per job
# ---------------------------------------------------------------------------
python run_experiment.py \
    --config config_olmo2 \
    --datasets xstest alpaca advbench jailbreakbench \
    --checkpoints "$CHECKPOINT" \
    --system-prompts "$SYSPROMPT"

EXIT_CODE=$?

rm -rf "$TMPDIR"
echo "=== Done: $CHECKPOINT × $SYSPROMPT | Exit: $EXIT_CODE | End: $(date) ==="
exit $EXIT_CODE