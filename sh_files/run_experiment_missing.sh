#!/bin/bash
#SBATCH --no-requeue
#SBATCH --job-name="overrefusal-missing-sources"
#SBATCH --partition=Main
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=4:00:00
#SBATCH --mem=100G
#SBATCH --cpus-per-task=8
#SBATCH --output=slurm_outputs/exp_missing_%A_%a.out
#SBATCH --array=0-3
#SBATCH --export=ALL

# ---------------------------------------------------------------------------
# Solo i 4 checkpoint × system-prompt "none" (niente mistral_safety, come
# richiesto) e SOLO i due dataset che risultano assenti dalle attivazioni
# già pushate su saracandu/olmo-activations (verificato con check_sources.py):
# jailbreakbench (harmful) e wildguard (harmless/mixed).
#
# Le chiavi --datasets DEVONO corrispondere esattamente a quelle registrate
# in dataset_config.py / ALL_DATASETS: or_bench, false_reject, harmbench,
# jailbreakbench, wildguard, toxicchat, beavertails. Chiavi diverse (es.
# xstest, alpaca, advbench, come nello script originale) vengono rifiutate
# silenziosamente da run_experiment.py — logger.error + return, nessun
# crash, nessun output utile. Verificare sempre nei log del job che compaia
# "Loading datasets: ['jailbreakbench', 'wildguard']" e non un errore
# "Unknown dataset key(s)".
# ---------------------------------------------------------------------------

CHECKPOINTS=(
    "base"
    "sft"
    "dpo"
    "final"
)

CHECKPOINT="${CHECKPOINTS[$SLURM_ARRAY_TASK_ID]}"

if [[ -z "$CHECKPOINT" ]]; then
    echo "ERROR: checkpoint non definito per SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID"
    exit 1
fi

echo "=== Job $SLURM_ARRAY_TASK_ID — checkpoint: $CHECKPOINT | system-prompt: none | dataset: wildguard ==="
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
export TMPDIR="/scratch/scandussio/tmp/overrefusal_missing_${SLURM_ARRAY_TASK_ID}"
mkdir -p "$TMPDIR"

# ---------------------------------------------------------------------------
# Run — SOLO wildguard, mai generato finora (jailbreakbench/xstest/alpaca/
# advbench sono già in raw_results.csv: per quelli serve solo extract_and_push.py
# su un CSV pre-filtrato con filter_missing_sources.py, non una nuova
# generazione GPU qui).
# ---------------------------------------------------------------------------
python run_experiment.py \
    --config config_olmo2 \
    --datasets wildguard \
    --checkpoints "$CHECKPOINT" \
    --system-prompts none

EXIT_CODE=$?

rm -rf "$TMPDIR"
echo "=== Done: $CHECKPOINT × none | Exit: $EXIT_CODE | End: $(date) ==="
exit $EXIT_CODE
