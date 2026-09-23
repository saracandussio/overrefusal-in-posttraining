#!/bin/bash
#SBATCH --no-requeue
#SBATCH --job-name="overrefusal-olmo3-missing-sources"
#SBATCH --partition=lovelace
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=4:00:00
#SBATCH --mem=100G
#SBATCH --cpus-per-task=8
#SBATCH --output=slurm_outputs/exp_olmo3_missing_%A_%a.out
#SBATCH --array=0-3
#SBATCH --export=ALL

# ---------------------------------------------------------------------------
# OLMo3 — genera le tracce per i 5 dataset assenti da raw_results.csv
# (jailbreakbench, xstest, alpaca, advbench, wildguard), su ENTRAMBI i
# system prompt (none + mistral_safety), dato che i 5 source già presenti
# per OLMo3 coprono entrambi e vogliamo evitare una copertura asimmetrica.
#
# Un solo job per checkpoint copre entrambi i system-prompt in una singola
# chiamata a run_experiment.py (il modello viene caricato una volta sola,
# non due).
#
# Verificato con --dry-run prima di questo lancio (vedi conversazione):
# tutti e 5 i dataset caricano puliti con dataset_loader.py patchato
# (fix: jailbreakbench config="behaviors" + split="harmful", wildguard
# config="wildguardtest" + split="test", xstest/alpaca/advbench loader
# mancanti aggiunti). Se questo job fallisce su un errore di dataset
# loading invece che di generazione, i fix potrebbero non essere stati
# applicati su questa macchina/checkpoint del codice — controllare
# data/dataset_loader.py prima di rilanciare.
#
# NOTA: config_olmo3.py importa da "datasets_config" (plurale) ma il file
# sul disco è "dataset_config.py" (singolare) — verificare che l'alias/fix
# sia applicato, altrimenti il job fallisce subito con ModuleNotFoundError
# prima ancora di caricare i dataset.
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

echo "=== Job $SLURM_ARRAY_TASK_ID — checkpoint: $CHECKPOINT | system-prompts: none, mistral_safety ==="
echo "=== Datasets: jailbreakbench xstest alpaca advbench wildguard ==="
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

# Tmp dir dedicata per job (evita collisioni su nodi condivisi).
# NOTA: in un run precedente "mkdir: cannot create directory
# '/scratch/scandussio': Permission denied" è comparso nei log senza far
# fallire il job (lo script continua comunque). Se TMPDIR non viene creata
# per davvero, eventuali tool che vi scrivono dentro potrebbero fallire in
# modo meno esplicito più avanti — verificare se serve creare
# /scratch/scandussio una volta a monte con i permessi giusti.
export TMPDIR="/scratch/scandussio/tmp/overrefusal_olmo3_missing_${SLURM_ARRAY_TASK_ID}"
mkdir -p "$TMPDIR"

# ---------------------------------------------------------------------------
# Run — tutti i dataset mancanti, entrambi i system prompt, un checkpoint
# ---------------------------------------------------------------------------
python run_experiment.py \
    --config config_olmo3 \
    --datasets jailbreakbench xstest alpaca advbench wildguard \
    --checkpoints "$CHECKPOINT" \
    --system-prompts none mistral_safety

EXIT_CODE=$?

rm -rf "$TMPDIR"
echo "=== Done: $CHECKPOINT × [none, mistral_safety] | Exit: $EXIT_CODE | End: $(date) ==="
exit $EXIT_CODE
