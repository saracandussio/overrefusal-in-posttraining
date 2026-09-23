#!/bin/bash
#SBATCH --no-requeue
#SBATCH --job-name="overrefusal-olmo2-extract-missing"
#SBATCH --partition=Main
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=4:00:00
#SBATCH --mem=100G
#SBATCH --cpus-per-task=8
#SBATCH --output=slurm_outputs/extract_olmo2_missing_%A_%a.out
#SBATCH --array=0-3
#SBATCH --export=ALL

# ---------------------------------------------------------------------------
# Estrae le attivazioni MANCANTI per OLMo2 sulla repo esistente
# (saracandu/olmo-activations): jailbreakbench, xstest, alpaca, advbench
# (già in raw_results.csv da prima di questa conversazione) + wildguard
# (generato dal job 92018, in coda con questo tramite --dependency=afterok
# — va lanciato con quella dipendenza, non da solo, altrimenti wildguard
# non sarà ancora nel CSV).
#
# Solo system-prompt "none" (--checkpoint-filter esatto per tag, come
# richiesto: niente mistral_safety per OLMo2).
#
# Step 1: filtra il CSV alle sole righe mancanti (niente ri-estrazione
#         inutile di or_bench/beavertails/harmbench/toxicchat/false_reject,
#         già su HF).
# Step 2: extract_and_push.py con --model-family olmo2 (il vero
# extract_and_push.py sul cluster è ora una v3 con interfaccia diversa da
# quella patchata in questa conversazione: usa --model-family {olmo2,olmo3}
# invece di --config <modulo>. Stesso problema risolto — model id non più
# hardcoded — ma nomi di flag diversi. Verificare sempre con --help se
# questo script diverge di nuovo in futuro.
#
# Lancio raccomandato (dall'esterno, non da questo script):
#   sbatch --dependency=afterok:92018 run_extract_olmo2_missing.sh
# ---------------------------------------------------------------------------

CHECKPOINT_TAGS=(
    "base__none"
    "sft__none"
    "dpo__none"
    "final__none"
)

TAG="${CHECKPOINT_TAGS[$SLURM_ARRAY_TASK_ID]}"

if [[ -z "$TAG" ]]; then
    echo "ERROR: checkpoint-tag non definito per SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID"
    exit 1
fi

echo "=== Job $SLURM_ARRAY_TASK_ID — checkpoint-tag: $TAG | repo: saracandu/olmo-activations ==="
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

export TMPDIR="/scratch/scandussio/tmp/overrefusal_extract_olmo2_missing_${SLURM_ARRAY_TASK_ID}"
mkdir -p "$TMPDIR"

# ---------------------------------------------------------------------------
# Step 1 — filtra il CSV completo alle sole righe mancanti (una volta per
# task array; leggero, CPU-only, nessun conflitto se i 4 task lo rifanno
# in parallelo su file di output diversi grazie a --out per-tag).
# ---------------------------------------------------------------------------
FILTERED_CSV="results/olmo2/raw_results_missing_sources_${TAG}.csv"

python filter_missing_sources.py \
    --csv results/olmo2/raw_results.csv \
    --sources jailbreakbench xstest alpaca advbench wildguard \
    --checkpoint-suffix "__none" \
    --out "$FILTERED_CSV"

if [[ ! -s "$FILTERED_CSV" ]]; then
    echo "ERROR: $FILTERED_CSV vuoto o non creato — controllare che wildguard sia stato generato (job 92018 completato con successo)."
    exit 1
fi

# ---------------------------------------------------------------------------
# Step 2 — estrazione, un solo checkpoint-tag per questo task array
# ---------------------------------------------------------------------------
python analysis/extract_and_push.py \
    --csv "$FILTERED_CSV" \
    --model-family olmo2 \
    --hf-repo saracandu/olmo-activations \
    --checkpoint-filter "$TAG" \
    --batch-size 32 --device cuda

EXIT_CODE=$?

rm -f "$FILTERED_CSV"
rm -rf "$TMPDIR"
echo "=== Done: $TAG | Exit: $EXIT_CODE | End: $(date) ==="
exit $EXIT_CODE
