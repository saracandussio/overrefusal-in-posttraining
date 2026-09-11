#!/bin/bash
#SBATCH --no-requeue
#SBATCH --job-name="overrefusal-olmo3-extract"
#SBATCH --partition=lovelace
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=4:00:00
#SBATCH --mem=100G
#SBATCH --cpus-per-task=8
#SBATCH --output=slurm_outputs/extract_olmo3_%A_%a.out
#SBATCH --array=0-7
#SBATCH --export=ALL

# ---------------------------------------------------------------------------
# Estrae le attivazioni per OLMo3 — TUTTI i checkpoint x system-prompt,
# TUTTI i dataset presenti in results/olmo3/raw_results.csv (i 5 già
# esistenti + i 5 appena aggiunti da run_experiment_olmo3_missing.sh).
# Non serve filtrare per source: a differenza di OLMo2, per OLMo3 non
# esiste ANCORA nessuna attivazione pushata — quindi ogni riga del CSV va
# estratta, non solo quelle mancanti.
#
# Repo HF SEPARATA da quella di OLMo2 (saracandu/olmo-activations), per il
# motivo discusso in conversazione: OLMo-2-7B e OLMo-3-7B hanno hidden_size
# e num_hidden_layers identici (4096, 32) — le attivazioni sarebbero
# geometricamente indistinguibili se finissero sotto gli stessi
# checkpoint-tag sulla stessa repo. Da qui in poi extract_and_push.py scrive
# anche una colonna model_id esplicita per riga (patch applicata), che
# aggiunge un secondo livello di sicurezza indipendente dalla repo.
#
# --model-family olmo3 è OBBLIGATORIO. Il vero extract_and_push.py sul
# cluster è una v3 con interfaccia diversa da quella patchata prima in
# questa conversazione: usa --model-family {olmo2,olmo3} al posto di
# --config <modulo>, ma risolve lo stesso identico problema (model id non
# più hardcoded per checkpoint-tag). Verificare sempre con --help se lo
# script diverge ulteriormente in futuro.
#
# 8 combinazioni checkpoint x system-prompt = 8 task array, uno ciascuno,
# usando --checkpoint-filter con match ESATTO (non serve il suffisso "*"
# qui, dato che vogliamo un tag preciso per task, non un gruppo).
# ---------------------------------------------------------------------------

HF_REPO_OLMO3="saracandu/olmo3-activations"

CHECKPOINT_TAGS=(
    "base__none"
    "base__mistral_safety"
    "sft__none"
    "sft__mistral_safety"
    "dpo__none"
    "dpo__mistral_safety"
    "final__none"
    "final__mistral_safety"
)

TAG="${CHECKPOINT_TAGS[$SLURM_ARRAY_TASK_ID]}"

if [[ -z "$TAG" ]]; then
    echo "ERROR: checkpoint-tag non definito per SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID"
    exit 1
fi

echo "=== Job $SLURM_ARRAY_TASK_ID — checkpoint-tag: $TAG | repo: $HF_REPO_OLMO3 ==="
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

export TMPDIR="/scratch/scandussio/tmp/overrefusal_extract_olmo3_${SLURM_ARRAY_TASK_ID}"
mkdir -p "$TMPDIR"

# ---------------------------------------------------------------------------
# Run — estrazione per un solo checkpoint-tag, tutti i dataset presenti
# ---------------------------------------------------------------------------
python analysis/extract_and_push.py \
    --csv results/olmo3/raw_results.csv \
    --model-family olmo3 \
    --hf-repo "$HF_REPO_OLMO3" \
    --checkpoint-filter "$TAG" \
    --batch-size 32 --device cuda

EXIT_CODE=$?

rm -rf "$TMPDIR"
echo "=== Done: $TAG | Exit: $EXIT_CODE | End: $(date) ==="
exit $EXIT_CODE
