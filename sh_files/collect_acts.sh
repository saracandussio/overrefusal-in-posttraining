#!/bin/bash
#SBATCH --no-requeue
#SBATCH --job-name="overrefusal-activations"
#SBATCH --partition=lovelace
#SBATCH --gres=gpu:1g.20gb:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=4:00:00
#SBATCH --mem=100G
#SBATCH --cpus-per-task=8
#SBATCH --output=slurm_outputs/act_%A_%a.out
#SBATCH --array=0-71
#SBATCH --export=ALL

# ---------------------------------------------------------------------------
# 2 families × 4 checkpoints × 9 datasets = 72 jobs
# ---------------------------------------------------------------------------
FAMILIES=(    "olmo2" "olmo2" "olmo2" "olmo2" "olmo2" "olmo2" "olmo2" "olmo2" "olmo2"
              "olmo2" "olmo2" "olmo2" "olmo2" "olmo2" "olmo2" "olmo2" "olmo2" "olmo2"
              "olmo2" "olmo2" "olmo2" "olmo2" "olmo2" "olmo2" "olmo2" "olmo2" "olmo2"
              "olmo2" "olmo2" "olmo2" "olmo2" "olmo2" "olmo2" "olmo2" "olmo2" "olmo2"
              "olmo3" "olmo3" "olmo3" "olmo3" "olmo3" "olmo3" "olmo3" "olmo3" "olmo3"
              "olmo3" "olmo3" "olmo3" "olmo3" "olmo3" "olmo3" "olmo3" "olmo3" "olmo3"
              "olmo3" "olmo3" "olmo3" "olmo3" "olmo3" "olmo3" "olmo3" "olmo3" "olmo3"
              "olmo3" "olmo3" "olmo3" "olmo3" "olmo3" "olmo3" "olmo3" "olmo3" "olmo3" )

CHECKPOINTS=( "base__none"  "base__none"  "base__none"  "base__none"  "base__none"  "base__none"  "base__none"  "base__none"  "base__none"
              "sft__none"   "sft__none"   "sft__none"   "sft__none"   "sft__none"   "sft__none"   "sft__none"   "sft__none"   "sft__none"
              "dpo__none"   "dpo__none"   "dpo__none"   "dpo__none"   "dpo__none"   "dpo__none"   "dpo__none"   "dpo__none"   "dpo__none"
              "final__none" "final__none" "final__none" "final__none" "final__none" "final__none" "final__none" "final__none" "final__none"
              "base__none"  "base__none"  "base__none"  "base__none"  "base__none"  "base__none"  "base__none"  "base__none"  "base__none"
              "sft__none"   "sft__none"   "sft__none"   "sft__none"   "sft__none"   "sft__none"   "sft__none"   "sft__none"   "sft__none"
              "dpo__none"   "dpo__none"   "dpo__none"   "dpo__none"   "dpo__none"   "dpo__none"   "dpo__none"   "dpo__none"   "dpo__none"
              "final__none" "final__none" "final__none" "final__none" "final__none" "final__none" "final__none" "final__none" "final__none" )

DATASETS=(    "advbench" "alpaca" "beavertails" "false_reject" "harmbench" "jailbreakbench" "or_bench" "toxicchat" "xstest"
              "advbench" "alpaca" "beavertails" "false_reject" "harmbench" "jailbreakbench" "or_bench" "toxicchat" "xstest"
              "advbench" "alpaca" "beavertails" "false_reject" "harmbench" "jailbreakbench" "or_bench" "toxicchat" "xstest"
              "advbench" "alpaca" "beavertails" "false_reject" "harmbench" "jailbreakbench" "or_bench" "toxicchat" "xstest"
              "advbench" "alpaca" "beavertails" "false_reject" "harmbench" "jailbreakbench" "or_bench" "toxicchat" "xstest"
              "advbench" "alpaca" "beavertails" "false_reject" "harmbench" "jailbreakbench" "or_bench" "toxicchat" "xstest"
              "advbench" "alpaca" "beavertails" "false_reject" "harmbench" "jailbreakbench" "or_bench" "toxicchat" "xstest"
              "advbench" "alpaca" "beavertails" "false_reject" "harmbench" "jailbreakbench" "or_bench" "toxicchat" "xstest" )

CSVS=(        "results/olmo2/raw_results.csv" "results/olmo2/raw_results.csv" "results/olmo2/raw_results.csv"
              "results/olmo2/raw_results.csv" "results/olmo2/raw_results.csv" "results/olmo2/raw_results.csv"
              "results/olmo2/raw_results.csv" "results/olmo2/raw_results.csv" "results/olmo2/raw_results.csv"
              "results/olmo2/raw_results.csv" "results/olmo2/raw_results.csv" "results/olmo2/raw_results.csv"
              "results/olmo2/raw_results.csv" "results/olmo2/raw_results.csv" "results/olmo2/raw_results.csv"
              "results/olmo2/raw_results.csv" "results/olmo2/raw_results.csv" "results/olmo2/raw_results.csv"
              "results/olmo2/raw_results.csv" "results/olmo2/raw_results.csv" "results/olmo2/raw_results.csv"
              "results/olmo2/raw_results.csv" "results/olmo2/raw_results.csv" "results/olmo2/raw_results.csv"
              "results/olmo2/raw_results.csv" "results/olmo2/raw_results.csv" "results/olmo2/raw_results.csv"
              "results/olmo2/raw_results.csv" "results/olmo2/raw_results.csv" "results/olmo2/raw_results.csv"
              "results/olmo2/raw_results.csv" "results/olmo2/raw_results.csv" "results/olmo2/raw_results.csv"
              "results/olmo2/raw_results.csv" "results/olmo2/raw_results.csv" "results/olmo2/raw_results.csv"
              "results/olmo3/raw_results.csv" "results/olmo3/raw_results.csv" "results/olmo3/raw_results.csv"
              "results/olmo3/raw_results.csv" "results/olmo3/raw_results.csv" "results/olmo3/raw_results.csv"
              "results/olmo3/raw_results.csv" "results/olmo3/raw_results.csv" "results/olmo3/raw_results.csv"
              "results/olmo3/raw_results.csv" "results/olmo3/raw_results.csv" "results/olmo3/raw_results.csv"
              "results/olmo3/raw_results.csv" "results/olmo3/raw_results.csv" "results/olmo3/raw_results.csv"
              "results/olmo3/raw_results.csv" "results/olmo3/raw_results.csv" "results/olmo3/raw_results.csv"
              "results/olmo3/raw_results.csv" "results/olmo3/raw_results.csv" "results/olmo3/raw_results.csv"
              "results/olmo3/raw_results.csv" "results/olmo3/raw_results.csv" "results/olmo3/raw_results.csv"
              "results/olmo3/raw_results.csv" "results/olmo3/raw_results.csv" "results/olmo3/raw_results.csv"
              "results/olmo3/raw_results.csv" "results/olmo3/raw_results.csv" "results/olmo3/raw_results.csv"
              "results/olmo3/raw_results.csv" "results/olmo3/raw_results.csv" "results/olmo3/raw_results.csv"
              "results/olmo3/raw_results.csv" "results/olmo3/raw_results.csv" "results/olmo3/raw_results.csv" )

FAMILY="${FAMILIES[$SLURM_ARRAY_TASK_ID]}"
CHECKPOINT="${CHECKPOINTS[$SLURM_ARRAY_TASK_ID]}"
DATASET="${DATASETS[$SLURM_ARRAY_TASK_ID]}"
CSV="${CSVS[$SLURM_ARRAY_TASK_ID]}"

if [[ -z "$FAMILY" || -z "$CHECKPOINT" || -z "$DATASET" ]]; then
    echo "ERROR: combinazione non definita per SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID"
    exit 1
fi

echo "=== Job $SLURM_ARRAY_TASK_ID — $FAMILY / $CHECKPOINT / $DATASET ==="
echo "=== Node: $SLURMD_NODENAME | Start: $(date) ==="

STAGGER=$(( SLURM_ARRAY_TASK_ID * 15 ))
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

export TMPDIR="/share/ai-lab/scandussio/tmp/overrefusal_act_${SLURM_ARRAY_TASK_ID}"
mkdir -p "$TMPDIR"

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
python analysis/extract_and_push.py \
    --csv "$CSV" \
    --model-family "$FAMILY" \
    --checkpoint-filter "$CHECKPOINT" \
    --dataset-filter "$DATASET" \
    --hf-repo saracandu/olmo-activations \
    --hf-token "$HF_TOKEN" \
    --device cuda

EXIT_CODE=$?
rm -rf "$TMPDIR"
echo "=== Done: $FAMILY / $CHECKPOINT / $DATASET | Exit: $EXIT_CODE | End: $(date) ==="
exit $EXIT_CODE