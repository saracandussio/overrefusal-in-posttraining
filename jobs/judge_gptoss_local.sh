#!/bin/bash
#SBATCH --job-name=judge-gptoss
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=slurm_outputs/%x-%j.out

# gpt-oss-120b (MXFP4, ~65 GB) servito con vLLM su una A100 da 80 GB, poi il
# giudice di sempre punta a questo server invece che a ORFEO.
#
# Serve una GPU da 80 GB (A100 80GB, H100); la si chiede al lancio:
#   sbatch -p <partizione> --gres=gpu:1 jobs/judge_gptoss_local.sh olmo2 calibrate
#   sbatch -p <partizione> --gres=gpu:1 jobs/judge_gptoss_local.sh olmo2 base
# calibrate: 300 risposte già giudicate da ORFEO -> accordo (kappa)
# base:      il base rigenerato, scritto in raw_results.csv
#
# Prerequisito, una volta sola (vLLM in un ambiente suo, per non toccare .overenv):
#   python -m venv .vllmenv && .vllmenv/bin/pip install vllm

FAMILY=${1:?famiglia}; MODE=${2:?calibrate oppure base}
set -euo pipefail
export HF_HOME=/share/ai-lab/scandussio/hf_cache
export HF_TOKEN=${HF_TOKEN:-$(cat ~/.cache/huggingface/token 2>/dev/null || cat ~/.hf_token 2>/dev/null || true)}
export PYTHONUNBUFFERED=1
MODEL=openai/gpt-oss-120b
PORT=$((8000 + SLURM_JOB_ID % 1000))
step() { echo; echo "=== $(date '+%H:%M') $* ==="; }

step "avvio vLLM ($MODEL) sulla porta $PORT"
nvidia-smi --query-gpu=name,memory.total --format=csv
.vllmenv/bin/vllm serve "$MODEL" --port "$PORT" --max-model-len 16384 \
    --gpu-memory-utilization 0.92 > "slurm_outputs/vllm-$SLURM_JOB_ID.log" 2>&1 &
SERVER=$!
trap 'kill $SERVER 2>/dev/null' EXIT
for i in $(seq 1 120); do   # fino a 60 minuti: il primo avvio scarica ~65 GB
    curl -sf "http://localhost:$PORT/v1/models" > /dev/null && break
    kill -0 $SERVER 2>/dev/null || { echo "vLLM si è fermato:"; tail -40 "slurm_outputs/vllm-$SLURM_JOB_ID.log"; exit 1; }
    sleep 30
done
curl -sf "http://localhost:$PORT/v1/models" > /dev/null || { echo "vLLM non risponde"; exit 1; }
echo "server pronto"

source .overenv/bin/activate
export JUDGE_API_KEY=local JUDGE_API_BASE_URL="http://localhost:$PORT/v1"

case "$MODE" in
  calibrate)
    step "calibrazione: 300 risposte di sft e dpo già giudicate da ORFEO"
    python scripts/judge.py --family "$FAMILY" --backend api --model "$MODEL" --workers 32 \
        --tag gptoss_local --sample 300 --checkpoints sft__none dpo__none final__none
    python scripts/judge_agreement.py --family "$FAMILY" --tag gptoss_local ;;
  base)
    step "giudice del base rigenerato"
    python scripts/judge.py --family "$FAMILY" --backend api --model "$MODEL" --workers 32 \
        --checkpoints base__none
    python scripts/behavior.py --family "$FAMILY" ;;
  *) echo "modo sconosciuto: $MODE"; exit 1 ;;
esac
step "fine"
