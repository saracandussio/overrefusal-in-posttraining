#!/bin/bash
#SBATCH --job-name=centroids
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=slurm_outputs/%x-%j.out

# Distanze dai centroidi harmless/harmful, tutti i checkpoint, layer e posizioni.
#   sbatch jobs/centroids.sh olmo2

FAMILY=${1:?uso: sbatch jobs/centroids.sh olmo2}
source .overenv/bin/activate
set -euo pipefail
export HF_HOME=/share/ai-lab/scandussio/hf_cache
export PYTHONUNBUFFERED=1

python scripts/centroid_distances.py --family "$FAMILY"
