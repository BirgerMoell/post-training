#!/bin/bash
#SBATCH --job-name=stage-dpo-full
#SBATCH --account=project_465002530
#SBATCH --partition=small
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=1-00:00:00
#SBATCH --output=/flash/project_465002530/users/bmoell/helpsteer3-dpo/logs/%j.out
#SBATCH --error=/flash/project_465002530/users/bmoell/helpsteer3-dpo/logs/%j.err

set -euo pipefail
module purge

CONTAINER=/scratch/project_465002530/users/bmoell/containers/laif-rocm-6.4.4-pytorch-2.9.1-te-2.4.0-fa-2.8.0-triton-3.2.0.sif
BIND=/pfs,/scratch,/flash,/project,/projappl,/appl,/opt/cray,/var/spool/slurmd
HF_CACHE=/flash/project_465002530/users/bmoell/helpsteer3-dpo/artifacts/cache/full-data-huggingface

mkdir -p "$HF_CACHE"
srun --wait=60 --kill-on-bad-exit=1 singularity exec -B "$BIND" "$CONTAINER" bash -lc "
  set -euo pipefail
  export HF_HOME=$HF_CACHE HF_HUB_CACHE=$HF_CACHE/hub HUGGINGFACE_HUB_CACHE=$HF_CACHE/hub
  export HF_DATASETS_CACHE=$HF_CACHE/datasets HF_HUB_DISABLE_TELEMETRY=1
  cd $SLURM_SUBMIT_DIR
  python3 /flash/project_465002530/users/bmoell/helpsteer3-dpo/scripts/stage_threeway_dpo_full.py
"
