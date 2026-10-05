#!/bin/bash
#SBATCH --job-name=publish-dpo9000
#SBATCH --account=project_465002530
#SBATCH --partition=small
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=1-00:00:00
#SBATCH --output=/flash/project_465002530/users/bmoell/helpsteer3-dpo/logs/%j.out
#SBATCH --error=/flash/project_465002530/users/bmoell/helpsteer3-dpo/logs/%j.err

set -euo pipefail
module purge

CONTAINER=/scratch/project_465002530/users/bmoell/containers/laif-rocm-6.4.4-pytorch-2.9.1-te-2.4.0-fa-2.8.0-triton-3.2.0.sif
BIND=/pfs,/scratch,/flash,/project,/projappl,/appl,/opt/cray,/var/spool/slurmd
UPLOAD_ROOT=/flash/project_465002530/users/bmoell/helpsteer3-dpo
mkdir -p "$UPLOAD_ROOT/artifacts/cache/upload-xet" "$UPLOAD_ROOT/artifacts/tmp"

srun --wait=60 --kill-on-bad-exit=1 singularity exec -B "$BIND" "$CONTAINER" bash -c "
  set -euo pipefail
  export HF_XET_CACHE=$UPLOAD_ROOT/artifacts/cache/upload-xet
  export TMPDIR=$UPLOAD_ROOT/artifacts/tmp
  export HF_HUB_DISABLE_PROGRESS_BARS=1 HF_HUB_DISABLE_TELEMETRY=1
  python3 -u $UPLOAD_ROOT/scripts/publish_threeway_dpo_step9000.py
"
