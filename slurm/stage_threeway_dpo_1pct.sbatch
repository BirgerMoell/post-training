#!/bin/bash
#SBATCH --job-name=stage-dpo-1pct
#SBATCH --account=project_465002530
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
set -euo pipefail
module purge
CONTAINER=/scratch/project_465002530/users/bmoell/containers/laif-rocm-6.4.4-pytorch-2.9.1-te-2.4.0-fa-2.8.0-triton-3.2.0.sif
OVERLAY=/scratch/project_465002530/users/bmoell/pylibs-overlay
BIND=/pfs,/scratch,/flash,/project,/projappl,/appl,/opt/cray,/var/spool/slurmd
singularity exec -B "$BIND" --overlay "$OVERLAY" "$CONTAINER" bash -lc "export PYTHONPATH=$OVERLAY:$SLURM_SUBMIT_DIR/src; cd $SLURM_SUBMIT_DIR; python3 scripts/stage_threeway_dpo_1pct.py"
