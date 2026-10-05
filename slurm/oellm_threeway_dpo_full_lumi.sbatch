#!/bin/bash
#SBATCH --job-name=oellm-3way-dpo-full
#SBATCH --account=project_465002530
#SBATCH --partition=small-g
#SBATCH --nodes=4
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=56
#SBATCH --mem=480G
#SBATCH --time=2-00:00:00
#SBATCH --signal=B:USR1@300
#SBATCH --output=/flash/project_465002530/users/bmoell/helpsteer3-dpo/logs/%j.out
#SBATCH --error=/flash/project_465002530/users/bmoell/helpsteer3-dpo/logs/%j.err

set -euo pipefail
module purge

GPUS_PER_NODE=8
TRAIN_CONFIG=/flash/project_465002530/users/bmoell/helpsteer3-dpo/configs/oellm-9b-threeway-dpo-full-lumi.yaml
CONTAINER=/scratch/project_465002530/users/bmoell/containers/laif-rocm-6.4.4-pytorch-2.9.1-te-2.4.0-fa-2.8.0-triton-3.2.0.sif
OVERLAY=/scratch/project_465002530/users/bmoell/pylibs-overlay
BIND=/pfs,/scratch,/flash,/project,/projappl,/appl,/opt/cray,/var/spool/slurmd
HF_CACHE=/flash/project_465002530/users/bmoell/helpsteer3-dpo/artifacts/cache/huggingface

mkdir -p "$HF_CACHE" /flash/project_465002530/users/bmoell/helpsteer3-dpo/outputs
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=$(python3 -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")
export NCCL_SOCKET_IFNAME=hsn0,hsn1,hsn2,hsn3 NCCL_NET_GDR_LEVEL=PHB FI_CXI_DEFAULT_CQ_SIZE=131072
export TORCH_NCCL_TRACE_BUFFER_SIZE=1048576 TORCH_NCCL_DUMP_ON_TIMEOUT=1 TORCH_NCCL_DESYNC_DEBUG=1
export OMP_NUM_THREADS=7 PYTORCH_HIP_ALLOC_CONF=expandable_segments:True PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
WORLD_SIZE=$(( GPUS_PER_NODE * SLURM_NNODES ))

echo "Full three-way DPO: config=$TRAIN_CONFIG nodes=$SLURM_NNODES world=$WORLD_SIZE"
srun --label --export=ALL --wait=60 --kill-on-bad-exit=1 singularity exec -B "$BIND" "$CONTAINER" bash -lc "
  set -euo pipefail
  export PYTHONPATH=$OVERLAY:$SLURM_SUBMIT_DIR/src:\${PYTHONPATH:-}
  export HF_HOME=$HF_CACHE HF_HUB_CACHE=$HF_CACHE/hub HUGGINGFACE_HUB_CACHE=$HF_CACHE/hub HF_DATASETS_CACHE=$HF_CACHE/datasets
  export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 WANDB_MODE=disabled
  export ACCELERATE_USE_FSDP=true FSDP_CPU_RAM_EFFICIENT_LOADING=1
  cd $SLURM_SUBMIT_DIR
  accelerate launch \
    --config_file configs/accelerate/fsdp_lumi.yaml \
    --num_machines $SLURM_NNODES \
    --num_processes $WORLD_SIZE \
    --machine_rank \$SLURM_PROCID \
    --main_process_ip '$MASTER_ADDR' \
    --main_process_port '$MASTER_PORT' \
    scripts/train.py --config $TRAIN_CONFIG accelerate.use_deepspeed=false deepspeed=null
"
