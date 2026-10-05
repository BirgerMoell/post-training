#!/bin/bash
#SBATCH --job-name=oellm-hs3-dpo-smoke-v4
#SBATCH --account=project_465002530
#SBATCH --partition=dev-g
#SBATCH --nodes=4
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=56
#SBATCH --mem=480G
#SBATCH --time=0-01:00:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

set -euo pipefail
module purge

CONTAINER=/scratch/project_465002530/users/bmoell/containers/laif-rocm-6.4.4-pytorch-2.9.1-te-2.4.0-fa-2.8.0-triton-3.2.0.sif
OVERLAY=/scratch/project_465002530/users/bmoell/pylibs-overlay
BIND=/pfs,/scratch,/flash,/project,/projappl,/appl,/opt/cray,/var/spool/slurmd
HF_CACHE=/scratch/project_465002530/users/bmoell/helpsteer3-dpo/artifacts/cache/huggingface
TRAIN_CONFIG=configs/trl/oellm-9b-helpsteer3-dpo-lumi.yaml
GPUS_PER_NODE=8

mkdir -p logs "$HF_CACHE" /scratch/project_465002530/users/bmoell/helpsteer3-dpo/outputs

export MASTER_ADDR
MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT
MASTER_PORT=$(python3 -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")
export NCCL_SOCKET_IFNAME=hsn0,hsn1,hsn2,hsn3
export NCCL_NET_GDR_LEVEL=PHB
export FI_CXI_DEFAULT_CQ_SIZE=131072
export TORCH_NCCL_TRACE_BUFFER_SIZE=1048576
export TORCH_NCCL_DUMP_ON_TIMEOUT=1
export TORCH_NCCL_DESYNC_DEBUG=1
export OMP_NUM_THREADS=7
export PYTORCH_HIP_ALLOC_CONF=expandable_segments:True
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

WORLD_SIZE=$(( GPUS_PER_NODE * SLURM_NNODES ))
echo "HelpSteer3 DPO smoke: nodes=$SLURM_NNODES world=$WORLD_SIZE"

srun --label --export=ALL --wait=60 --kill-on-bad-exit=1 \
  singularity exec -B "$BIND" "$CONTAINER" bash -lc "
    set -euo pipefail
    export PYTHONPATH=$OVERLAY:$SLURM_SUBMIT_DIR/src:\${PYTHONPATH:-}
    export HF_HOME=$HF_CACHE HF_HUB_CACHE=$HF_CACHE/hub HUGGINGFACE_HUB_CACHE=$HF_CACHE/hub
    export HF_DATASETS_CACHE=$HF_CACHE/datasets
    export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
    export HF_HUB_DISABLE_TELEMETRY=1 WANDB_MODE=disabled
    export ACCELERATE_USE_FSDP=true FSDP_CPU_RAM_EFFICIENT_LOADING=1
    cd $SLURM_SUBMIT_DIR
    accelerate launch \
      --config_file configs/accelerate/fsdp_lumi.yaml \
      --num_machines $SLURM_NNODES \
      --num_processes $WORLD_SIZE \
      --machine_rank \$SLURM_PROCID \
      --main_process_ip '$MASTER_ADDR' \
      --main_process_port '$MASTER_PORT' \
      scripts/train.py \
        --config $TRAIN_CONFIG \
        run_name=oellm-9b-helpsteer3-dpo-smoke-v4 \
        training.num_train_epochs=null \
        training.max_steps=10 \
        dpo.precompute_ref_log_probs=false \
        evaluation.final_eval=false \
        checkpointing.save_steps=10 \
        checkpointing.save_total_limit=1 \
        checkpointing.save_only_model=true \
        checkpointing.save_final_model=false \
        logging.report_to='[none]' \
        accelerate.use_deepspeed=false \
        deepspeed=null
  "
