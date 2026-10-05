#!/bin/bash
#SBATCH --job-name=oellm-hs3-dpo-4gpu
#SBATCH --account=project_465002530
#SBATCH --partition=dev-g
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=14
#SBATCH --mem=480G
#SBATCH --time=0-00:30:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

set -euo pipefail
module purge

CONTAINER=/scratch/project_465002530/users/bmoell/containers/laif-rocm-6.4.4-pytorch-2.9.1-te-2.4.0-fa-2.8.0-triton-3.2.0.sif
OVERLAY=/scratch/project_465002530/users/bmoell/pylibs-overlay
BIND=/pfs,/scratch,/flash,/project,/projappl,/appl,/opt/cray,/var/spool/slurmd
HF_CACHE=/scratch/project_465002530/users/bmoell/helpsteer3-dpo/artifacts/cache/huggingface
TRAIN_CONFIG=configs/trl/oellm-9b-helpsteer3-dpo-lumi.yaml
GPUS_PER_NODE=4

mkdir -p logs "$HF_CACHE" /scratch/project_465002530/users/bmoell/helpsteer3-dpo/outputs

export MASTER_ADDR
MASTER_ADDR=$(hostname)
export MASTER_PORT=29501
export NCCL_SOCKET_IFNAME=hsn0,hsn1,hsn2,hsn3
export NCCL_NET_GDR_LEVEL=PHB
export OMP_NUM_THREADS=7
export PYTORCH_HIP_ALLOC_CONF=expandable_segments:True
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

WORLD_SIZE=4
echo "HelpSteer3 DPO 4-GPU smoke: world=$WORLD_SIZE"

srun --label --export=ALL --wait=60 --kill-on-bad-exit=1 \
  singularity exec -B "$BIND" "$CONTAINER" bash -lc "
    set -euo pipefail
    export PYTHONPATH=$OVERLAY:$SLURM_SUBMIT_DIR/src:\${PYTHONPATH:-}
    export HF_HOME=$HF_CACHE HF_HUB_CACHE=$HF_CACHE/hub HUGGINGFACE_HUB_CACHE=$HF_CACHE/hub
    export HF_DATASETS_CACHE=$HF_CACHE/datasets
    export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
    export HF_HUB_DISABLE_TELEMETRY=1 WANDB_MODE=disabled
    export ACCELERATE_USE_FSDP=true FSDP_CPU_RAM_EFFICIENT_LOADING=1
    python3 -c 'import omegaconf; print("omegaconf", omegaconf.__version__)'
    cd $SLURM_SUBMIT_DIR
    accelerate launch \
      --config_file configs/accelerate/fsdp_lumi.yaml \
      --num_machines 1 \
      --num_processes 4 \
      --machine_rank 0 \
      --main_process_ip '$MASTER_ADDR' \
      --main_process_port '$MASTER_PORT' \
      scripts/train.py \
        --config $TRAIN_CONFIG \
        run_name=oellm-9b-helpsteer3-dpo-4gpu-smoke \
        training.num_train_epochs=null \
        training.max_steps=1 \
        dpo.max_seq_length=1024 \
        dpo.precompute_ref_log_probs=false \
        evaluation.final_eval=false \
        checkpointing.save_steps=100 \
        checkpointing.save_final_model=false \
        logging.report_to='[none]' \
        accelerate.use_deepspeed=false \
        deepspeed=null
  "
