#!/bin/bash
#SBATCH --job-name=cifar100_dist
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=32
#SBATCH --partition=gpu_partition  # Change to your partition name
#SBATCH --output=logs/cifar_%j.log

# Networking Configuration
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=29505
export NCCL_DEBUG=INFO

# Launch with torchrun
srun torchrun \
    --nnodes=2 \
    --nproc_per_node=8 \
    --rdzv_id=$SLURM_JOB_ID \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    cifar_hpc_test.py