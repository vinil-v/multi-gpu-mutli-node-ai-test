#!/bin/bash
#SBATCH --job-name=cifar100_dist
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=32
#SBATCH --partition=gpu
#SBATCH --output=cifar_%j.log

# Networking Configuration
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)

# Deterministic port (same on all nodes, avoids collisions)
export MASTER_PORT=$((10000 + SLURM_JOB_ID % 50000))

export NCCL_DEBUG=INFO

# Debug (important for multi-node issues)
echo "[$(hostname)] MASTER_ADDR=$MASTER_ADDR"
echo "[$(hostname)] MASTER_PORT=$MASTER_PORT"
echo "[$(hostname)] NODE_RANK=$SLURM_NODEID"

# Path to your project directory
PROJECT_DIR=$SLURM_SUBMIT_DIR

# Activate venv
source $PROJECT_DIR/venv/bin/activate

# Launch with torchrun
srun $PROJECT_DIR/venv/bin/torchrun \
    --nnodes=$SLURM_JOB_NUM_NODES \
    --nproc_per_node=8 \
    --node_rank=$SLURM_NODEID \
    --rdzv_id=$SLURM_JOB_ID \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    $PROJECT_DIR/cifar_hpc_test.py