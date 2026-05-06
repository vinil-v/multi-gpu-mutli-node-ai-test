#!/bin/bash
#SBATCH --job-name=pytorch_ddp_test
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=32
#SBATCH --partition=gpu
#SBATCH --output=ddp_smoketest_%j.log

# Master node
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)

# Random port (range 10000–65535 to avoid well-known ports)
export MASTER_PORT=$(shuf -i 10000-65535 -n 1)

# NCCL
export NCCL_DEBUG=WARN
export NCCL_IB_DISABLE=0
export NCCL_P2P_DISABLE=0
export NCCL_SOCKET_IFNAME=eth0

echo "MASTER_ADDR=$MASTER_ADDR"
echo "MASTER_PORT=$MASTER_PORT"
echo "NODE_RANK=$SLURM_NODEID"

# Project dir
PROJECT_DIR=$SLURM_SUBMIT_DIR

# Activate virtual environment
source $PROJECT_DIR/venv/bin/activate

# Launch
srun $PROJECT_DIR/venv/bin/torchrun \
    --nnodes=$SLURM_JOB_NUM_NODES \
    --nproc_per_node=8 \
    --node_rank=$SLURM_NODEID \
    --rdzv_id=$SLURM_JOB_ID \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    $PROJECT_DIR/dist_test.py
    
