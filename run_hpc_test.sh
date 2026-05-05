#!/bin/bash
#SBATCH --job-name=pytorch_ddp_test
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1          # ONE launcher per node
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=32
#SBATCH --partition=gpu
#SBATCH --output=ddp_%j.log

# Master node
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=29500

# NCCL
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0
export NCCL_P2P_DISABLE=0
export NCCL_SOCKET_IFNAME=eth0   # or ib0

echo "MASTER_ADDR=$MASTER_ADDR"
echo "NODE_RANK=$SLURM_NODEID"

# Launch
srun /opt/pyenv/bin/torchrun \
    --nnodes=$SLURM_JOB_NUM_NODES \
    --nproc_per_node=8 \
    --node_rank=$SLURM_NODEID \
    --rdzv_id=$SLURM_JOB_ID \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    dist_test.py
    
