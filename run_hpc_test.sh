#!/bin/bash
#SBATCH --job-name=pytorch_hpc_test
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1          # We launch 1 torchrun process per node
#SBATCH --gres=gpu:8                 # Request all 8 GPUs per node
#SBATCH --cpus-per-task=32           # Adjust based on your CPU core count
#SBATCH --partition=gpu              # Replace with your actual partition name
#SBATCH --output=pytorch_%j.log

# Identify the Master Node
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=29500

# Optimization for NCCL (HPC interconnects)
export NCCL_DEBUG=INFO               # Detailed logs for interconnect handshake
export NCCL_IB_DISABLE=0             # Ensure InfiniBand is enabled if available
export NCCL_P2P_DISABLE=0            # Enable Peer-to-Peer (NVLink)

echo "Master Node: $MASTER_ADDR"

# Launching with torchrun
# --nproc_per_node=8 matches your GPU count per node
srun torchrun \
    --nnodes=2 \
    --nproc_per_node=8 \
    --rdzv_id=$SLURM_JOB_ID \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    dist_test.py
    
