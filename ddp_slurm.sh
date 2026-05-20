#!/bin/bash
#SBATCH --job-name=pytorch_ddp_test
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=4
#SBATCH --output=ddp_test_%j.out
#SBATCH --error=ddp_test_%j.err
#SBATCH --time=00:15:00

# Exit immediately if any command fails
set -e

# --- Activate Centralized NFS PyTorch Environment ---
# This ensures all spawned processes point to the same /shared binary stack
source /shared/apps/pytorch_env/bin/activate

# --- Network & InfiniBand Tuning Layout ---
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0
export NCCL_P2P_DISABLE=0

# --- PyTorch Master Node Discovery Setup ---
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=29500
export WORLD_SIZE=$SLURM_NTASKS

echo "Master Node IP/Hostname: $MASTER_ADDR"
echo "Total Execution Ranks: $WORLD_SIZE"
echo "Using PyTorch Environment: $(which python3)"

# --- Run the Workload ---
# Passing the active environment context smoothly down to individual srun tasks
srun bash -c "
    source /shared/apps/pytorch_env/bin/activate;
    export RANK=\$SLURM_PROCID;
    export LOCAL_RANK=\$SLURM_LOCALID;
    python3 ddp_ib_test.py
"
