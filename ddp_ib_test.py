#!/usr/bin/env python3
import os
import sys
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP

def setup_distributed():
    """Initializes the distributed environment using Slurm environment variables."""
    # Slurm automatically provides these to the tasks
    world_size = int(os.environ["WORLD_SIZE"])
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    
    # Master node handling (Rank 0 info)
    master_addr = os.environ["MASTER_ADDR"]
    master_port = os.environ.get("MASTER_PORT", "29500")
    
    if rank == 0:
        print(f"--> Initializing DDP on Master: {master_addr}:{master_port}")
        print(f"--> Total GPUs in World: {world_size}")
        sys.stdout.flush()

    # Initialize the process group with NCCL backend
    dist.init_process_group(
        backend="nccl",
        init_method=f"tcp://{master_addr}:{master_port}",
        world_size=world_size,
        rank=rank
    )
    
    # Assign the specific GPU to this local process
    torch.cuda.set_device(local_rank)
    return rank, local_rank, world_size

def cleanup():
    dist.destroy_process_group()

def run_test():
    rank, local_rank, world_size = setup_distributed()
    
    # InfiniBand / Device verification printout
    print(f"[Node {os.environ['SLURMD_NODENAME']}] Rank {rank} | Local Rank {local_rank} | Device: {torch.cuda.get_device_name(local_rank)}")
    sys.stdout.flush()
    
    # Define a simple linear layer and move it to the targeted GPU
    model = nn.Linear(10, 10).to(local_rank)
    ddp_model = DDP(model, device_ids=[local_rank])
    
    # Create dummy inputs and targets
    inputs = torch.randn(20, 10).to(local_rank)
    targets = torch.randn(20, 10).to(local_rank)
    loss_fn = nn.MSELoss()
    
    # Forward pass
    outputs = ddp_model(inputs)
    loss = loss_fn(outputs, targets)
    
    # Backward pass (Triggers NCCL AllReduce over InfiniBand fabric)
    loss.backward()
    
    # Barrier synchronization ensures all ranks reached this point successfully
    dist.barrier()
    
    if rank == 0:
        print("--> DDP Multi-Node Test Successful! Communication fabric verified.")
        sys.stdout.flush()
        
    cleanup()

if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("CUDA is not available on this node.")
        sys.exit(1)
    run_test()