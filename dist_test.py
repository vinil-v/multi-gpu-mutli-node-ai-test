import os
import torch
import torch.distributed as dist

def test_distributed():
    # These are set automatically by torchrun
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    
    # Set the device for this process
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    
    # Initialize process group using NCCL (Standard for NVIDIA GPUs)
    dist.init_process_group(backend="nccl")
    
    # Create a tensor and move to GPU
    tensor = torch.ones(1).to(device) * rank
    
    # Sum tensors across all 16 GPUs
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    
    if rank == 0:
        print(f"-----------------------------------------------------------")
        print(f"--- Cluster Test Results ---")
        print(f"-----------------------------------------------------------")
        print(f"Total GPUs (World Size): {world_size}")
        print(f"Expected Sum: {sum(range(world_size))}")
        print(f"Actual Sum: {tensor.item()}")
        print(f"Status: {'SUCCESS' if tensor.item() == sum(range(world_size)) else 'FAILURE'}")
        print(f"-----------------------------------------------------------")

if __name__ == "__main__":
    test_distributed()