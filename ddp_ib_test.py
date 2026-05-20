#!/usr/bin/env python3
import os
import sys
import socket
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP

def get_nvlink_status(local_rank):
    """Checks if NVLink/MIG or basic P2P capability is active for this device."""
    try:
        if torch.cuda.is_available():
            # Basic sanity check for peer-to-peer capability via PyTorch
            current_dev = local_rank
            peer_dev = (local_rank + 1) % torch.cuda.device_count()
            if current_dev != peer_dev and torch.cuda.can_device_access_peer(current_dev, peer_dev):
                return "Active (P2P/NVLink capable)"
        return "Internal Link Only"
    except Exception:
        return "Unknown"

def setup_distributed():
    """Initializes the distributed environment using Slurm environment variables."""
    world_size = int(os.environ["WORLD_SIZE"])
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    
    master_addr = os.environ["MASTER_ADDR"]
    master_port = os.environ["MASTER_PORT"]
    
    if rank == 0:
        print("=" * 80)
        print(" BEYOND FLOPS - HPC CLUSTER INTERACTION MONITOR")
        print("=" * 80)
        print(f"--> Initializing DDP on Master Node : {master_addr}")
        print(f"--> Dynamic Coordination Port     : {master_port}")
        print(f"--> Target World Cluster Size      : {world_size} GPUs")
        print("-" * 80)
        sys.stdout.flush()

    # Initialize the process group with NCCL backend
    dist.init_process_group(
        backend="nccl",
        init_method=f"tcp://{master_addr}:{master_port}",
        world_size=world_size,
        rank=rank
    )
    
    torch.cuda.set_device(local_rank)
    return rank, local_rank, world_size

def cleanup():
    dist.destroy_process_group()

def run_test():
    rank, local_rank, world_size = setup_distributed()
    
    # --- Local Metadata Gathering ---
    nodename = os.environ.get('SLURMD_NODENAME', socket.gethostname())
    gpu_name = torch.cuda.get_device_name(local_rank)
    gpu_mem = f"{torch.cuda.get_device_properties(local_rank).total_memory / (1024**3):.1f} GB"
    
    # Interrogate environmental network tuning mapping
    ib_env = os.environ.get('NCCL_SOCKET_IFNAME', 'Not Explicitly Set')
    nccl_debug_level = os.environ.get('NCCL_DEBUG', 'None')
    
    # Determine CPU core pinning slice assigned to this process task
    try:
        cpu_affinity = len(os.sched_getaffinity(0))
        cpu_slice = f"{cpu_affinity} Cores Allocated"
    except AttributeError:
        cpu_slice = "Unknown Affinity"

    nvlink_info = get_nvlink_status(local_rank)

    # Pack local node info to pass to Rank 0
    local_data = {
        'rank': rank,
        'node': nodename,
        'l_rank': local_rank,
        'gpu': gpu_name,
        'mem': gpu_mem,
        'cores': cpu_slice,
        'nvlink': nvlink_info
    }

    # Gather data from all ranks onto Rank 0
    gather_list = [None] * world_size if rank == 0 else None
    dist.gather_object(local_data, gather_list, dst=0)

    # --- Simple DDP Computational Block ---
    torch.manual_seed(42 + rank)
    model = nn.Linear(10, 10).to(local_rank)
    ddp_model = DDP(model, device_ids=[local_rank])
    
    inputs = torch.randn(20, 10).to(local_rank)
    targets = torch.randn(20, 10).to(local_rank)
    loss_fn = nn.MSELoss()
    
    outputs = ddp_model(inputs)
    loss = loss_fn(outputs, targets)
    loss.backward()
    
    # Synchronize and average losses across the fabric
    loss_tensor = loss.detach().clone()
    dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
    global_avg_loss = loss_tensor.item() / world_size

    dist.barrier()

    # --- Structured Presentation Report Generation ---
    if rank == 0:
        print("\n" + "=" * 80)
        print("                       CLUSTER HARDWARE TOPOLOGY REPORT")
        print("=" * 80)
        print(f"| {'Rank':<4} | {'Node Name':<12} | {'Local ID':<8} | {'GPU Model':<18} | {'VRAM':<8} | {'CPU Core Allocation':<20} |")
        print("-" * 80)
        
        for data in sorted(gather_list, key=lambda x: x['rank']):
            print(f"| {data['rank']:<4} | {data['node']:<12} | {data['l_rank']:<8} | {data['gpu'][:18]:<18} | {data['mem']:<8} | {data['cores']:<20} |")
        
        print("=" * 80)
        print("                     NETWORK INTERCONNECT & FABRIC STATUS")
        print("-" * 80)
        print(f"--> Target Communication Interface (NCCL_SOCKET_IFNAME) : {ib_env}")
        print(f"--> Active Telemetry Tracking Level (NCCL_DEBUG)       : {nccl_debug_level}")
        print(f"--> Inter-GPU Topo Link Verification                 : {gather_list[0]['nvlink']}")
        print("-" * 80)
        print(" SUCCESS: DDP Multi-Node AllReduce Ring Complete!")
        print(f"--> Computed System Verification Convergence Loss    : {global_avg_loss:.6f}")
        print("=" * 80 + "\n")
        sys.stdout.flush()

    cleanup()

if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("CRITICAL: CUDA architecture runtime is missing on this node execution block.")
        sys.exit(1)
    run_test()