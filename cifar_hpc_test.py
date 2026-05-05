import os
import socket
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torchvision import datasets, transforms, models


def setup():
    dist.init_process_group(backend="nccl")

    local_rank = int(os.environ["LOCAL_RANK"])
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    torch.cuda.set_device(local_rank)
    torch.backends.cudnn.benchmark = True

    return local_rank, rank, world_size


def cleanup():
    dist.destroy_process_group()


def train():
    local_rank, rank, world_size = setup()

    # ----------------------------
    # Debug (optional but useful)
    # ----------------------------
    print(f"Host: {socket.gethostname()} | Rank: {rank} | Local Rank: {local_rank}")

    # ----------------------------
    # Data preparation
    # ----------------------------
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408),
                             (0.2675, 0.2565, 0.2761)),
    ])

    data_root = "./data"  # NFS-backed path, home dir

    # SAFE DOWNLOAD PATTERN
    if rank == 0:
        print("Rank 0 downloading dataset (if needed)...")
        datasets.CIFAR100(root=data_root, train=True, download=True)

    dist.barrier()  # <-- critical synchronization

    dataset = datasets.CIFAR100(
        root=data_root,
        train=True,
        download=False,
        transform=transform
    )

    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True
    )

    train_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=128,
        sampler=sampler,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )

    # ----------------------------
    # Model
    # ----------------------------
    model = models.resnet18(num_classes=100).cuda(local_rank)
    model = DDP(model, device_ids=[local_rank])

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(),
        lr=0.1 * (world_size / 8),  # linear scaling rule
        momentum=0.9
    )

    # ----------------------------
    # Training
    # ----------------------------
    epochs = 10

    if rank == 0:
        print(f"Starting training on {world_size} GPUs")

    model.train()

    for epoch in range(epochs):
        sampler.set_epoch(epoch)

        for batch_idx, (data, target) in enumerate(train_loader):
            data = data.cuda(local_rank, non_blocking=True)
            target = target.cuda(local_rank, non_blocking=True)

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            if rank == 0 and batch_idx % 20 == 0:
                print(
                    f"Epoch {epoch} | Batch {batch_idx}/{len(train_loader)} "
                    f"| Loss: {loss.item():.4f}"
                )

    if rank == 0:
        print("Training complete. Cluster verified.")

    cleanup()


if __name__ == "__main__":
    train()