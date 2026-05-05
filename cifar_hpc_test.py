import os
import socket
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torchvision import datasets, transforms, models


# ----------------------------
# Setup / Cleanup
# ----------------------------
def setup():
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)

    dist.init_process_group(
        backend="nccl",
        device_id=local_rank
    )

    rank = dist.get_rank()
    world_size = dist.get_world_size()

    torch.backends.cudnn.benchmark = True

    return local_rank, rank, world_size


def cleanup():
    dist.destroy_process_group()


# ----------------------------
# Debug
# ----------------------------
def log_rank_info(rank, local_rank):
    print(f"Host: {socket.gethostname()} | Rank: {rank} | Local Rank: {local_rank}")


# ----------------------------
# Data
# ----------------------------
def build_transforms():
    return transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize(
            (0.5071, 0.4867, 0.4408),
            (0.2675, 0.2565, 0.2761)
        ),
    ])


def ensure_dataset(rank, data_root):
    dataset_path = os.path.join(data_root, "cifar-100-python")

    if rank == 0:
        if not os.path.exists(dataset_path):
            print("Dataset not found. Downloading...")
            datasets.CIFAR100(root=data_root, train=True, download=True)
        else:
            print("Dataset found. Skipping download.")

    dist.barrier()


def build_dataloader(rank, world_size, transform, data_root):
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

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=128,
        sampler=sampler,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )

    return dataloader, sampler


# ----------------------------
# Model
# ----------------------------
def build_model(local_rank, world_size):
    model = models.resnet18(num_classes=100).cuda(local_rank)
    model = DDP(model, device_ids=[local_rank])

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.SGD(
        model.parameters(),
        lr=0.1 * (world_size / 8),
        momentum=0.9
    )

    return model, criterion, optimizer


# ----------------------------
# Training
# ----------------------------
def train_one_epoch(model, dataloader, sampler, optimizer, criterion,
                    epoch, local_rank, rank):

    model.train()
    sampler.set_epoch(epoch)

    for batch_idx, (data, target) in enumerate(dataloader):
        data = data.cuda(local_rank, non_blocking=True)
        target = target.cuda(local_rank, non_blocking=True)

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        if rank == 0 and batch_idx % 20 == 0:
            print(
                f"Epoch {epoch} | Batch {batch_idx}/{len(dataloader)} "
                f"| Loss: {loss.item():.4f}"
            )


# ----------------------------
# Main
# ----------------------------
def train():
    local_rank, rank, world_size = setup()

    log_rank_info(rank, local_rank)

    data_root = "./data"
    transform = build_transforms()

    ensure_dataset(rank, data_root)

    dataloader, sampler = build_dataloader(
        rank, world_size, transform, data_root
    )

    model, criterion, optimizer = build_model(local_rank, world_size)

    epochs = 10

    if rank == 0:
        print(f"Starting training on {world_size} GPUs")

    for epoch in range(epochs):
        train_one_epoch(
            model,
            dataloader,
            sampler,
            optimizer,
            criterion,
            epoch,
            local_rank,
            rank
        )

    if rank == 0:
        print("Training complete. Cluster verified.")

    cleanup()


if __name__ == "__main__":
    train()