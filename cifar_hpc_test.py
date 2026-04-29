import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torchvision import datasets, transforms, models

def train():
    # 1. Initialize Distributed Environment
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(local_rank)

    # 2. Data Preparation
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])

    # Download only on rank 0 to avoid race conditions
    dataset = datasets.CIFAR100(root='./data', train=True, download=(rank == 0), transform=transform)
    
    # Sampler ensures each GPU gets different data
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    train_loader = torch.utils.data.DataLoader(
        dataset, batch_size=128, sampler=sampler, num_workers=4, pin_memory=True
    )

    # 3. Model, Optimizer, and Loss
    # Using ResNet18 for CIFAR to keep throughput high
    model = models.resnet18(num_classes=100).cuda(local_rank)
    model = DDP(model, device_ids=[local_rank])
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1 * (world_size / 8), momentum=0.9)

    # 4. Training Loop (Set epochs to control duration)
    model.train()
    epochs = 10 
    
    if rank == 0:
        print(f"Starting training for {epochs} epochs across {world_size} GPUs...")

    for epoch in range(epochs):
        sampler.set_epoch(epoch)
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.cuda(local_rank), target.cuda(local_rank)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            if rank == 0 and batch_idx % 20 == 0:
                print(f"Epoch: {epoch} | Batch: {batch_idx}/{len(train_loader)} | Loss: {loss.item():.4f}")

    if rank == 0:
        print("Test Complete. Cluster verified.")

if __name__ == "__main__":
    train()