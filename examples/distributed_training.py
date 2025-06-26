"""Distributed training example with DiscoSeqSampler."""

import os
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

from lhotse import CutSet
from lhotse.dataset import K2SpeechRecognitionDataset

from discoseqsampler import DiscoSeqSampler, SamplerConfig, SamplingStrategy


def setup_distributed(rank, world_size):
    """Initialize distributed training."""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    
    # Initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup_distributed():
    """Clean up distributed training."""
    dist.destroy_process_group()


def train_worker(rank, world_size, cutset_path):
    """Training function for each worker."""
    print(f"Worker {rank}/{world_size} starting...")
    
    # Setup distributed training
    setup_distributed(rank, world_size)
    
    # Load data
    cuts = CutSet.from_file(cutset_path)
    print(f"Worker {rank}: Loaded {len(cuts)} cuts")
    
    # Configure sampler for distributed training
    config = SamplerConfig(
        strategy=SamplingStrategy.BUCKETED,
        max_duration=20.0,  # Smaller batches for distributed training
        world_size=world_size,
        rank=rank,
        shuffle=True,
        seed=42,
        num_buckets=20,
        num_workers=2,  # Fewer workers per GPU
        drop_last=True,  # Important for distributed training
    )
    
    # Create sampler and dataloader
    sampler = DiscoSeqSampler(cuts, config)
    
    dataset = K2SpeechRecognitionDataset()
    dataloader = DataLoader(
        sampler,
        batch_size=None,
        num_workers=config.num_workers,
        pin_memory=True,
        collate_fn=dataset.collate,
    )
    
    # Model
    class DistributedModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Linear(80, 512),
                nn.ReLU(),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Linear(256, 1000),  # Vocab size
            )
            
        def forward(self, x):
            return self.encoder(x)
    
    # Create model and wrap with DDP
    model = DistributedModel().cuda(rank)
    model = DDP(model, device_ids=[rank])
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    # Training loop
    for epoch in range(5):
        print(f"Worker {rank}: Epoch {epoch + 1}")
        
        # Important: set epoch for deterministic distributed sampling
        sampler.set_epoch(epoch)
        
        model.train()
        total_loss = 0
        batch_count = 0
        
        for batch in dataloader:
            # Simulate getting features and targets
            # In practice, you'd extract these from the batch
            batch_size = len(batch)
            fake_features = torch.randn(batch_size, 100, 80).cuda(rank)
            fake_targets = torch.randint(0, 1000, (batch_size, 100)).cuda(rank)
            
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(fake_features)
            outputs = outputs.view(-1, outputs.size(-1))
            targets = fake_targets.view(-1)
            
            loss = criterion(outputs, targets)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            batch_count += 1
            
            if batch_count % 20 == 0:
                print(f"  Worker {rank}: Batch {batch_count}, Loss: {loss.item():.4f}")
        
        avg_loss = total_loss / batch_count if batch_count > 0 else 0
        print(f"Worker {rank}: Epoch {epoch + 1} completed, Avg Loss: {avg_loss:.4f}")
        
        # Synchronize workers at epoch end
        dist.barrier()
    
    print(f"Worker {rank}: Training completed")
    cleanup_distributed()


def main():
    """Main function for distributed training."""
    world_size = torch.cuda.device_count()
    cutset_path = "/path/to/your/cutset.jsonl.gz"  # Replace with your data
    
    if world_size < 2:
        print("This example requires at least 2 GPUs")
        print("For single GPU training, use basic_usage.py")
        return
    
    print(f"Starting distributed training with {world_size} GPUs")
    
    # Spawn processes for each GPU
    mp.spawn(
        train_worker,
        args=(world_size, cutset_path),
        nprocs=world_size,
        join=True
    )


if __name__ == "__main__":
    main()
