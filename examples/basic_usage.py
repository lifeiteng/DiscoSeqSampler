"""Basic example of using DiscoSeqSampler."""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from lhotse import CutSet
from lhotse.dataset import K2SpeechRecognitionDataset

from discoseqsampler import DiscoSeqSampler, SamplerConfig, SamplingStrategy


def basic_example():
    """Basic usage example with a simple model."""
    
    # Create a dummy cutset for demonstration
    # In practice, you would load this from your data
    print("Creating demo cutset...")
    cuts = CutSet.from_dir("/path/to/your/audio/data")  # Replace with your data path
    
    # Or create from manifests
    # cuts = CutSet.from_manifests(
    #     cuts="/path/to/cuts.jsonl.gz",
    #     features="/path/to/features.jsonl.gz"  # optional
    # )
    
    print(f"Loaded {len(cuts)} cuts")
    
    # Configure the sampler
    config = SamplerConfig(
        strategy=SamplingStrategy.BUCKETED,  # Use bucketed sampling for efficiency
        max_duration=30.0,  # Maximum 30 seconds per batch
        world_size=1,  # Single GPU training
        rank=0,
        shuffle=True,
        seed=42,
        num_buckets=10,
        num_workers=4,
    )
    
    # Create sampler and dataloader
    sampler = DiscoSeqSampler(cuts, config)
    
    # Create dataset transform (optional)
    dataset = K2SpeechRecognitionDataset(cut_transforms=[])  # Add your transforms
    
    # Create dataloader
    dataloader = DataLoader(
        sampler,
        batch_size=None,  # Batch size handled by sampler
        num_workers=config.num_workers,
        pin_memory=True,
        collate_fn=dataset.collate,  # Use dataset's collate function
    )
    
    # Simple model for demonstration
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(80, 256)  # Assuming 80-dim features
            
        def forward(self, x):
            return self.linear(x)
    
    model = SimpleModel()
    optimizer = torch.optim.Adam(model.parameters())
    
    # Training loop
    print("Starting training...")
    for epoch in range(3):
        print(f"\nEpoch {epoch + 1}")
        
        # Set epoch for deterministic sampling
        sampler.set_epoch(epoch)
        
        batch_count = 0
        for batch in dataloader:
            # Your training code here
            # batch is a CutSet containing the cuts for this batch
            
            batch_count += 1
            if batch_count % 10 == 0:
                print(f"  Processed {batch_count} batches...")
            
            # Example: get features and targets
            # features = batch.load_features()  # Shape: [B, T, F]
            # supervisions = batch.supervisions
            
            # Forward pass, loss computation, backprop would go here
            # loss = model(features)
            # optimizer.zero_grad()
            # loss.backward()
            # optimizer.step()
        
        print(f"  Completed epoch with {batch_count} batches")


if __name__ == "__main__":
    basic_example()
