"""Advanced usage examples for DiscoSeqSampler."""

import torch
import json
from pathlib import Path
from lhotse import CutSet, load_manifest
from lhotse.dataset import K2SpeechRecognitionDataset
from lhotse.dataset.sampling import SamplingConstraint

from discoseqsampler import (
    DiscoSeqSampler, 
    SamplerConfig, 
    SamplingStrategy,
    create_dataloader
)


def custom_constraint_example():
    """Example with custom sampling constraints."""
    print("=== Custom Constraint Example ===")
    
    # Load your cutset
    cuts = CutSet.from_file("path/to/your/cuts.jsonl.gz")
    
    # Create custom constraint
    constraint = SamplingConstraint(
        max_duration=60.0,  # Max 60 seconds per batch
        max_cuts=16,        # Max 16 utterances per batch
        max_frames=300000,  # Max frames (for memory control)
    )
    
    config = SamplerConfig(
        strategy=SamplingStrategy.BUCKETED,
        world_size=1,
        rank=0,
        shuffle=True,
        num_buckets=15,
        quadratic_duration=True,  # Better memory estimation
    )
    
    sampler = DiscoSeqSampler(cuts, config, constraint)
    
    print(f"Estimated batches per epoch: {len(sampler)}")
    
    # Analyze first few batches
    for i, batch in enumerate(sampler):
        if i >= 5:
            break
        
        total_duration = sum(cut.duration for cut in batch)
        total_frames = sum(cut.num_frames for cut in batch)
        
        print(f"Batch {i+1}:")
        print(f"  Cuts: {len(batch)}")
        print(f"  Duration: {total_duration:.2f}s")
        print(f"  Frames: {total_frames:,}")


def bucket_analysis_example():
    """Example showing bucket analysis for bucketed sampling."""
    print("\n=== Bucket Analysis Example ===")
    
    cuts = CutSet.from_file("path/to/your/cuts.jsonl.gz")
    
    config = SamplerConfig(
        strategy=SamplingStrategy.BUCKETED,
        max_duration=30.0,
        num_buckets=20,
        bucket_method="duration",  # Can be "duration", "num_frames", "num_features"
    )
    
    sampler = DiscoSeqSampler(cuts, config)
    
    # Get bucket statistics
    bucket_stats = sampler.sampler.get_bucket_stats(cuts)
    
    print("Bucket Statistics:")
    print(f"  Method: {bucket_stats['method']}")
    print(f"  Number of buckets: {bucket_stats['num_buckets']}")
    print(f"  Value range: {bucket_stats['min_value']:.2f} - {bucket_stats['max_value']:.2f}")
    print(f"  Mean value: {bucket_stats['mean_value']:.2f}")
    
    print("\nBucket boundaries:")
    for i, (start, end) in enumerate(bucket_stats['boundaries']):
        print(f"  Bucket {i+1:2d}: {start:6.2f} - {end:6.2f}")


def checkpoint_example():
    """Example showing how to save and load sampler state."""
    print("\n=== Checkpoint Example ===")
    
    cuts = CutSet.from_file("path/to/your/cuts.jsonl.gz")
    
    config = SamplerConfig(
        strategy=SamplingStrategy.SEQUENTIAL,
        max_duration=25.0,
        shuffle=True,
        seed=12345,
    )
    
    sampler = DiscoSeqSampler(cuts, config)
    
    # Simulate training for a few epochs
    for epoch in range(3):
        sampler.set_epoch(epoch)
        
        # Process a few batches
        for i, batch in enumerate(sampler):
            if i >= 10:  # Process only first 10 batches
                break
        
        # Save checkpoint
        checkpoint = {
            "epoch": epoch,
            "sampler_state": sampler.state_dict(),
            "model_state": {},  # Your model state would go here
        }
        
        checkpoint_path = f"checkpoint_epoch_{epoch}.pt"
        torch.save(checkpoint, checkpoint_path)
        print(f"Saved checkpoint for epoch {epoch}")
    
    # Load checkpoint
    checkpoint = torch.load("checkpoint_epoch_2.pt")
    new_sampler = DiscoSeqSampler(cuts, config)
    new_sampler.load_state_dict(checkpoint["sampler_state"])
    
    print("Loaded checkpoint successfully")


def performance_comparison():
    """Compare performance of different sampling strategies."""
    print("\n=== Performance Comparison ===")
    
    cuts = CutSet.from_file("path/to/your/cuts.jsonl.gz")
    
    strategies = [
        (SamplingStrategy.SEQUENTIAL, "Sequential"),
        (SamplingStrategy.BUCKETED, "Bucketed (10 buckets)"),
    ]
    
    import time
    
    for strategy, name in strategies:
        print(f"\nTesting {name}:")
        
        config = SamplerConfig(
            strategy=strategy,
            max_duration=20.0,
            shuffle=True,
            num_buckets=10 if strategy == SamplingStrategy.BUCKETED else None,
        )
        
        sampler = DiscoSeqSampler(cuts, config)
        
        start_time = time.time()
        batch_count = 0
        total_cuts = 0
        
        for batch in sampler:
            batch_count += 1
            total_cuts += len(batch)
            
            if batch_count >= 100:  # Test first 100 batches
                break
        
        elapsed = time.time() - start_time
        
        print(f"  Batches: {batch_count}")
        print(f"  Total cuts: {total_cuts}")
        print(f"  Time: {elapsed:.2f}s")
        print(f"  Throughput: {batch_count/elapsed:.1f} batches/sec")


def multi_dataset_example():
    """Example with multiple datasets and mixed sampling."""
    print("\n=== Multi-Dataset Example ===")
    
    # Load multiple datasets
    train_cuts = CutSet.from_file("path/to/train_cuts.jsonl.gz")
    dev_cuts = CutSet.from_file("path/to/dev_cuts.jsonl.gz")
    
    # Combine with different weights
    combined_cuts = train_cuts + dev_cuts
    
    # Or sample from each dataset separately
    train_config = SamplerConfig(
        strategy=SamplingStrategy.BUCKETED,
        max_duration=30.0,
        shuffle=True,
        num_buckets=15,
    )
    
    dev_config = SamplerConfig(
        strategy=SamplingStrategy.SEQUENTIAL,  # Sequential for evaluation
        max_duration=60.0,  # Larger batches for evaluation
        shuffle=False,
    )
    
    train_sampler = DiscoSeqSampler(train_cuts, train_config)
    dev_sampler = DiscoSeqSampler(dev_cuts, dev_config)
    
    print(f"Train batches per epoch: {len(train_sampler)}")
    print(f"Dev batches: {len(dev_sampler)}")
    
    # Training loop with evaluation
    for epoch in range(2):
        print(f"\nEpoch {epoch + 1}")
        
        # Training
        train_sampler.set_epoch(epoch)
        train_batches = 0
        for batch in train_sampler:
            train_batches += 1
            if train_batches >= 50:  # Simulate training
                break
        
        print(f"  Training: {train_batches} batches")
        
        # Evaluation
        dev_batches = 0
        for batch in dev_sampler:
            dev_batches += 1
            if dev_batches >= 10:  # Simulate evaluation
                break
        
        print(f"  Evaluation: {dev_batches} batches")


def save_config_example():
    """Example of saving and loading configurations."""
    print("\n=== Configuration Management ===")
    
    # Create configuration
    config = SamplerConfig(
        strategy=SamplingStrategy.BUCKETED,
        max_duration=25.0,
        world_size=4,
        rank=0,
        shuffle=True,
        seed=42,
        num_buckets=12,
        bucket_method="duration",
        num_workers=8,
        quadratic_duration=True,
        extra_config={
            "experiment_name": "speech_recognition_v1",
            "model_type": "transformer",
            "custom_param": 123,
        }
    )
    
    # Save configuration
    config_dict = config.to_dict()
    with open("sampler_config.json", "w") as f:
        json.dump(config_dict, f, indent=2)
    
    print("Saved configuration to sampler_config.json")
    
    # Load configuration
    with open("sampler_config.json", "r") as f:
        loaded_config_dict = json.load(f)
    
    loaded_config = SamplerConfig.from_dict(loaded_config_dict)
    
    print("Loaded configuration:")
    print(f"  Strategy: {loaded_config.strategy.value}")
    print(f"  Max duration: {loaded_config.max_duration}")
    print(f"  Extra config: {loaded_config.extra_config}")


if __name__ == "__main__":
    print("Advanced DiscoSeqSampler Examples")
    print("=" * 40)
    
    # Note: You'll need to replace the file paths with your actual data
    print("Please update the file paths in the examples to point to your actual data.")
    
    # Uncomment the examples you want to run:
    # custom_constraint_example()
    # bucket_analysis_example()
    # checkpoint_example()
    # performance_comparison()
    # multi_dataset_example()
    # save_config_example()
