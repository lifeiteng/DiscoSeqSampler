"""Command line interface for DiscoSeqSampler."""

import click
import logging
import json
from pathlib import Path
from typing import Optional

from lhotse import CutSet
from .core.sampler import DiscoSeqSampler, create_dataloader
from .utils.config import SamplerConfig, SamplingStrategy

logger = logging.getLogger(__name__)


@click.group()
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose logging")
def main(verbose: bool) -> None:
    """DiscoSeqSampler CLI - Distributed Coordinated Sequenced Sampler."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )


@main.command()
@click.argument("cutset_path", type=click.Path(exists=True))
@click.option("--strategy", type=click.Choice(["sequential", "bucketed"]), default="sequential")
@click.option("--max-duration", type=float, help="Maximum batch duration in seconds")
@click.option("--max-cuts", type=int, help="Maximum number of cuts per batch")
@click.option("--world-size", type=int, default=1, help="Number of distributed workers")
@click.option("--rank", type=int, default=0, help="Current worker rank")
@click.option("--num-buckets", type=int, default=10, help="Number of buckets for bucketed sampling")
@click.option("--shuffle", is_flag=True, help="Shuffle data")
@click.option("--seed", type=int, help="Random seed")
@click.option("--output", "-o", type=click.Path(), help="Output statistics to file")
def analyze(
    cutset_path: str,
    strategy: str,
    max_duration: Optional[float],
    max_cuts: Optional[int],
    world_size: int,
    rank: int,
    num_buckets: int,
    shuffle: bool,
    seed: Optional[int],
    output: Optional[str]
) -> None:
    """Analyze sampling behavior for a given cutset."""
    
    # Load cutset
    click.echo(f"Loading cutset from {cutset_path}")
    cutset = CutSet.from_file(cutset_path)
    click.echo(f"Loaded {len(cutset)} cuts")
    
    # Create config
    config = SamplerConfig(
        strategy=SamplingStrategy(strategy),
        max_duration=max_duration,
        max_cuts=max_cuts,
        world_size=world_size,
        rank=rank,
        num_buckets=num_buckets,
        shuffle=shuffle,
        seed=seed,
    )
    
    # Create sampler
    sampler = DiscoSeqSampler(cutset, config)
    
    # Collect statistics
    stats = {
        "cutset_size": len(cutset),
        "strategy": strategy,
        "config": config.to_dict(),
        "estimated_batches": len(sampler),
    }
    
    # Sample a few batches to get actual statistics
    batch_sizes = []
    batch_durations = []
    
    click.echo("Sampling batches for analysis...")
    for i, batch in enumerate(sampler):
        if i >= 10:  # Only sample first 10 batches
            break
        
        batch_sizes.append(len(batch))
        batch_durations.append(sum(cut.duration for cut in batch))
    
    if batch_sizes:
        stats.update({
            "sample_batch_sizes": batch_sizes,
            "sample_batch_durations": batch_durations,
            "avg_batch_size": sum(batch_sizes) / len(batch_sizes),
            "avg_batch_duration": sum(batch_durations) / len(batch_durations),
        })
    
    # Add bucketing stats if using bucketed strategy
    if strategy == "bucketed":
        bucket_stats = sampler.sampler.get_bucket_stats(cutset)
        stats["bucket_stats"] = bucket_stats
    
    # Output results
    click.echo("\nSampling Statistics:")
    click.echo(f"  Strategy: {stats['strategy']}")
    click.echo(f"  Total cuts: {stats['cutset_size']:,}")
    click.echo(f"  Estimated batches: {stats['estimated_batches']:,}")
    
    if batch_sizes:
        click.echo(f"  Average batch size: {stats['avg_batch_size']:.1f} cuts")
        click.echo(f"  Average batch duration: {stats['avg_batch_duration']:.1f}s")
    
    if output:
        click.echo(f"\nWriting detailed statistics to {output}")
        with open(output, "w") as f:
            json.dump(stats, f, indent=2, default=str)


@main.command()
@click.argument("cutset_path", type=click.Path(exists=True))
@click.argument("config_path", type=click.Path(exists=True))
@click.option("--epochs", type=int, default=1, help="Number of epochs to simulate")
@click.option("--dry-run", is_flag=True, help="Don't actually process batches")
def benchmark(
    cutset_path: str,
    config_path: str,
    epochs: int,
    dry_run: bool
) -> None:
    """Benchmark sampling performance."""
    
    # Load cutset and config
    cutset = CutSet.from_file(cutset_path)
    
    with open(config_path) as f:
        config_dict = json.load(f)
    config = SamplerConfig.from_dict(config_dict)
    
    click.echo(f"Benchmarking with {len(cutset)} cuts for {epochs} epochs")
    click.echo(f"Strategy: {config.strategy.value}")
    
    # Create dataloader
    dataloader = create_dataloader(cutset, config)
    
    import time
    total_batches = 0
    total_time = 0
    
    for epoch in range(epochs):
        click.echo(f"\nEpoch {epoch + 1}/{epochs}")
        
        # Set epoch for deterministic sampling
        dataloader.dataset.set_epoch(epoch)
        
        epoch_start = time.time()
        batch_count = 0
        
        for batch in dataloader:
            batch_count += 1
            if not dry_run:
                # Simulate some processing time
                time.sleep(0.001)
            
            if batch_count % 100 == 0:
                click.echo(f"  Processed {batch_count} batches...")
        
        epoch_time = time.time() - epoch_start
        total_batches += batch_count
        total_time += epoch_time
        
        click.echo(f"  Epoch completed: {batch_count} batches in {epoch_time:.2f}s")
        click.echo(f"  Throughput: {batch_count / epoch_time:.1f} batches/sec")
    
    click.echo(f"\nBenchmark Summary:")
    click.echo(f"  Total batches: {total_batches:,}")
    click.echo(f"  Total time: {total_time:.2f}s")
    click.echo(f"  Average throughput: {total_batches / total_time:.1f} batches/sec")


if __name__ == "__main__":
    main()
