"""Utility functions for DiscoSeqSampler."""

from typing import Union, List, Optional
import torch
import torch.distributed as dist
from lhotse import CutSet
import logging

logger = logging.getLogger(__name__)


def setup_distributed() -> tuple[int, int]:
    """Setup distributed training environment.
    
    Returns:
        Tuple of (world_size, rank)
    """
    if dist.is_available() and dist.is_initialized():
        world_size = dist.get_world_size()
        rank = dist.get_rank()
    else:
        world_size = 1
        rank = 0
    
    return world_size, rank


def get_device() -> torch.device:
    """Get the appropriate device for current process."""
    if torch.cuda.is_available():
        if dist.is_available() and dist.is_initialized():
            local_rank = dist.get_rank() % torch.cuda.device_count()
            return torch.device(f"cuda:{local_rank}")
        else:
            return torch.device("cuda:0")
    else:
        return torch.device("cpu")


def split_cutset_by_rank(
    cutset: CutSet, 
    world_size: int, 
    rank: int, 
    shuffle: bool = False
) -> CutSet:
    """Split a CutSet across distributed workers.
    
    Args:
        cutset: Input CutSet to split
        world_size: Number of distributed workers
        rank: Current worker rank
        shuffle: Whether to shuffle before splitting
        
    Returns:
        CutSet subset for current worker
    """
    if shuffle:
        cutset = cutset.shuffle()
    
    # Split the cutset into chunks for each worker
    cuts_per_worker = len(cutset) // world_size
    start_idx = rank * cuts_per_worker
    
    if rank == world_size - 1:
        # Last worker gets remaining cuts
        end_idx = len(cutset)
    else:
        end_idx = start_idx + cuts_per_worker
    
    return cutset.subset(first=start_idx, last=end_idx)


def estimate_batch_duration(cuts: List, quadratic: bool = False) -> float:
    """Estimate the duration of a batch of cuts.
    
    Args:
        cuts: List of cuts
        quadratic: Whether to use quadratic duration estimation
        
    Returns:
        Estimated batch duration in seconds
    """
    if not cuts:
        return 0.0
    
    if quadratic:
        # Use quadratic duration estimation for better memory prediction
        max_duration = max(cut.duration for cut in cuts)
        return len(cuts) * max_duration
    else:
        # Simple sum of durations
        return sum(cut.duration for cut in cuts)


def log_sampler_info(
    world_size: int,
    rank: int, 
    cutset_size: int,
    batch_size: Optional[int] = None,
    max_duration: Optional[float] = None
) -> None:
    """Log sampler configuration information."""
    logger.info(f"DiscoSeqSampler initialized:")
    logger.info(f"  World size: {world_size}")
    logger.info(f"  Rank: {rank}")
    logger.info(f"  CutSet size: {cutset_size}")
    if batch_size:
        logger.info(f"  Batch size: {batch_size}")
    if max_duration:
        logger.info(f"  Max duration: {max_duration}s")
