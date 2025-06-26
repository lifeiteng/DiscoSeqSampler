"""Distributed sampling coordinator for DiscoSeqSampler."""

from typing import Optional, Dict, Any, Iterator
import torch
import torch.distributed as dist
from lhotse import CutSet
import logging
from ..utils.config import SamplerConfig

logger = logging.getLogger(__name__)


class SamplingCoordinator:
    """Coordinates sampling across distributed workers.
    
    This class handles synchronization and coordination of sampling
    across multiple distributed workers to ensure:
    - Each sample is seen exactly once per epoch
    - Load balancing across workers
    - Deterministic ordering when needed
    - Fault tolerance and recovery
    """
    
    def __init__(self, config: SamplerConfig):
        """Initialize the sampling coordinator.
        
        Args:
            config: Sampler configuration
        """
        self.config = config
        self.world_size = config.world_size
        self.rank = config.rank
        
        # State tracking
        self._epoch = 0
        self._step = 0
        self._synchronized = False
        
        # Initialize distributed if needed
        if self.world_size > 1 and not dist.is_initialized():
            logger.warning("Distributed not initialized but world_size > 1")
    
    def set_epoch(self, epoch: int) -> None:
        """Set the current epoch for deterministic sampling.
        
        Args:
            epoch: Current training epoch
        """
        self._epoch = epoch
        self._step = 0
        
        if self.world_size > 1 and dist.is_initialized():
            # Synchronize epoch across all workers
            epoch_tensor = torch.tensor(epoch, dtype=torch.long)
            dist.broadcast(epoch_tensor, src=0)
            self._epoch = epoch_tensor.item()
            
        logger.debug(f"Coordinator set to epoch {self._epoch}")
    
    def get_worker_cutset(self, cutset: CutSet) -> CutSet:
        """Get the cutset subset for current worker.
        
        Args:
            cutset: Full cutset to split
            
        Returns:
            CutSet subset for current worker
        """
        if self.world_size == 1:
            return cutset
        
        # Create deterministic shuffle based on epoch and seed
        if self.config.shuffle:
            seed = (self.config.seed or 0) + self._epoch
            cutset = cutset.shuffle(rng=torch.Generator().manual_seed(seed))
        
        # Split cutset across workers
        total_cuts = len(cutset)
        cuts_per_worker = total_cuts // self.world_size
        
        start_idx = self.rank * cuts_per_worker
        if self.rank == self.world_size - 1:
            # Last worker gets remaining cuts
            end_idx = total_cuts
        else:
            end_idx = start_idx + cuts_per_worker
        
        worker_cutset = cutset.subset(first=start_idx, last=end_idx)
        
        logger.debug(
            f"Worker {self.rank}/{self.world_size} got {len(worker_cutset)} cuts "
            f"(indices {start_idx}:{end_idx})"
        )
        
        return worker_cutset
    
    def synchronize_workers(self) -> None:
        """Synchronize all workers at a barrier."""
        if self.world_size > 1 and dist.is_initialized():
            dist.barrier()
            self._synchronized = True
            logger.debug(f"Worker {self.rank} synchronized")
    
    def all_gather_info(self, local_info: Dict[str, Any]) -> Dict[int, Dict[str, Any]]:
        """Gather information from all workers.
        
        Args:
            local_info: Local worker information
            
        Returns:
            Dictionary mapping rank to worker info
        """
        if self.world_size == 1:
            return {0: local_info}
        
        if not dist.is_initialized():
            logger.warning("Cannot gather info: distributed not initialized")
            return {self.rank: local_info}
        
        # Convert info to tensor for all_gather
        # This is a simplified version - in practice you'd need proper serialization
        gathered_info = {}
        
        try:
            # Gather batch counts as an example
            if "batch_count" in local_info:
                batch_counts = [torch.tensor(0, dtype=torch.long) for _ in range(self.world_size)]
                dist.all_gather(batch_counts, torch.tensor(local_info["batch_count"], dtype=torch.long))
                
                for rank, count in enumerate(batch_counts):
                    gathered_info[rank] = {"batch_count": count.item()}
        
        except Exception as e:
            logger.warning(f"Failed to gather info: {e}")
            gathered_info[self.rank] = local_info
        
        return gathered_info
    
    def should_stop_epoch(self, local_exhausted: bool) -> bool:
        """Determine if epoch should stop based on worker states.
        
        Args:
            local_exhausted: Whether current worker has exhausted its data
            
        Returns:
            Whether to stop the current epoch
        """
        if self.world_size == 1:
            return local_exhausted
        
        if not dist.is_initialized():
            return local_exhausted
        
        # Check if all workers are exhausted
        exhausted_tensor = torch.tensor(int(local_exhausted), dtype=torch.int)
        
        try:
            dist.all_reduce(exhausted_tensor, op=dist.ReduceOp.SUM)
            all_exhausted = exhausted_tensor.item() == self.world_size
            
            logger.debug(
                f"Worker {self.rank}: local_exhausted={local_exhausted}, "
                f"total_exhausted={exhausted_tensor.item()}/{self.world_size}"
            )
            
            return all_exhausted
            
        except Exception as e:
            logger.warning(f"Failed to check epoch completion: {e}")
            return local_exhausted
    
    def get_state(self) -> Dict[str, Any]:
        """Get coordinator state for checkpointing."""
        return {
            "epoch": self._epoch,
            "step": self._step,
            "rank": self.rank,
            "world_size": self.world_size,
        }
    
    def load_state(self, state: Dict[str, Any]) -> None:
        """Load coordinator state from checkpoint."""
        self._epoch = state.get("epoch", 0)
        self._step = state.get("step", 0)
        
        # Validate consistency
        if state.get("world_size") != self.world_size:
            logger.warning(
                f"Loaded world_size {state.get('world_size')} != current {self.world_size}"
            )
        
        logger.info(f"Loaded coordinator state: epoch={self._epoch}, step={self._step}")
