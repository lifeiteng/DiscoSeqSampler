"""Main DiscoSeqSampler implementation."""

from typing import Iterator, Optional, Union, Any, Dict
import logging
from torch.utils.data import DataLoader, IterableDataset
from lhotse import CutSet
from lhotse.dataset import make_worker_init_fn
from lhotse.dataset.sampling import SamplingConstraint

from ..utils.config import SamplerConfig, SamplingStrategy
from ..core.coordinator import SamplingCoordinator
from ..samplers.sequential import SequentialSampler
from ..samplers.bucketed import BucketedSampler

logger = logging.getLogger(__name__)


class DiscoSeqSampler(IterableDataset):
    """Distributed Coordinated Sequenced Sampler.
    
    This is the main class that implements distributed coordinated sampling
    for speech data using Lhotse. It provides:
    
    - Multiple sampling strategies (sequential, bucketed, random, balanced)
    - Distributed training support with coordination
    - Dynamic batch sizing based on duration or cut count
    - Efficient buffering and prefetching
    - Fault tolerance and state management
    
    Example:
        >>> from discoseqsampler import DiscoSeqSampler, SamplerConfig
        >>> from lhotse import CutSet
        >>> 
        >>> # Load your cutset
        >>> cuts = CutSet.from_manifests(...)
        >>> 
        >>> # Configure sampler
        >>> config = SamplerConfig(
        ...     strategy=SamplingStrategy.BUCKETED,
        ...     max_duration=30.0,
        ...     world_size=4,
        ...     rank=0
        ... )
        >>> 
        >>> # Create sampler
        >>> sampler = DiscoSeqSampler(cuts, config)
        >>> 
        >>> # Use with DataLoader
        >>> dataloader = DataLoader(
        ...     sampler,
        ...     batch_size=None,  # Handled by sampler
        ...     num_workers=config.num_workers
        ... )
        >>> 
        >>> for batch in dataloader:
        ...     # Your training code here
        ...     pass
    """
    
    def __init__(
        self, 
        cutset: CutSet, 
        config: SamplerConfig,
        constraint: Optional[SamplingConstraint] = None
    ):
        """Initialize DiscoSeqSampler.
        
        Args:
            cutset: The CutSet to sample from
            config: Sampling configuration
            constraint: Optional sampling constraint for batch construction
        """
        super().__init__()
        
        self.cutset = cutset
        self.config = config
        self.constraint = constraint or self._create_default_constraint()
        
        # Initialize coordinator for distributed sampling
        self.coordinator = SamplingCoordinator(config)
        
        # Initialize the appropriate sampler based on strategy
        self.sampler = self._create_sampler()
        
        # State management
        self._epoch = 0
        self._exhausted = False
        
        logger.info(f"DiscoSeqSampler initialized with {len(cutset)} cuts")
        logger.info(f"Strategy: {config.strategy.value}")
        logger.info(f"Distributed: {config.world_size} workers (rank {config.rank})")
    
    def _create_default_constraint(self) -> SamplingConstraint:
        """Create default sampling constraint from config."""
        if self.config.max_duration is not None:
            return SamplingConstraint(max_duration=self.config.max_duration)
        elif self.config.max_cuts is not None:
            return SamplingConstraint(max_cuts=self.config.max_cuts)
        else:
            # Default to reasonable batch size
            return SamplingConstraint(max_cuts=self.config.batch_size or 32)
    
    def _create_sampler(self):
        """Create the appropriate sampler based on strategy."""
        if self.config.strategy == SamplingStrategy.SEQUENTIAL:
            return SequentialSampler(self.config, self.constraint)
        elif self.config.strategy == SamplingStrategy.BUCKETED:
            return BucketedSampler(self.config, self.constraint)
        else:
            raise NotImplementedError(f"Strategy {self.config.strategy} not implemented yet")
    
    def set_epoch(self, epoch: int) -> None:
        """Set the current epoch for deterministic sampling.
        
        Args:
            epoch: Current training epoch
        """
        self._epoch = epoch
        self.coordinator.set_epoch(epoch)
        self.sampler.set_epoch(epoch)
        self._exhausted = False
        
        logger.debug(f"DiscoSeqSampler set to epoch {epoch}")
    
    def __iter__(self) -> Iterator[CutSet]:
        """Iterate over batches of cuts."""
        # Get worker-specific cutset
        worker_cutset = self.coordinator.get_worker_cutset(self.cutset)
        
        if len(worker_cutset) == 0:
            logger.warning(f"Worker {self.config.rank} got empty cutset")
            return
        
        # Initialize sampler with worker cutset
        batch_iterator = self.sampler.sample_batches(worker_cutset)
        
        batch_count = 0
        try:
            for batch in batch_iterator:
                yield batch
                batch_count += 1
                
        except StopIteration:
            pass
        
        # Mark as exhausted and coordinate with other workers
        self._exhausted = True
        
        # Synchronize workers before epoch ends
        self.coordinator.synchronize_workers()
        
        logger.debug(f"Worker {self.config.rank} produced {batch_count} batches in epoch {self._epoch}")
    
    def __len__(self) -> int:
        """Estimate number of batches per worker per epoch."""
        worker_cutset = self.coordinator.get_worker_cutset(self.cutset)
        return self.sampler.estimate_num_batches(worker_cutset)
    
    def state_dict(self) -> Dict[str, Any]:
        """Get sampler state for checkpointing."""
        return {
            "config": self.config.to_dict(),
            "epoch": self._epoch,
            "coordinator_state": self.coordinator.get_state(),
            "sampler_state": self.sampler.state_dict() if hasattr(self.sampler, 'state_dict') else {},
        }
    
    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Load sampler state from checkpoint."""
        self._epoch = state_dict.get("epoch", 0)
        
        if "coordinator_state" in state_dict:
            self.coordinator.load_state(state_dict["coordinator_state"])
        
        if "sampler_state" in state_dict and hasattr(self.sampler, 'load_state_dict'):
            self.sampler.load_state_dict(state_dict["sampler_state"])
        
        logger.info(f"Loaded DiscoSeqSampler state for epoch {self._epoch}")


def create_dataloader(
    cutset: CutSet,
    config: SamplerConfig,
    constraint: Optional[SamplingConstraint] = None,
    **dataloader_kwargs
) -> DataLoader:
    """Create a DataLoader with DiscoSeqSampler.
    
    Args:
        cutset: The CutSet to sample from
        config: Sampling configuration
        constraint: Optional sampling constraint
        **dataloader_kwargs: Additional arguments for DataLoader
        
    Returns:
        Configured DataLoader
    """
    sampler = DiscoSeqSampler(cutset, config, constraint)
    
    # Set default DataLoader arguments
    default_kwargs = {
        "batch_size": None,  # Handled by sampler
        "num_workers": config.num_workers,
        "pin_memory": config.pin_memory,
        "prefetch_factor": config.prefetch_factor if config.num_workers > 0 else 2,
        "worker_init_fn": make_worker_init_fn(rank=config.rank, world_size=config.world_size),
        "persistent_workers": config.num_workers > 0,
    }
    
    # Update with user-provided kwargs
    default_kwargs.update(dataloader_kwargs)
    
    return DataLoader(sampler, **default_kwargs)
