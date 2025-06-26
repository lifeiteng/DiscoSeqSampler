"""Sequential sampling implementation."""

from typing import Iterator, Dict, Any
import logging
from lhotse import CutSet
from lhotse.dataset.sampling import SamplingConstraint
from lhotse.dataset.sampling.dynamic import DynamicCutSampler

from ..utils.config import SamplerConfig

logger = logging.getLogger(__name__)


class SequentialSampler:
    """Sequential sampler that processes cuts in order.
    
    This sampler maintains the original order of cuts in the CutSet,
    making it suitable for:
    - Reproducible training runs
    - Evaluation scenarios
    - Debugging and analysis
    """
    
    def __init__(self, config: SamplerConfig, constraint: SamplingConstraint):
        """Initialize sequential sampler.
        
        Args:
            config: Sampler configuration
            constraint: Sampling constraint for batch construction
        """
        self.config = config
        self.constraint = constraint
        self._epoch = 0
    
    def set_epoch(self, epoch: int) -> None:
        """Set current epoch (no-op for sequential sampler)."""
        self._epoch = epoch
    
    def sample_batches(self, cutset: CutSet) -> Iterator[CutSet]:
        """Sample batches sequentially from cutset.
        
        Args:
            cutset: CutSet to sample from
            
        Yields:
            Batches of cuts as CutSets
        """
        if len(cutset) == 0:
            return
        
        logger.debug(f"Sequential sampling from {len(cutset)} cuts")
        
        # Use Lhotse's DynamicCutSampler for efficient batching
        sampler = DynamicCutSampler(
            cutset,
            max_duration=self.constraint.max_duration,
            max_cuts=self.constraint.max_cuts,
            shuffle=False,  # Sequential = no shuffle
            drop_last=self.config.drop_last,
            quadratic_duration=self.config.quadratic_duration,
        )
        
        batch_count = 0
        for batch in sampler:
            batch_count += 1
            yield batch
        
        logger.debug(f"Sequential sampler produced {batch_count} batches")
    
    def estimate_num_batches(self, cutset: CutSet) -> int:
        """Estimate number of batches for given cutset.
        
        Args:
            cutset: CutSet to estimate for
            
        Returns:
            Estimated number of batches
        """
        if len(cutset) == 0:
            return 0
        
        if self.constraint.max_cuts:
            return len(cutset) // self.constraint.max_cuts
        elif self.constraint.max_duration:
            # Rough estimate based on average duration
            total_duration = sum(cut.duration for cut in cutset)
            return int(total_duration / self.constraint.max_duration)
        else:
            return len(cutset)  # One cut per batch
    
    def state_dict(self) -> Dict[str, Any]:
        """Get sampler state."""
        return {
            "epoch": self._epoch,
            "type": "sequential"
        }
    
    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Load sampler state."""
        self._epoch = state_dict.get("epoch", 0)
