"""Bucketed sampling implementation."""

from typing import Iterator, Dict, Any, List
import logging
import torch
from lhotse import CutSet
from lhotse.dataset.sampling import SamplingConstraint
from lhotse.dataset.sampling.dynamic import DynamicBucketingSampler

from ..utils.config import SamplerConfig

logger = logging.getLogger(__name__)


class BucketedSampler:
    """Bucketed sampler that groups cuts by similar characteristics.
    
    This sampler groups cuts into buckets based on duration, number of frames,
    or other characteristics to improve training efficiency by:
    - Reducing padding in batches
    - Improving memory utilization  
    - Enabling more consistent batch processing times
    """
    
    def __init__(self, config: SamplerConfig, constraint: SamplingConstraint):
        """Initialize bucketed sampler.
        
        Args:
            config: Sampler configuration
            constraint: Sampling constraint for batch construction
        """
        self.config = config
        self.constraint = constraint
        self._epoch = 0
        
        # Bucketing configuration
        self.num_buckets = config.num_buckets
        self.bucket_method = config.bucket_method
        self.buffer_size = config.buffer_size
    
    def set_epoch(self, epoch: int) -> None:
        """Set current epoch for deterministic shuffling."""
        self._epoch = epoch
    
    def sample_batches(self, cutset: CutSet) -> Iterator[CutSet]:
        """Sample batches using bucketing strategy.
        
        Args:
            cutset: CutSet to sample from
            
        Yields:
            Batches of cuts as CutSets
        """
        if len(cutset) == 0:
            return
        
        logger.debug(f"Bucketed sampling from {len(cutset)} cuts with {self.num_buckets} buckets")
        
        # Shuffle cutset if configured
        if self.config.shuffle:
            seed = (self.config.seed or 0) + self._epoch
            cutset = cutset.shuffle(rng=torch.Generator().manual_seed(seed))
        
        # Use Lhotse's DynamicBucketingSampler
        sampler = DynamicBucketingSampler(
            cutset,
            max_duration=self.constraint.max_duration,
            max_cuts=self.constraint.max_cuts,
            shuffle=self.config.shuffle,
            drop_last=self.config.drop_last,
            quadratic_duration=self.config.quadratic_duration,
            num_buckets=self.num_buckets,
            buffer_size=self.buffer_size,
            bucket_method=self._get_bucket_method(),
        )
        
        batch_count = 0
        for batch in sampler:
            batch_count += 1
            yield batch
        
        logger.debug(f"Bucketed sampler produced {batch_count} batches")
    
    def _get_bucket_method(self):
        """Get the bucket method function."""
        if self.bucket_method == "duration":
            return lambda cut: cut.duration
        elif self.bucket_method == "num_frames":
            return lambda cut: cut.num_frames
        elif self.bucket_method == "num_features":
            return lambda cut: cut.num_features if cut.has_features else cut.num_frames
        elif self.bucket_method == "num_samples":
            return lambda cut: cut.num_samples
        else:
            logger.warning(f"Unknown bucket method '{self.bucket_method}', using duration")
            return lambda cut: cut.duration
    
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
            # For bucketed sampling, estimate is less accurate due to bucketing
            # Use a conservative estimate
            total_duration = sum(cut.duration for cut in cutset)
            estimated_batches = int(total_duration / self.constraint.max_duration)
            # Add some buffer for bucketing inefficiency
            return int(estimated_batches * 0.9)
        else:
            return len(cutset)  # One cut per batch
    
    def get_bucket_stats(self, cutset: CutSet) -> Dict[str, Any]:
        """Get statistics about bucket distribution.
        
        Args:
            cutset: CutSet to analyze
            
        Returns:
            Dictionary with bucket statistics
        """
        if len(cutset) == 0:
            return {}
        
        bucket_fn = self._get_bucket_method()
        values = [bucket_fn(cut) for cut in cutset]
        
        # Calculate bucket boundaries
        sorted_values = sorted(values)
        bucket_size = len(sorted_values) // self.num_buckets
        
        bucket_boundaries = []
        for i in range(self.num_buckets):
            start_idx = i * bucket_size
            if i == self.num_buckets - 1:
                end_idx = len(sorted_values) - 1
            else:
                end_idx = (i + 1) * bucket_size - 1
            
            bucket_boundaries.append((sorted_values[start_idx], sorted_values[end_idx]))
        
        return {
            "method": self.bucket_method,
            "num_buckets": self.num_buckets,
            "boundaries": bucket_boundaries,
            "min_value": min(values),
            "max_value": max(values),
            "mean_value": sum(values) / len(values),
        }
    
    def state_dict(self) -> Dict[str, Any]:
        """Get sampler state."""
        return {
            "epoch": self._epoch,
            "type": "bucketed",
            "num_buckets": self.num_buckets,
            "bucket_method": self.bucket_method,
        }
    
    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Load sampler state."""
        self._epoch = state_dict.get("epoch", 0)
        self.num_buckets = state_dict.get("num_buckets", self.num_buckets)
        self.bucket_method = state_dict.get("bucket_method", self.bucket_method)
