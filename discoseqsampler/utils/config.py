"""Configuration management for DiscoSeqSampler."""

from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field
from enum import Enum


class SamplingStrategy(Enum):
    """Sampling strategies supported by DiscoSeqSampler."""
    SEQUENTIAL = "sequential"
    BUCKETED = "bucketed"
    RANDOM = "random"
    BALANCED = "balanced"


@dataclass
class SamplerConfig:
    """Configuration for DiscoSeqSampler.
    
    Args:
        strategy: Sampling strategy to use
        batch_size: Base batch size per worker
        max_duration: Maximum duration in seconds for batches
        max_cuts: Maximum number of cuts per batch
        world_size: Number of distributed workers
        rank: Rank of current worker
        seed: Random seed for reproducibility
        shuffle: Whether to shuffle data
        drop_last: Whether to drop incomplete batches
        bucket_method: Method for bucketing (duration, num_frames, etc.)
        num_buckets: Number of buckets for bucketed sampling
        buffer_size: Buffer size for sampling
        quadratic_duration: Use quadratic duration for batching
        num_workers: Number of data loading workers
        pin_memory: Pin memory for data loading
        prefetch_factor: Prefetch factor for data loading
    """
    
    # Sampling strategy
    strategy: SamplingStrategy = SamplingStrategy.SEQUENTIAL
    
    # Batch configuration
    batch_size: Optional[int] = None
    max_duration: Optional[float] = None
    max_cuts: Optional[int] = None
    
    # Distributed configuration
    world_size: int = 1
    rank: int = 0
    
    # Randomization
    seed: Optional[int] = None
    shuffle: bool = False
    drop_last: bool = False
    
    # Bucketing configuration
    bucket_method: str = "duration"
    num_buckets: int = 10
    
    # Performance configuration  
    buffer_size: int = 10000
    quadratic_duration: bool = False
    num_workers: int = 0
    pin_memory: bool = False
    prefetch_factor: int = 2
    
    # Additional options
    extra_config: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.batch_size is None and self.max_duration is None:
            raise ValueError("Either batch_size or max_duration must be specified")
            
        if self.rank >= self.world_size:
            raise ValueError(f"Rank {self.rank} must be less than world_size {self.world_size}")
            
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        config_dict = {}
        for key, value in self.__dict__.items():
            if isinstance(value, Enum):
                config_dict[key] = value.value
            else:
                config_dict[key] = value
        return config_dict
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "SamplerConfig":
        """Create config from dictionary."""
        if "strategy" in config_dict and isinstance(config_dict["strategy"], str):
            config_dict["strategy"] = SamplingStrategy(config_dict["strategy"])
        return cls(**config_dict)
