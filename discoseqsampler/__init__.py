"""
DiscoSeqSampler: Distributed Coordinated Sequenced Sampler
===========================================================

A distributed coordinated sequenced sampler for speech data using Lhotse.

This package provides efficient sampling strategies for distributed training
of speech models, with support for:

- Distributed sampling coordination
- Sequence-aware batching
- Dynamic batch sizing
- Multi-GPU support
- Fault tolerance
"""

__version__ = "0.1.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

from .core.sampler import DiscoSeqSampler
from .core.coordinator import SamplingCoordinator
from .samplers.sequential import SequentialSampler
from .samplers.bucketed import BucketedSampler
from .utils.config import SamplerConfig

__all__ = [
    "DiscoSeqSampler",
    "SamplingCoordinator", 
    "SequentialSampler",
    "BucketedSampler",
    "SamplerConfig",
]
