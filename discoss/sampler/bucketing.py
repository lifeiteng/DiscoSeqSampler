from lhotse.dataset.sampling import (
    DynamicBucketingSampler as LhotseDynamicBucketingSampler,
)
from dataclasses import asdict, dataclass
from typing import (
    Iterable,
    List,
    Literal,
    Optional,
    Union,
)

from lhotse import Seconds, CutSet
from .constraint import TokenConstraint

__all__ = ["BucketingCutSampler"]


class BucketingCutSampler(LhotseDynamicBucketingSampler):
    def __init__(
        self,
        *cuts: CutSet,
        constraint: TokenConstraint,
        num_buckets: Optional[int] = 10,
        shuffle: bool = False,
        drop_last: bool = False,
        consistent_ids: bool = True,
        bucket_bins: List[int] = None,
        num_cuts_for_bins_estimate: int = 10000,
        buffer_size: int = 20000,
        quadratic_duration: Optional[Seconds] = None,
        world_size: Optional[int] = None,
        rank: Optional[int] = None,
        seed: Union[int, Literal["randomized", "trng"]] = 0,
        sync_buckets: bool = True,
        concurrent: bool = False,
        strict=None,
    ):
        super().__init__(
            *cuts,
            constraint=constraint,
            num_buckets=num_buckets,
            shuffle=shuffle,
            drop_last=drop_last,
            consistent_ids=consistent_ids,
            duration_bins=bucket_bins,
            num_cuts_for_bins_estimate=num_cuts_for_bins_estimate,
            buffer_size=buffer_size,
            quadratic_duration=quadratic_duration,
            world_size=world_size,
            rank=rank,
            seed=seed,
            sync_buckets=sync_buckets,
            concurrent=concurrent,
            strict=strict,
        )
