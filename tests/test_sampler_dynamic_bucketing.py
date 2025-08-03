"""Test the discoss package initialization."""

from lhotse.cut import CutSet
from discoss.sampler import DynamicBucketingCutSampler, TokenConstraint
import logging


def _test_sampler(
    cuts_file, rank, world_size, max_cuts: int = 10, drop_last: bool = False
):
    """Test that sampler is accessible and data items."""
    cuts = CutSet.from_jsonl_lazy(cuts_file)
    sampler = DynamicBucketingCutSampler(
        cuts,
        constraint=TokenConstraint(
            max_tokens=4000,
            max_cuts=max_cuts,
            quadratic_length=None,
        ),
        num_buckets=10,
        drop_last=drop_last,
        shuffle=True,
        rank=rank,
        world_size=world_size,
    )

    batches, num_cuts = [], 0
    for batch in sampler:
        assert batch is not None
        num_cuts += len(batch)
        batches.append(batch)

    assert (
        len(cuts) // world_size - num_cuts <= max_cuts
    ), f"{rank=} {world_size=} Expected {len(cuts) // world_size} cuts, got {num_cuts} cuts({max_cuts=})."

    # num_batches = len(batches)
    # logging.info(
    #     f"{rank=} {world_size=}: Number of cuts sampled: {num_cuts} vs {len(cuts) // world_size} expected({num_batches=})."
    # )
    return batches


def test_sampler_dynamic_bucketing_gpu1(cuts_file):
    """Test that sampler is accessible."""
    _test_sampler(cuts_file, 0, 1)


def test_sampler_dynamic_bucketing_gpu2(cuts_file):
    """Test that sampler is accessible."""
    _test_sampler(cuts_file, 0, 2)
    _test_sampler(cuts_file, 1, 2)


def test_sampler_dynamic_bucketing_gpu4(cuts_file):
    """Test that sampler is accessible."""
    _test_sampler(cuts_file, 2, 4)
    _test_sampler(cuts_file, 3, 4)


def test_sampler_dynamic_bucketing_gpu8(cuts_file):
    """Test that sampler is accessible."""
    _test_sampler(cuts_file, 0, 8)
    _test_sampler(cuts_file, 1, 8)
    _test_sampler(cuts_file, 2, 8)
    _test_sampler(cuts_file, 3, 8)
    _test_sampler(cuts_file, 4, 8)
    _test_sampler(cuts_file, 5, 8)
    _test_sampler(cuts_file, 6, 8)
    _test_sampler(cuts_file, 7, 8)
