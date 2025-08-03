"""Test the discoss package initialization."""

from lhotse.cut import CutSet
from discoss.sampler import BucketingCutSampler, TokenConstraint
import logging


def _test_sampler(
    cuts_file,
    rank,
    world_size,
    constraint: TokenConstraint = None,
    max_cuts: int = 10,
    drop_last: bool = False,
):
    """Test that sampler is accessible and data items."""
    cuts = CutSet.from_file(cuts_file)
    sampler = BucketingCutSampler(
        cuts,
        constraint=constraint
        or TokenConstraint(
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

    assert len(cuts) // world_size - num_cuts <= (
        max_cuts * (int(drop_last) + 1)
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


def test_sampler_dynamic_bucketing_gpu4_data_sanity(cuts_file):
    max_tokens, max_cuts = 4000, 10
    constraint = TokenConstraint(
        max_tokens=max_tokens,
        max_cuts=max_cuts,
        quadratic_length=None,
    )

    batches_1 = _test_sampler(cuts_file, 0, 4, constraint=constraint, drop_last=True)
    batches_2 = _test_sampler(cuts_file, 1, 4, constraint=constraint, drop_last=True)
    batches_3 = _test_sampler(cuts_file, 2, 4, constraint=constraint, drop_last=True)
    batches_4 = _test_sampler(cuts_file, 3, 4, constraint=constraint, drop_last=True)

    # 1. check num batches(drop_last=True)
    assert len(batches_1) == len(batches_2) == len(batches_3) == len(batches_4)

    bad_count = 0
    for b1, b2, b3, b4 in zip(batches_1, batches_2, batches_3, batches_4):
        batch_size = [len(b1), len(b2), len(b3), len(b4)]

        # 2. check max_cuts
        assert max(batch_size) <= max_cuts
        if max(batch_size) - min(batch_size) >= 2:
            bad_count += 1

        # 3. check data uniqueness
        # Ensure that cuts in each batch are unique across all batches
        cut_ids = (
            [c.id for c in b1]
            + [c.id for c in b2]
            + [c.id for c in b3]
            + [c.id for c in b4]
        )
        assert len(cut_ids) == len(
            set(cut_ids)
        ), "Cut IDs are not unique across batches."

        # 4. check num tokens
        for b in [b1, b2, b3, b4]:
            lengths = [constraint.measure_length(cut) for cut in b]
            if max(lengths) <= max_tokens:
                assert sum(lengths) <= max_tokens + max(lengths)
            else:
                logging.info(f"{lengths=} sum > {max_tokens=}")

    # 5. check batch size consistency across ranks
    assert (
        bad_count / len(batches_1) < 0.1
    ), f"Batch sizes are not consistent across ranks: {bad_count}/{len(batches_1)}."
