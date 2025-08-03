"""Test the discoss package initialization."""

from lhotse.cut import CutSet
from discoss.sampler import SimpleCutSampler, TokenConstraint
import logging


def _test_sampler(cuts, rank, world_size, max_cuts: int = 10):
    """Test that sampler is accessible and data items."""
    sampler = SimpleCutSampler(
        cuts=cuts,
        constraint=TokenConstraint(
            max_tokens=1000,
            max_cuts=max_cuts,
            quadratic_length=None,
        ),
        rank=rank,
        world_size=world_size,
    )

    num_cuts = 0
    for batch in sampler:
        assert batch is not None
        num_cuts += len(batch)

    assert (
        len(cuts) // world_size - num_cuts <= max_cuts
    ), f"{rank=} {world_size=} Expected {len(cuts) // world_size} cuts, got {num_cuts} cuts({max_cuts=})."
    # logging.info(f"{rank=} {world_size=}: Number of cuts sampled: {num_cuts} vs {len(cuts) // world_size} expected.")


def test_sampler_simple_gpu1(audio_cuts):
    """Test that sampler is accessible."""
    _test_sampler(audio_cuts, 0, 1)


def test_sampler_simple_gpu2(audio_cuts):
    """Test that sampler is accessible."""
    _test_sampler(audio_cuts, 0, 2)
    _test_sampler(audio_cuts, 1, 2)


def test_sampler_simple_gpu4(audio_cuts):
    """Test that sampler is accessible."""
    _test_sampler(audio_cuts, 2, 4)
    _test_sampler(audio_cuts, 3, 4)


def test_sampler_simple_gpu8(audio_cuts):
    """Test that sampler is accessible."""
    _test_sampler(audio_cuts, 0, 8)
    _test_sampler(audio_cuts, 1, 8)
    _test_sampler(audio_cuts, 2, 8)
    _test_sampler(audio_cuts, 3, 8)
    _test_sampler(audio_cuts, 4, 8)
    _test_sampler(audio_cuts, 5, 8)
    _test_sampler(audio_cuts, 6, 8)
    _test_sampler(audio_cuts, 7, 8)
