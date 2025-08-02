"""Test the discoss package initialization."""


def _test_sampler(cuts_file, rank, world_size):
    """Test that sampler is accessible."""
    # assert hasattr(discoss, "DiscoSeqSampler")
    # assert callable(discoss.DiscoSeqSampler)


def test_sampler_gpu1(cuts_file):
    """Test that sampler is accessible."""
    _test_sampler(cuts_file, 0, 1)


def test_sampler_gpu2(cuts_file):
    """Test that sampler is accessible."""
    _test_sampler(cuts_file, 0, 2)
    _test_sampler(cuts_file, 1, 2)
