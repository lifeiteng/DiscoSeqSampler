"""Test configuration and fixtures."""
import logging
import pytest

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


@pytest.fixture()
def cuts_file():
    """Provide sample data for tests."""
    return "example/tests/data/audio_cuts.jsonl.gz"


@pytest.fixture()
def audio_cuts():
    """Provide sample data for tests."""
    from lhotse import CutSet

    return CutSet.from_jsonl("example/tests/data/audio_cuts.jsonl.gz")


@pytest.fixture()
def _mock_sampler():
    """Provide a mock sampler instance for tests (placeholder)."""
    # This will be implemented when you add the actual sampler class
    return
