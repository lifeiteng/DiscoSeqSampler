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
    return "examples/audio_cuts.jsonl.gz"


@pytest.fixture()
def audio_cuts():
    """Provide sample data for tests."""
    from lhotse import CutSet

    return CutSet.from_jsonl("examples/audio_cuts.jsonl.gz")


@pytest.fixture()
def image_cuts():
    """Provide sample data for tests."""
    from lhotse import CutSet

    return CutSet.from_jsonl("examples/image_cuts.jsonl.gz")


@pytest.fixture()
def video_cuts():
    """Provide sample data for tests."""
    from lhotse import CutSet

    return CutSet.from_jsonl("examples/video_cuts.jsonl.gz")


@pytest.fixture()
def text_cuts():
    """Provide sample text cuts for tests."""
    raise NotImplementedError(
        "This fixture should be implemented when you add the actual text cuts."
    )

    from lhotse import CutSet

    return CutSet.from_jsonl("examples/text_cuts.jsonl.gz")


@pytest.fixture()
def _mock_sampler():
    """Provide a mock sampler instance for tests (placeholder)."""
    # This will be implemented when you add the actual sampler class
    return
