"""Test configuration and fixtures."""

import pytest


@pytest.fixture()
def cuts_file():
    """Provide sample data for tests."""
    return "tests/data/libritts_cuts_dev-clean.jsonl.gz"


@pytest.fixture()
def _mock_sampler():
    """Provide a mock sampler instance for tests (placeholder)."""
    # This will be implemented when you add the actual sampler class
    return
