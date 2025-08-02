"""Test configuration and fixtures."""

import pytest


@pytest.fixture()
def sample_data():
    """Provide sample data for tests."""
    return {"test": "data"}


@pytest.fixture()
def _mock_sampler():
    """Provide a mock sampler instance for tests (placeholder)."""
    # This will be implemented when you add the actual sampler class
    return
