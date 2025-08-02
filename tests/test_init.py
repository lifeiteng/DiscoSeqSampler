"""Test the discoss package initialization."""

import discoss


def test_version():
    """Test that version is accessible."""
    assert hasattr(discoss, "__version__")
    assert isinstance(discoss.__version__, str)


def test_author():
    """Test that author information is accessible."""
    assert hasattr(discoss, "__author__")
    assert discoss.__author__ == "lifeiteng"


def test_description():
    """Test that description is accessible."""
    assert hasattr(discoss, "__description__")
    assert "Distributed Coordinated Sequence Sampler" in discoss.__description__
