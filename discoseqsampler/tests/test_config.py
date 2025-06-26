"""Test configuration for DiscoSeqSampler."""

import pytest
from discoseqsampler.utils.config import SamplerConfig, SamplingStrategy


class TestSamplerConfig:
    """Test cases for SamplerConfig."""
    
    def test_default_config(self):
        """Test default configuration creation."""
        config = SamplerConfig(max_duration=30.0)
        
        assert config.strategy == SamplingStrategy.SEQUENTIAL
        assert config.max_duration == 30.0
        assert config.world_size == 1
        assert config.rank == 0
        assert config.shuffle is False
        
    def test_config_validation(self):
        """Test configuration validation."""
        # Should raise error if neither batch_size nor max_duration is set
        with pytest.raises(ValueError):
            SamplerConfig()
        
        # Should raise error if rank >= world_size
        with pytest.raises(ValueError):
            SamplerConfig(max_duration=30.0, rank=2, world_size=2)
    
    def test_config_serialization(self):
        """Test configuration serialization and deserialization."""
        config = SamplerConfig(
            strategy=SamplingStrategy.BUCKETED,
            max_duration=25.0,
            world_size=4,
            rank=1,
            shuffle=True,
            seed=42,
            num_buckets=15,
        )
        
        # Convert to dict and back
        config_dict = config.to_dict()
        loaded_config = SamplerConfig.from_dict(config_dict)
        
        assert loaded_config.strategy == config.strategy
        assert loaded_config.max_duration == config.max_duration
        assert loaded_config.world_size == config.world_size
        assert loaded_config.rank == config.rank
        assert loaded_config.shuffle == config.shuffle
        assert loaded_config.seed == config.seed
        assert loaded_config.num_buckets == config.num_buckets
    
    def test_extra_config(self):
        """Test extra configuration parameters."""
        extra = {"custom_param": 123, "model_name": "test"}
        config = SamplerConfig(max_duration=30.0, extra_config=extra)
        
        assert config.extra_config == extra
        
        # Test serialization with extra config
        config_dict = config.to_dict()
        loaded_config = SamplerConfig.from_dict(config_dict)
        
        assert loaded_config.extra_config == extra
