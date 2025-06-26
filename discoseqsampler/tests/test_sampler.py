"""Test sampler functionality."""

import pytest
import torch
from lhotse import CutSet, MonoCut, AudioSource
from lhotse.dataset.sampling import SamplingConstraint

from discoseqsampler import DiscoSeqSampler, SamplerConfig, SamplingStrategy


def create_dummy_cutset(num_cuts=100, min_duration=1.0, max_duration=10.0):
    """Create a dummy cutset for testing."""
    cuts = []
    
    for i in range(num_cuts):
        duration = min_duration + (max_duration - min_duration) * (i / num_cuts)
        
        cut = MonoCut(
            id=f"cut_{i:04d}",
            start=0.0,
            duration=duration,
            channel=0,
            recording=AudioSource(
                id=f"recording_{i:04d}",
                sources=[],
                sampling_rate=16000,
                num_samples=int(duration * 16000),
                duration=duration,
            ),
        )
        cuts.append(cut)
    
    return CutSet.from_cuts(cuts)


class TestDiscoSeqSampler:
    """Test cases for DiscoSeqSampler."""
    
    def test_sequential_sampling(self):
        """Test sequential sampling strategy."""
        cutset = create_dummy_cutset(50)
        
        config = SamplerConfig(
            strategy=SamplingStrategy.SEQUENTIAL,
            max_duration=20.0,
            shuffle=False,
        )
        
        sampler = DiscoSeqSampler(cutset, config)
        
        # Test iteration
        batches = list(sampler)
        assert len(batches) > 0
        
        # Check that all cuts are included
        total_cuts = sum(len(batch) for batch in batches)
        assert total_cuts == len(cutset)
    
    def test_bucketed_sampling(self):
        """Test bucketed sampling strategy."""
        cutset = create_dummy_cutset(100)
        
        config = SamplerConfig(
            strategy=SamplingStrategy.BUCKETED,
            max_duration=15.0,
            shuffle=True,
            seed=42,
            num_buckets=5,
        )
        
        sampler = DiscoSeqSampler(cutset, config)
        
        # Test iteration
        batches = list(sampler)
        assert len(batches) > 0
        
        # Check bucket statistics
        bucket_stats = sampler.sampler.get_bucket_stats(cutset)
        assert bucket_stats["num_buckets"] == 5
        assert bucket_stats["method"] == "duration"
    
    def test_custom_constraint(self):
        """Test custom sampling constraints."""
        cutset = create_dummy_cutset(50)
        
        constraint = SamplingConstraint(
            max_duration=10.0,
            max_cuts=5,
        )
        
        config = SamplerConfig(
            strategy=SamplingStrategy.SEQUENTIAL,
            max_duration=30.0,  # This will be overridden by constraint
        )
        
        sampler = DiscoSeqSampler(cutset, config, constraint)
        
        # Test that constraint is respected
        for batch in sampler:
            assert len(batch) <= 5
            total_duration = sum(cut.duration for cut in batch)
            assert total_duration <= 10.0
            break  # Just test first batch
    
    def test_distributed_setup(self):
        """Test distributed sampling setup."""
        cutset = create_dummy_cutset(100)
        
        config = SamplerConfig(
            strategy=SamplingStrategy.SEQUENTIAL,
            max_duration=20.0,
            world_size=4,
            rank=1,
        )
        
        sampler = DiscoSeqSampler(cutset, config)
        
        # Test that coordinator is set up correctly
        assert sampler.coordinator.world_size == 4
        assert sampler.coordinator.rank == 1
    
    def test_epoch_setting(self):
        """Test epoch setting for deterministic sampling."""
        cutset = create_dummy_cutset(50)
        
        config = SamplerConfig(
            strategy=SamplingStrategy.SEQUENTIAL,
            max_duration=20.0,
            shuffle=True,
            seed=42,
        )
        
        sampler = DiscoSeqSampler(cutset, config)
        
        # Test epoch setting
        sampler.set_epoch(5)
        assert sampler._epoch == 5
        assert sampler.coordinator._epoch == 5
    
    def test_state_dict(self):
        """Test state saving and loading."""
        cutset = create_dummy_cutset(30)
        
        config = SamplerConfig(
            strategy=SamplingStrategy.BUCKETED,
            max_duration=15.0,
            seed=42,
        )
        
        sampler = DiscoSeqSampler(cutset, config)
        sampler.set_epoch(3)
        
        # Save state
        state = sampler.state_dict()
        assert state["epoch"] == 3
        assert "config" in state
        assert "coordinator_state" in state
        
        # Create new sampler and load state
        new_sampler = DiscoSeqSampler(cutset, config)
        new_sampler.load_state_dict(state)
        
        assert new_sampler._epoch == 3
    
    def test_batch_size_estimation(self):
        """Test batch size estimation."""
        cutset = create_dummy_cutset(100, min_duration=2.0, max_duration=8.0)
        
        config = SamplerConfig(
            strategy=SamplingStrategy.SEQUENTIAL,
            max_duration=30.0,
        )
        
        sampler = DiscoSeqSampler(cutset, config)
        
        # Test length estimation
        estimated_batches = len(sampler)
        actual_batches = len(list(sampler))
        
        # Should be reasonably close (within 50%)
        assert abs(estimated_batches - actual_batches) / actual_batches < 0.5


class TestSamplerCoordinator:
    """Test cases for SamplingCoordinator."""
    
    def test_single_worker(self):
        """Test coordinator with single worker."""
        config = SamplerConfig(max_duration=30.0, world_size=1, rank=0)
        
        from discoseqsampler.core.coordinator import SamplingCoordinator
        coordinator = SamplingCoordinator(config)
        
        cutset = create_dummy_cutset(50)
        worker_cutset = coordinator.get_worker_cutset(cutset)
        
        # Single worker should get all cuts
        assert len(worker_cutset) == len(cutset)
    
    def test_multi_worker_split(self):
        """Test cutset splitting across workers."""
        cutset = create_dummy_cutset(100)
        
        # Simulate 4 workers
        worker_cutsets = []
        for rank in range(4):
            config = SamplerConfig(max_duration=30.0, world_size=4, rank=rank)
            
            from discoseqsampler.core.coordinator import SamplingCoordinator
            coordinator = SamplingCoordinator(config)
            worker_cutset = coordinator.get_worker_cutset(cutset)
            worker_cutsets.append(worker_cutset)
        
        # Check that all cuts are distributed
        total_cuts = sum(len(cutset) for cutset in worker_cutsets)
        assert total_cuts == len(cutset)
        
        # Check that splits are reasonably balanced
        lengths = [len(cutset) for cutset in worker_cutsets]
        assert max(lengths) - min(lengths) <= 1  # At most 1 cut difference
