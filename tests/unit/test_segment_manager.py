import pytest
from unittest.mock import Mock

import numpy as np

from api.query_router import SegmentManager
from core.config import IndexConfig, SegmentConfig

@pytest.fixture
def mock_dependencies():
    """Provides mock dependencies for the SegmentManager."""
    return {
        "wal_writer": Mock(),
        "binlog_writer": Mock(),
        "snapshot_manager": Mock(),
    }

@pytest.fixture
def segment_manager(mock_dependencies):
    """Provides a SegmentManager instance with a small sealing threshold for testing."""
    index_config = IndexConfig(M=8, ef_construction=100, metric="l2")
    segment_config = SegmentConfig(max_vectors_per_segment=2, dim=4)
    
    return SegmentManager(
        dim=segment_config.dim,
        index_config=index_config,
        segment_config=segment_config,
        **mock_dependencies
    )

def test_initial_state(segment_manager):
    """Test that the manager starts with one empty growing segment."""
    assert len(segment_manager.sealed_segments) == 0
    assert segment_manager.growing_segment is not None
    assert len(segment_manager.growing_segment) == 0

def test_insert_into_growing_segment(segment_manager):
    """Test that a single insert goes into the growing segment and doesn't seal."""
    segment_manager.insert(0, np.random.rand(4))
    
    assert len(segment_manager.growing_segment) == 1
    assert len(segment_manager.sealed_segments) == 0

def test_automatic_sealing_at_threshold(segment_manager):
    """Test that the segment is sealed automatically when it reaches capacity."""
    segment_manager.insert(0, np.random.rand(4))
    segment_manager.insert(1, np.random.rand(4))
    
    assert len(segment_manager.sealed_segments) == 1, "A segment should have been sealed"
    assert segment_manager.sealed_segments[0].segment_id is not None
    assert len(segment_manager.sealed_segments[0]) == 2, "The sealed segment should have 2 vectors"
    
    assert segment_manager.growing_segment is not None, "A new growing segment should have been created"
    assert len(segment_manager.growing_segment) == 0, "The new growing segment should be empty"

def test_multi_segment_state_after_multiple_seals(segment_manager):
    """Test the manager's state after several sealing cycles."""
    for i in range(5):
        segment_manager.insert(i, np.random.rand(4))
        
    assert len(segment_manager.sealed_segments) == 2
    assert len(segment_manager.growing_segment) == 1
    
    stats = segment_manager.get_stats()
    assert stats["total_vectors"] == 5
    assert stats["num_segments"] == 3
    assert stats["num_sealed"] == 2

def test_checkpoint_is_triggered(segment_manager):
    """Test that a checkpoint is triggered after a certain number of seals."""
    segment_manager.segment_config.max_vectors_per_segment = 2
    snapshot_manager_mock = segment_manager.snapshot_manager
    
    for i in range(10):
        segment_manager.insert(i, np.random.rand(4))
    
    assert len(segment_manager.sealed_segments) == 5
    snapshot_manager_mock.checkpoint.assert_called_once()
    all_segments = segment_manager.get_all_segments()
    snapshot_manager_mock.checkpoint.assert_called_with(all_segments)