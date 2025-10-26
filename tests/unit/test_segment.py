import time
import pytest
import numpy as np

from core.segment import Segment, SegmentState
from core.config import IndexConfig

@pytest.fixture
def segment_config():
    """Provides a default IndexConfig."""
    return IndexConfig(M=16, ef_construction=200, metric="l2")

@pytest.fixture
def growing_segment(segment_config):
    """Provides a new empty segment in the GROWING state."""
    return Segment(dim=4, config=segment_config)

def test_initial_state(growing_segment):
    """Test that a new segment starts in the GROWING state."""
    assert growing_segment.state == SegmentState.GROWING
    assert growing_segment.index is None
    assert growing_segment.sealed_at is None
    assert len(growing_segment) == 0

def test_insert_into_growing_segment(growing_segment):
    """Test that vectors can be inserted into a growing segment."""
    vec1 = np.array([1.0, 2.0, 3.0, 4.0])
    growing_segment.insert(1, vec1)
    
    assert len(growing_segment) == 1
    assert 1 in growing_segment.vectors
    np.testing.assert_array_equal(growing_segment.vectors[1], vec1)

def test_insert_with_wrong_dimension(growing_segment):
    """Test that inserting a vector with the wrong dimension raises an error."""
    vec_wrong_dim = np.array([1.0, 2.0])
    with pytest.raises(ValueError, match="Vector dimension mismatch"):
        growing_segment.insert(1, vec_wrong_dim)

def test_seal_transitions_and_timestamps(growing_segment):
    """Test the seal method for correct state transitions and timestamping."""
    growing_segment.insert(1, np.random.rand(4))
    initial_time = growing_segment.created_at
    
    growing_segment.seal()
    
    assert growing_segment.state == SegmentState.SEALED
    assert growing_segment.sealed_at is not None
    assert growing_segment.sealed_at > initial_time
    assert growing_segment.index is not None

def test_cannot_insert_into_sealed_segment(growing_segment):
    """Test that inserting into a sealed segment raises a ValueError."""
    growing_segment.seal()
    
    assert growing_segment.state == SegmentState.SEALED
    with pytest.raises(ValueError, match="Cannot insert into segment"):
        growing_segment.insert(1, np.random.rand(4))

def test_search_on_growing_segment_brute_force(growing_segment):
    """Test brute-force search on a growing segment."""
    vecs = {
        1: np.array([1.0, 1.0, 1.0, 1.0]),
        2: np.array([9.0, 9.0, 9.0, 9.0]),
        3: np.array([2.0, 2.0, 2.0, 2.0])
    }
    for i, v in vecs.items():
        growing_segment.insert(i, v)
        
    query = np.array([1.1, 1.1, 1.1, 1.1])
    
    results = growing_segment.search(query, k=3)
    
    assert len(results) == 3
    assert results[0][0] == 1
    assert results[1][0] == 3
    assert results[2][0] == 2
    
def test_search_on_sealed_segment(growing_segment):
    """Test that search on a sealed segment calls the index's search method."""
    growing_segment.insert(1, np.array([1.0, 1.0, 1.0, 1.0]))
    growing_segment.seal()
    
    query = np.array([1.1, 1.1, 1.1, 1.1])
    results = growing_segment.search(query, k=1)
    
    assert len(results) == 1
    assert results[0][0] == 1

def test_get_stats(growing_segment):
    """Test the get_stats method."""
    growing_segment.insert(1, np.random.rand(4))
    stats = growing_segment.get_stats()
    
    assert stats["num_vectors"] == 1
    assert stats["state"] == "growing"
    assert "segment_id" in stats

