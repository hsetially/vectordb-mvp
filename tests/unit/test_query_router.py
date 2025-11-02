import pytest
import numpy as np

from unittest.mock import Mock, MagicMock
from api.query_router import QueryRouter, SegmentManager

@pytest.fixture
def mock_segment_manager():
    """Provides a mock SegmentManager for testing the QueryRouter."""
    return MagicMock(spec=SegmentManager)

@pytest.fixture
def query_router(mock_segment_manager):
    """Provides a QueryRouter instance with a mock SegmentManager."""
    return QueryRouter(segment_manager=mock_segment_manager)

def test_search_on_single_segment(query_router, mock_segment_manager):
    """Test that the router correctly queries a single segment."""
    mock_segment = Mock()
    mock_segment.search.return_value = [(1, 0.1), (2, 0.2)]
    
    mock_segment_manager.get_all_segments.return_value = [mock_segment]
    
    query = np.random.rand(4)
    results = query_router.search(query, k=2, ef=100)
    
    mock_segment.search.assert_called_once_with(query, 2, 100)
    assert results == [(1, 0.1), (2, 0.2)]

def test_search_merges_results_from_multiple_segments(query_router, mock_segment_manager):
    """Test that results from multiple segments are correctly merged and sorted."""
    segment1 = Mock()
    segment1.search.return_value = [(10, 0.1), (30, 0.3), (50, 0.5)]

    segment2 = Mock()
    segment2.search.return_value = [(20, 0.2), (40, 0.4), (60, 0.6)]

    mock_segment_manager.get_all_segments.return_value = [segment1, segment2]
    
    query = np.random.rand(4)
    results = query_router.search(query, k=4, ef=100)
    
    expected_results = [
        (10, 0.1),
        (20, 0.2),
        (30, 0.3),
        (40, 0.4)
    ]
    assert results == expected_results

def test_search_with_no_segments(query_router, mock_segment_manager):
    """Test that an empty list is returned when there are no segments."""
    mock_segment_manager.get_all_segments.return_value = []
    
    results = query_router.search(np.random.rand(4), k=5, ef=100)
    
    assert results == []

def test_search_handles_fewer_results_than_k(query_router, mock_segment_manager):
    """Test that if the total number of results is less than k, all are returned."""
    segment1 = Mock()
    segment1.search.return_value = [(1, 0.1)]
    
    segment2 = Mock()
    segment2.search.return_value = [(2, 0.2)]

    mock_segment_manager.get_all_segments.return_value = [segment1, segment2]
    results = query_router.search(np.random.rand(4), k=5, ef=100)
    
    assert len(results) == 2
    assert results == [(1, 0.1), (2, 0.2)]

def test_search_handles_failing_segment(query_router, mock_segment_manager):
    """Test that the router continues if one segment's search fails."""
    segment1 = Mock()
    segment1.search.return_value = [(1, 0.1), (3, 0.3)]
    
    failing_segment = Mock()
    failing_segment.search.side_effect = Exception("Simulated search failure")
    
    mock_segment_manager.get_all_segments.return_value = [segment1, failing_segment]
    results = query_router.search(np.random.rand(4), k=5, ef=100)
    
    assert len(results) == 2
    assert results == [(1, 0.1), (3, 0.3)]

