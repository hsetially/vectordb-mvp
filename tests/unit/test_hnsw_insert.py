import time
import pytest
import numpy as np

from core.hnsw_index import HNSWIndex
from core.config import IndexConfig

@pytest.fixture
def index_config():
    """Provides a default IndexConfig for tests."""
    return IndexConfig(M=8, ef_construction=100, metric="l2")

@pytest.fixture
def hnsw_index(index_config):
    """Provides a new, empty HNSW index."""
    return HNSWIndex(dim=4, config=index_config)

def test_insert_first_vector(hnsw_index):
    """Test that the first inserted vector becomes the entry point."""
    vec = np.random.rand(4)
    hnsw_index.add(1, vec)
    
    assert hnsw_index.entry_point == 1
    assert 1 in hnsw_index.vectors
    assert hnsw_index.max_layer >= 0

def test_insert_no_duplicates(hnsw_index):
    """Test that inserting a vector with a duplicate ID raises an error."""
    hnsw_index.add(1, np.random.rand(4))
    with pytest.raises(ValueError, match="already exists"):
        hnsw_index.add(1, np.random.rand(4))

def test_insert_builds_graph(hnsw_index):
    """Test that inserting multiple vectors builds a connected graph."""
    for i in range(100):
        hnsw_index.add(i, np.random.rand(4))
    
    assert len(hnsw_index.vectors) == 100
    assert len(hnsw_index.graphs[0]) == 100
    
    total_connections = sum(len(neighbors) for neighbors in hnsw_index.graphs[0].values())
    assert total_connections > 0

def test_graph_connectivity_limits(hnsw_index):
    """Test that the number of connections per node is within the defined limits."""
    num_vectors = 100
    for i in range(num_vectors):
        hnsw_index.add(i, np.random.rand(4))
    limit_0 = hnsw_index.M_max_0
    for _, neighbors in hnsw_index.graphs[0].items():
        assert len(neighbors) <= limit_0
    
    if hnsw_index.max_layer > 0:
        limit_high = hnsw_index.M_max
        for layer_num in range(1, hnsw_index.max_layer + 1):
            for _, neighbors in hnsw_index.graphs[layer_num].items():
                assert len(neighbors) <= limit_high

def test_layer_selection_distribution(hnsw_index):
    """A simple statistical check on the layer selection distribution."""
    layers = [hnsw_index._select_layer() for _ in range(1000)]
    layer_counts = {i: layers.count(i) for i in set(layers)}
    if 0 in layer_counts and 1 in layer_counts:
        assert layer_counts[1] < layer_counts[0]
    if 1 in layer_counts and 2 in layer_counts:
        assert layer_counts[2] < layer_counts[1]

def test_stress_insert_performance():
    """Stress test: Insert a large number of vectors and check performance."""
    index = HNSWIndex(dim=16, config=IndexConfig(M=16, ef_construction=200, metric="l2"))
    num_vectors = 1000
    
    start_time = time.time()
    for i in range(num_vectors):
        index.add(i, np.random.rand(16))
    end_time = time.time()
    
    duration = end_time - start_time
    print(f"Inserted {num_vectors} vectors in {duration:.2f} seconds.")
    
    assert duration < 2.0 
    assert len(index.vectors) == num_vectors
    assert len(index.graphs[0]) == num_vectors
