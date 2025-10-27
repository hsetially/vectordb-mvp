import pytest
import numpy as np

from core.hnsw_index import HNSWIndex
from core.config import IndexConfig

@pytest.fixture
def populated_index():
    """Provides a small populated index for search tests"""
    index = HNSWIndex(dim=2, config=IndexConfig(M=8, ef_construction=100, metric="l2"))
    index.add(0, np.array([1.0, 1.0]))
    index.add(1, np.array([1.1, 1.1]))
    index.add(2, np.array([5.0, 5.0]))
    index.add(3, np.array([5.1, 5.1]))
    index.add(4, np.array([9.0, 9.0]))
    return index

def test_search_empty_index():
    """Test that searching an empty index returns an empty list"""
    index = HNSWIndex(dim=2)
    results = index.search(np.array([1.0, 1.0]), k=5, ef=10)
    assert results == []

def test_search_returns_k_results(populated_index):
    """Test that search returns the correct number of results"""
    query = np.array([0.0, 0.0])
    results = populated_index.search(query, k=3, ef=10)
    assert len(results) == 3

def test_search_exact_match(populated_index):
    """Test that searching for an existing vector finds it with distance 0"""
    query = np.array([5.0, 5.0])
    results = populated_index.search(query, k=1, ef=10)
    
    assert len(results) > 0
    best_match_id, best_match_dist = results[0]
    
    assert best_match_id == 2
    assert best_match_dist < 1e-6

def test_search_correctness(populated_index):
    """Test that search returns the nearest neighbors in the correct order"""
    query = np.array([4.0, 4.0])
    
    results = populated_index.search(query, k=5, ef=20)
    result_ids = [res[0] for res in results]
    
    assert result_ids[0] == 2
    assert result_ids[1] == 3
    assert result_ids[2] == 1
    assert result_ids[3] == 0
    assert result_ids[4] == 4

@pytest.mark.slow
def test_recall_on_large_dataset():
    """
    Test the recall of the HNSW index on a larger random dataset.
    recall@k is defined as the proportion of the true k-nearest neighbors
    that are found in the returned k results.
    """
    dim = 16
    num_vectors = 5000 
    num_queries = 20
    k = 10
    
    index = HNSWIndex(dim=dim, config=IndexConfig(M=16, ef_construction=200, metric="l2"))
    vectors = np.random.rand(num_vectors, dim).astype(np.float32)
    for i in range(num_vectors):
        index.add(i, vectors[i])
        
    queries = np.random.rand(num_queries, dim).astype(np.float32)
    
    total_recall = 0
    
    for i in range(num_queries):
        query = queries[i]
        
        diff = vectors - query
        all_distances = np.linalg.norm(diff, axis=1)
        ground_truth_indices = np.argsort(all_distances)[:k]
        
        hnsw_results = index.search(query, k=k, ef=100)
        hnsw_indices = {res[0] for res in hnsw_results}
        
        correct_found = len(set(ground_truth_indices) & hnsw_indices)
        recall_for_query = correct_found / k
        total_recall += recall_for_query
        
    avg_recall = total_recall / num_queries
    print(f"Average Recall@{k} on {num_vectors} vectors: {avg_recall:.2f}")
    
    assert avg_recall >= 0.9 

