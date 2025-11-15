import pytest
import numpy as np
from unittest.mock import Mock

from benchmarks.evaluator import BenchmarkEvaluator

@pytest.fixture
def mock_dataset():
    """
    Creates a mock ANNDataset for testing.
    This version is internally consistent.
    """
    dataset = Mock()
    num_queries = 50
    
    dataset.name = "mock_dataset"
    dataset.N = 1000
    dataset.Q = num_queries
    dataset.D = 16
    
    dataset.neighbors = np.array([np.arange(i, i + 5) for i in range(num_queries)])
    dataset.test = np.random.rand(num_queries, 16)
    
    return dataset

@pytest.fixture
def evaluator(mock_dataset):
    """Provides a BenchmarkEvaluator instance."""
    return BenchmarkEvaluator(dataset=mock_dataset)

def test_recall_at_k_perfect_match(evaluator):
    """Test recall when predicted neighbors are a perfect match."""
    predicted = evaluator.ground_truth_ids.tolist()
    
    recall = evaluator.compute_recall_at_k(predicted, k=5)
    assert recall == 1.0

def test_recall_at_k_partial_match(evaluator):
    """Test recall with partial overlap between predicted and ground truth."""
    predicted = evaluator.ground_truth_ids.tolist()
    # For query 0, ground truth is [0, 1, 2, 3, 4]
    # We predict 3 correct neighbors: 0, 1, 2
    predicted[0] = [0, 1, 99, 98, 2] 
    
    recall = evaluator.compute_recall_at_k(predicted, k=5)
    # Total queries = 50. 49 have recall 1.0. One has recall 3/5 = 0.6
    # Expected recall = (49 * 1.0 + 0.6) / 50 = 49.6 / 50 = 0.992
    expected_recall = (49 * 1.0 + 3 / 5) / 50
    assert abs(recall - expected_recall) < 1e-9

def test_recall_at_k_no_match(evaluator):
    """Test recall when there is no overlap."""
    num_queries = evaluator.dataset.Q
    predicted = [[1000, 1001, 1002, 1003, 1004]] * num_queries
    
    recall = evaluator.compute_recall_at_k(predicted, k=5)
    assert recall == 0.0

def test_recall_at_smaller_k(evaluator):
    """Test recall calculation when k is smaller than the list size."""
    predicted = evaluator.ground_truth_ids.tolist()
    # Ground truth for query 0 is [0, 1, 2, 3, 4]
    # We predict [0, 99, 1, ...], so top 3 are [0, 99, 1]
    predicted[0] = [0, 99, 1, 2, 3]
    
    # Recall@3: Ground truth top-3 is {0, 1, 2}. Predicted top-3 is {0, 99, 1}.
    # Intersection is {0, 1}, size = 2. So recall is 2/3.
    recall = evaluator.compute_recall_at_k(predicted, k=3)
    
    # Expected: (49 * 1.0 + 2/3) / 50
    expected_recall = (49 * 1.0 + 2 / 3) / 50
    assert abs(recall - expected_recall) < 1e-9
