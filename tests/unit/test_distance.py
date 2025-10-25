import numpy as np
import time
import pytest
from core.distance import L2Distance, InnerProductDistance, CosineDistance

@pytest.mark.parametrize("metric_cls", [L2Distance, InnerProductDistance, CosineDistance])
def test_batch_vs_single(metric_cls):
    metric = metric_cls()
    query = np.array([1.0, 0.0, 0.0])
    vectors = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 0.0, 0.0],
    ])
    expected = np.array([
        metric.compute_single(query, v) for v in vectors
    ])
    batch = metric.compute(query, vectors)
    np.testing.assert_allclose(batch, expected, rtol=1e-6)


def test_performance():
    query = np.random.rand(128)
    vectors = np.random.rand(10000, 128)
    for Metric in [L2Distance, InnerProductDistance, CosineDistance]:
        metric = Metric()
        start = time.time()
        result = metric.compute(query, vectors)
        duration = (time.time() - start) * 1000
        print(f"{Metric.__name__}: {duration:.2f} ms")
        assert duration < 5.0
        assert result.shape == (10000,)


def test_edge_cases():
    metric = CosineDistance()
    q = np.zeros(128)
    v = np.ones(128)
    assert metric.compute_single(q, v) == 1.0
    assert metric.compute_single(v, q) == 1.0
