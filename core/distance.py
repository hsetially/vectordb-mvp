import abc
import numpy as np

class DistanceMetric(abc.ABC):
    @abc.abstractmethod
    def compute(self, query: np.ndarray, vectors: np.ndarray) -> np.ndarray:
        """Compute distances from query to each vector."""
        pass

    @abc.abstractmethod
    def compute_single(self, query: np.ndarray, vector: np.ndarray) -> float:
        """Compute distance from query to a single vector."""
        pass

class L2Distance(DistanceMetric):
    def compute(self, query, vectors):
        # query shape: (dim,), vectors shape: (N, dim)
        diff = vectors - query
        return np.linalg.norm(diff, axis=1)
    def compute_single(self, query, vector):
        return np.linalg.norm(query - vector)

class InnerProductDistance(DistanceMetric):
    def compute(self, query, vectors):
        return -np.dot(vectors, query)
    def compute_single(self, query, vector):
        return -np.dot(query, vector)

class CosineDistance(DistanceMetric):
    def compute(self, query, vectors):
        q_norm = np.linalg.norm(query)
        v_norms = np.linalg.norm(vectors, axis=1)
        dot = np.dot(vectors, query)
        denom = v_norms * q_norm
        with np.errstate(divide='ignore', invalid='ignore'):
            cosine_sim = np.where(denom != 0, dot / denom, 0)
        return 1 - cosine_sim
    def compute_single(self, query, vector):
        q_norm = np.linalg.norm(query)
        v_norm = np.linalg.norm(vector)
        if v_norm == 0 or q_norm == 0:
            return 1.0
        return 1 - np.dot(query, vector) / (q_norm * v_norm)
