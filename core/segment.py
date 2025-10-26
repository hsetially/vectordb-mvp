import time
import uuid
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np

class HNSWIndex:
    """A placeholder for the actual HNSW index implementation."""
    def __init__(self, dim: int, metric: str, config: 'IndexConfig'):
        self.dim = dim
        self.metric = metric
        self.config = config
        self.data = {}

    def add(self, external_id: int, vector: np.ndarray):
        self.data[external_id] = vector

    def search(self, query: np.ndarray, k: int, ef: int) -> List[Tuple[int, float]]:
        # I am mocking brute-force search, not a real HNSW search.
        # It's just to make the interface work for testing.
        if not self.data:
            return []
        
        ids = list(self.data.keys())
        vectors = np.array(list(self.data.values()))
        
        # using L2 distance for the mock search
        diff = vectors - query
        distances = np.linalg.norm(diff, axis=1)
        
        # get top-k results
        sorted_indices = np.argsort(distances)
        top_k_indices = sorted_indices[:k]
        
        return [(ids[i], distances[i]) for i in top_k_indices]

from .config import IndexConfig
from .distance import L2Distance

class SegmentState(Enum):
    """lifecycle states of a segment"""
    GROWING = "growing"
    SEALING = "sealing"
    SEALED = "sealed"


class Segment:
    """
    Represents a collection of vectors with a defined lifecycle, from a mutable
    'growing' state to an immutable, indexed 'sealed' state.
    """
    def __init__(self, dim: int, metric: str = "l2", config: Optional[IndexConfig] = None):
        self.segment_id: str = str(uuid.uuid4())
        self.dim: int = dim
        self.metric_str: str = metric
        self.config: IndexConfig = config or IndexConfig(M=16, ef_construction=200, metric=metric)

        self.state: SegmentState = SegmentState.GROWING
        self.vectors: Dict[int, np.ndarray] = {}
        self.index: Optional[HNSWIndex] = None

        self.created_at: float = time.time()
        self.sealed_at: Optional[float] = None
        
        # for brute-force search in GROWING state, will be defaulting to L2 distance for now
        self._distance_metric = L2Distance()

    def insert(self, external_id: int, vector: np.ndarray):
        """
        Adds a vector to the segment. Only allowed in the GROWING state.

        Args:
            external_id: The unique identifier for the vector.
            vector: The vector data as a NumPy array.

        Raises:
            ValueError: If the segment is not in the GROWING state.
        """
        if self.state != SegmentState.GROWING:
            raise ValueError(f"Cannot insert into segment {self.segment_id} with state {self.state.value}")
        
        if vector.shape[0] != self.dim:
            raise ValueError(f"Vector dimension mismatch. Expected {self.dim}, got {vector.shape[0]}")
            
        self.vectors[external_id] = vector

    def seal(self):
        """
        Transitions the segment from GROWING to SEALED. This involves building
        the HNSW index over all vectors currently in the segment.
        """
        if self.state != SegmentState.GROWING:
            return
            
        self.state = SegmentState.SEALING
        print(f"Segment {self.segment_id}: Sealing with {len(self.vectors)} vectors...")
        
        # In a future commit, this will call the real HNSW build process.
        self.index = HNSWIndex(self.dim, self.metric_str, self.config)
        for vec_id, vec_data in self.vectors.items():
            self.index.add(vec_id, vec_data)
        
        self.state = SegmentState.SEALED
        self.sealed_at = time.time()
        print(f"Segment {self.segment_id}: Sealed successfully.")

    def search(self, query: np.ndarray, k: int, ef: int = 100) -> List[Tuple[int, float]]:
        """
        Searches for the k-nearest neighbors to the query vector.

        If the segment is SEALED, it uses the HNSW index.
        If the segment is GROWING, it performs a brute-force search for benchmarking and correctness.
        """
        if self.state == SegmentState.SEALED:
            if not self.index:
                return []
            return self.index.search(query, k, ef)
        
        elif self.state == SegmentState.GROWING:
            if not self.vectors:
                return []
            
            ids = list(self.vectors.keys())
            vectors = np.array(list(self.vectors.values()))
            distances = self._distance_metric.compute(query, vectors)

            if len(distances) < k:
                k = len(distances)
            
            sorted_indices = np.argsort(distances)
            top_k_indices = sorted_indices[:k]
            
            return [(ids[i], distances[i]) for i in top_k_indices]
        
        return []

    def get_stats(self) -> Dict:
        """Returns statistics about the segment."""
        return {
            "segment_id": self.segment_id,
            "num_vectors": len(self.vectors),
            "state": self.state.value,
            "created_at": self.created_at,
            "sealed_at": self.sealed_at,
            "dim": self.dim,
            "metric": self.metric_str
        }

    def __len__(self):
        return len(self.vectors)
