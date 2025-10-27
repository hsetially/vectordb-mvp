import math
import heapq
import random
from typing import Dict, List, Optional, Tuple

import numpy as np

from .config import IndexConfig
from .distance import L2Distance, InnerProductDistance, CosineDistance

class HNSWIndex:
    """
    Implements HNSW algorithm for approximate nearest neighbor search, 
    ref. based on paper by Malkov & Yashunin.
    """
    def __init__(self, dim: int, metric: str = "l2", config: Optional[IndexConfig] = None):
        self.dim = dim
        self.metric_str = metric
        self.config = config or IndexConfig(M=16, ef_construction=200, metric=metric)
        
        self.M = self.config.M
        self.M_max_0 = self.M * 2
        self.M_max = self.M
        self.ef_construction = self.config.ef_construction
        self.ml = 1.0 / math.log(self.M) if self.M > 1 else 0

        self.vectors: Dict[int, np.ndarray] = {}
        self.graphs: List[Dict[int, List[int]]] = []
        
        self.entry_point: Optional[int] = None
        self.max_layer: int = -1

        if metric == "l2":
            self.distance_metric = L2Distance()
        elif metric == "ip":
            self.distance_metric = InnerProductDistance()
        elif metric == "cosine":
            self.distance_metric = CosineDistance()
        else:
            raise ValueError(f"Unknown metric: {metric}")

    def _select_layer(self) -> int:
        """Selects a random layer for a new node using an exponential decay distribution."""
        return int(-math.log(random.uniform(0, 1)) * self.ml)

    def _search_layer(
        self, query: np.ndarray, entry_points: List[int], ef: int, layer_num: int
    ) -> List[Tuple[float, int]]:
        """
        Searches for the `ef` closest neighbors to the query on a specific layer.
        This is the core graph traversal algorithm.
        """
        if not entry_points:
            return []
        candidates = []
        visit_queue = []
        visited = set()

        for ep_id in entry_points:
            dist = self.distance_metric.compute_single(query, self.vectors[ep_id])
            visited.add(ep_id)
            heapq.heappush(visit_queue, (-dist, ep_id))
            heapq.heappush(candidates, (-dist, ep_id))
            if len(candidates) > ef:
                heapq.heappop(candidates)
        
        while visit_queue:
            neg_dist, current_node_id = heapq.heappop(visit_queue)
            if -neg_dist > -candidates[0][0] and len(candidates) == ef:
                break
                
            graph = self.graphs[layer_num]
            neighbors = graph.get(current_node_id, [])
            
            for neighbor_id in neighbors:
                if neighbor_id not in visited:
                    visited.add(neighbor_id)
                    dist = self.distance_metric.compute_single(query, self.vectors[neighbor_id])
                    
                    if -dist > candidates[0][0] or len(candidates) < ef:
                        heapq.heappush(candidates, (-dist, neighbor_id))
                        if len(candidates) > ef:
                            heapq.heappop(candidates)
                        heapq.heappush(visit_queue, (-dist, neighbor_id))
        return sorted([(abs(d), i) for d, i in candidates])

    def _select_neighbors_simple(self, candidates: List[Tuple[float, int]], M: int) -> List[int]:
        """A simple heuristic to select M neighbors from a list of candidates."""
        return [c[1] for c in candidates[:M]]

    def add(self, external_id: int, vector: np.ndarray):
        """Adds a new vector to the HNSW graph."""
        if external_id in self.vectors:
            raise ValueError(f"Vector with ID {external_id} already exists.")
        
        self.vectors[external_id] = vector
        new_node_layer = self._select_layer()
        
        current_ep_id = self.entry_point
        
        if current_ep_id is None:
            self.entry_point = external_id
            self.max_layer = new_node_layer
            self.graphs = [{} for _ in range(new_node_layer + 1)]
            for i in range(new_node_layer + 1):
                self.graphs[i][external_id] = []
            return
        
        if new_node_layer > self.max_layer:
            for _ in range(self.max_layer + 1, new_node_layer + 1):
                self.graphs.append({})
            self.max_layer = new_node_layer
        
        entry_points = [(None, current_ep_id)]
        for layer_num in range(self.max_layer, new_node_layer, -1):
            entry_points = self._search_layer(vector, [ep[1] for ep in entry_points], 1, layer_num)
        
        for layer_num in range(min(new_node_layer, self.max_layer), -1, -1):
            candidates = self._search_layer(vector, [ep[1] for ep in entry_points], self.ef_construction, layer_num)
            
            num_to_connect = self.M_max_0 if layer_num == 0 else self.M_max
            neighbors = self._select_neighbors_simple(candidates, num_to_connect)
            
            self.graphs[layer_num][external_id] = neighbors
            
            for neighbor_id in neighbors:
                neighbor_connections = self.graphs[layer_num][neighbor_id]
                
                limit = self.M_max_0 if layer_num == 0 else self.M_max
                if len(neighbor_connections) < limit:
                    neighbor_connections.append(external_id)
                else:
                    all_candidates = [(self.distance_metric.compute_single(self.vectors[neighbor_id], self.vectors[n_id]), n_id) for n_id in neighbor_connections]
                    all_candidates.append((self.distance_metric.compute_single(self.vectors[neighbor_id], vector), external_id))
                    
                    all_candidates.sort()
                    self.graphs[layer_num][neighbor_id] = [c[1] for c in all_candidates[:limit]]
            entry_points = candidates

        if new_node_layer > self.max_layer:
            self.entry_point = external_id
            self.max_layer = new_node_layer

    def search(self, query: np.ndarray, k: int, ef: int) -> List[Tuple[int, float]]:
        """Searches for the k-nearest neighbors to the query vector."""
        if self.entry_point is None:
            return []
        entry_points = [(None, self.entry_point)]
        for layer_num in range(self.max_layer, 0, -1):
            entry_points = self._search_layer(query, [ep[1] for ep in entry_points], 1, layer_num)

        candidates = self._search_layer(query, [ep[1] for ep in entry_points], ef, 0)
        return candidates[:k]
