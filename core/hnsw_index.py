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
        self.ml = 1.0 / math.log(self.M) if self.M > 1 else 1.0

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
        if not entry_points or layer_num >= len(self.graphs):
            return []
        candidates = []
        visit_queue = []
        visited = set()

        for ep_id in entry_points:
            if ep_id not in visited:
                dist = self.distance_metric.compute_single(query, self.vectors[ep_id])
                visited.add(ep_id)
                heapq.heappush(visit_queue, (dist, ep_id))
                heapq.heappush(candidates, (-dist, ep_id))
        
        while visit_queue:
            dist, current_node_id = heapq.heappop(visit_queue)
            
            if dist > -candidates[0][0] and len(candidates) >= ef:
                break
                
            graph = self.graphs[layer_num]
            neighbors = graph.get(current_node_id, [])
            
            for neighbor_id in neighbors:
                if neighbor_id not in visited:
                    visited.add(neighbor_id)
                    neighbor_dist = self.distance_metric.compute_single(query, self.vectors[neighbor_id])
                    
                    if len(candidates) < ef or neighbor_dist < -candidates[0][0]:
                        heapq.heappush(candidates, (-neighbor_dist, neighbor_id))
                        if len(candidates) > ef:
                            heapq.heappop(candidates)
                        heapq.heappush(visit_queue, (neighbor_dist, neighbor_id))
        
        result = sorted([(abs(d), i) for d, i in candidates])
        return result

    def _select_neighbors_heuristic(self, candidates: List[Tuple[float, int]], M: int) -> List[int]:
        return [node_id for _, node_id in candidates[:M]]

    def _prune_connections(self, node_id: int, neighbors: List[int], limit: int) -> List[int]:
        if len(neighbors) <= limit:
            return neighbors
        
        distances = [
            (self.distance_metric.compute_single(self.vectors[node_id], self.vectors[n_id]), n_id)
            for n_id in neighbors
        ]
        distances.sort()
        return [n_id for _, n_id in distances[:limit]]

    def add(self, external_id: int, vector: np.ndarray):
        if external_id in self.vectors:
            raise ValueError(f"Vector with ID {external_id} already exists.")
        
        self.vectors[external_id] = vector
        new_node_layer = self._select_layer()

        while len(self.graphs) <= new_node_layer:
            self.graphs.append({})
        
        for layer in range(new_node_layer + 1):
            self.graphs[layer][external_id] = []

        if self.entry_point is None:
            self.entry_point = external_id
            self.max_layer = new_node_layer
            return

        current_max_layer = self.max_layer
        entry_points = [self.entry_point]

        for layer in range(current_max_layer, new_node_layer, -1):
            nearest = self._search_layer(vector, entry_points, 1, layer)
            if not nearest:
                continue
            entry_points = [node_id for _, node_id in nearest]
        
        for layer in range(min(new_node_layer, current_max_layer), -1, -1):
            candidates = self._search_layer(vector, entry_points, self.ef_construction, layer)
            
            m_limit = self.M_max_0 if layer == 0 else self.M_max
            neighbors = self._select_neighbors_heuristic(candidates, m_limit)
            
            self.graphs[layer][external_id] = list(neighbors)
            
            for neighbor_id in neighbors:
                neighbor_connections = self.graphs[layer][neighbor_id]
                neighbor_connections.append(external_id)
                
                if len(neighbor_connections) > m_limit:
                    self.graphs[layer][neighbor_id] = self._prune_connections(
                        neighbor_id, neighbor_connections, m_limit
                    )
            
            entry_points = [node_id for _, node_id in candidates]

        if new_node_layer > self.max_layer:
            self.entry_point = external_id
            self.max_layer = new_node_layer

    def search(self, query: np.ndarray, k: int, ef: int = None) -> List[Tuple[int, float]]:
        if self.entry_point is None:
            return []
        
        if ef is None:
            ef = self.ef_construction
        
        k = min(k, len(self.vectors))
        
        entry_points = [self.entry_point]
        
        for layer in range(self.max_layer, 0, -1):
            nearest = self._search_layer(query, entry_points, 1, layer)
            if not nearest:
                continue
            entry_points = [node_id for _, node_id in nearest]
        
        candidates = self._search_layer(query, entry_points, ef, 0)
        
        return [(node_id, dist) for dist, node_id in candidates[:k]]

