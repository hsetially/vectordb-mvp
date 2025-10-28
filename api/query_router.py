import heapq
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Tuple
import os

import numpy as np

from core.segment import Segment
from core.config import IndexConfig

class SegmentManager:
    """Manages the collection of segments."""
    def __init__(self, dim: int, config: IndexConfig, wal_writer):
        self.dim = dim
        self.config = config
        self.wal_writer = wal_writer
        
        self.growing_segment: Segment = Segment(dim=dim, config=config)
        self.sealed_segments: List[Segment] = []
        self.lock = threading.Lock()

    def insert(self, external_id: int, vector: np.ndarray):
        with self.lock:
            self.growing_segment.insert(external_id, vector)
            
    def delete(self, external_id: int):
        with self.lock:
            if external_id in self.growing_segment.vectors:
                del self.growing_segment.vectors[external_id]

    def get_all_segments(self) -> List[Segment]:
        with self.lock:
            return self.sealed_segments + [self.growing_segment]
    
    def get_stats(self) -> Dict:
        total_vectors = sum(len(seg) for seg in self.get_all_segments())
        return {
            "total_vectors": total_vectors,
            "num_segments": len(self.sealed_segments) + 1,
            "num_sealed": len(self.sealed_segments)
        }

class QueryRouter:
    """
    Orchestrates search queries across multiple segments, merging the results.
    """
    def __init__(self, segment_manager: SegmentManager):
        self.segment_manager = segment_manager
        self.executor = ThreadPoolExecutor(max_workers=os.cpu_count() or 4)

    def search(self, query: np.ndarray, k: int, ef: int) -> List[Tuple[int, float]]:
        """
        Searches across all managed segments and merges the top-k results.
        """
        segments = self.segment_manager.get_all_segments()
        if not segments:
            return []

        futures = {
            self.executor.submit(seg.search, query, k, ef): seg for seg in segments
        }
        
        all_results = []
        for future in as_completed(futures):
            try:
                segment_results = future.result()
                all_results.extend(segment_results)
            except Exception as e:
                print(f"Segment search failed: {e}")

        return heapq.nsmallest(k, all_results, key=lambda x: x[1])

