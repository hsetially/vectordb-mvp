import heapq
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Tuple

import numpy as np

from core.segment import Segment
from core.config import IndexConfig, SegmentConfig
from storage.wal import WALWriter
from storage.binlog import BinlogWriter
from storage.snapshot import SnapshotManager

class SegmentManager:
    """
    Manages the lifecycle of vector segments. This class is thread-safe.
    """
    def __init__(
        self,
        dim: int,
        wal_writer: WALWriter,
        binlog_writer: BinlogWriter,
        snapshot_manager: SnapshotManager,
        index_config: IndexConfig,
        segment_config: SegmentConfig
    ):
        self.dim = dim
        self.wal_writer = wal_writer
        self.binlog_writer = binlog_writer
        self.snapshot_manager = snapshot_manager
        self.index_config = index_config
        self.segment_config = segment_config
        
        self.growing_segment: Segment
        self.sealed_segments: List[Segment] = []
        self.lock = threading.RLock()
        
        self._init_growing_segment()

    def _init_growing_segment(self):
        """Creates a new, empty growing segment."""
        print("Initializing new growing segment...")
        self.growing_segment = Segment(dim=self.dim, config=self.index_config)

    def insert(self, external_id: int, vector: np.ndarray):
        """
        Inserts a vector into the current growing segment. If the segment
        reaches its capacity, it is sealed and a new growing segment is created.
        """
        with self.lock:
            self.growing_segment.insert(external_id, vector)
            if len(self.growing_segment) >= self.segment_config.max_vectors_per_segment:
                self._seal_growing_segment()

    def _seal_growing_segment(self):
        """Seals the current growing segment."""
        print(f"Sealing segment {self.growing_segment.segment_id}...")
        self.growing_segment.seal()
        self.sealed_segments.append(self.growing_segment)
        self._init_growing_segment()
        
        if len(self.sealed_segments) % 5 == 0:
            print("Reached checkpoint threshold, triggering checkpoint...")
            all_segments = self.get_all_segments()
            self.snapshot_manager.checkpoint(all_segments)

    def delete(self, external_id: int):
        """Deletes a vector from the growing segment."""
        with self.lock:
            if external_id in self.growing_segment.vectors:
                del self.growing_segment.vectors[external_id]

    def get_all_segments(self) -> List[Segment]:
        """Returns a list of all segments (sealed and growing)."""
        with self.lock:
            return self.sealed_segments + [self.growing_segment]
    
    def get_stats(self) -> Dict:
        """Returns statistics about the segments."""
        with self.lock:
            total_vectors = sum(len(seg) for seg in self.get_all_segments())
            graph_overhead = 2 * self.index_config.M * 4
            memory_per_vector = (self.dim * 4) + graph_overhead
            memory_mb = (total_vectors * memory_per_vector) / (1024 * 1024)

            return {
                "total_vectors": total_vectors,
                "num_segments": len(self.sealed_segments) + 1,
                "num_sealed": len(self.sealed_segments),
                "memory_mb": round(memory_mb, 2)
            }
class QueryRouter:
    """
    Orchestrates search queries across multiple segments, merging the results.
    """
    def __init__(self, segment_manager: SegmentManager):
        self.segment_manager = segment_manager
        self.executor = ThreadPoolExecutor(max_workers=os.cpu_count() or 4)

    def search(self, query: np.ndarray, k: int, ef: int) -> List[Tuple[int, float]]:
        segments = self.segment_manager.get_all_segments()
        if not segments:
            return []

        futures = {self.executor.submit(seg.search, query, k, ef): seg for seg in segments}
        
        all_results = []
        for future in as_completed(futures):
            try:
                segment_results = future.result()
                all_results.extend(segment_results)
            except Exception as e:
                print(f"Segment search failed: {e}")

        return heapq.nsmallest(k, all_results, key=lambda x: x[1])
