from dataclasses import dataclass
from typing import Optional

@dataclass
class IndexConfig:
    M: int
    ef_construction: int
    metric: str

@dataclass
class SegmentConfig:
    max_vectors_per_segment: int
    dim: int
