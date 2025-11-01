import os
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


class BinlogWriter:
    """
    writes sealed segments to parquet files
    """
    def __init__(self, binlog_dir: str):
        self.binlog_dir = Path(binlog_dir)
        self.binlog_dir.mkdir(parents=True, exist_ok=True)

    def write_segment(
        self, 
        segment_id: str, 
        vectors: Dict[int, np.ndarray], 
        metadata: Optional[Dict] = None
    ) -> Path:
        """
        writes a segment's vectors to a parquet file
        
        Args:
            segment_id: Unique identifier for the segment
            vectors: Dictionary mapping external_id -> vector (np.ndarray)
            metadata: Optional metadata dictionary
            
        Returns:
            Path to the written Parquet file
        """
        if not vectors:
            raise ValueError("Cannot write empty segment")
        
        external_ids = list(vectors.keys())
        vector_arrays = [vectors[vid].astype(np.float32) for vid in external_ids]
        
        dim = vector_arrays[0].shape[0]
        
        ids_array = pa.array(external_ids, type=pa.int64())
        
        vectors_list_python = [v.tolist() for v in vector_arrays]
        vectors_array = pa.array(vectors_list_python, type=pa.list_(pa.float32()))
        
        if metadata is None:
            metadata = {}
        
        metadata_list = [metadata.get(vid, "") for vid in external_ids]
        metadata_array = pa.array(metadata_list, type=pa.string())
        
        dim_array = pa.array([dim] * len(external_ids), type=pa.int32())
        
        table = pa.table({
            'external_id': ids_array,
            'vector': vectors_array,
            'metadata': metadata_array,
            'dim': dim_array
        })
        
        output_path = self.binlog_dir / f"segment_{segment_id}.parquet"
        pq.write_table(table, output_path, compression='snappy')
        
        return output_path


class BinlogReader:
    """
    reads sealed segments from parquet files
    """
    def __init__(self, binlog_dir: str):
        self.binlog_dir = Path(binlog_dir)

    def read_segment(self, segment_id: str) -> Dict[int, np.ndarray]:
        """
        reads a segment from parquet and returns a dictionary of vectors
        
        Args:
            segment_id: Unique identifier for the segment
            
        Returns:
            Dictionary mapping external_id -> vector (np.ndarray)
        """
        file_path = self.binlog_dir / f"segment_{segment_id}.parquet"
        
        if not file_path.exists():
            raise FileNotFoundError(f"Segment file not found: {file_path}")
        
        table = pq.read_table(file_path)
        external_ids = table['external_id'].to_pylist()
        vectors = table['vector'].to_pylist()
        
        result = {}
        for ext_id, vec_list in zip(external_ids, vectors):
            result[ext_id] = np.array(vec_list, dtype=np.float32)
        
        return result

    def read_vectors_only(self, segment_id: str) -> np.ndarray:
        """
        Reads only the vector column from a segment, returning a dense (N, D) array.
        
        Useful for building indices without the overhead of dictionary lookups.
        
        Args:
            segment_id: Unique identifier for the segment
            
        Returns:
            NumPy array of shape (N, D) where N is number of vectors
        """
        file_path = self.binlog_dir / f"segment_{segment_id}.parquet"
        
        if not file_path.exists():
            raise FileNotFoundError(f"Segment file not found: {file_path}")
        
        table = pq.read_table(file_path, columns=['vector', 'dim'])
        vectors_list = table['vector'].to_pylist()
        dim = table['dim'][0].as_py()
        
        vectors_array = np.array(vectors_list, dtype=np.float32)
        
        return vectors_array

    def read_metadata(self, segment_id: str) -> Dict[int, str]:
        """
        Reads metadata for a segment.
        
        Args:
            segment_id: Unique identifier for the segment
            
        Returns:
            Dictionary mapping external_id -> metadata string
        """
        file_path = self.binlog_dir / f"segment_{segment_id}.parquet"
        
        if not file_path.exists():
            raise FileNotFoundError(f"Segment file not found: {file_path}")
        
        table = pq.read_table(file_path, columns=['external_id', 'metadata'])
        
        external_ids = table['external_id'].to_pylist()
        metadata_list = table['metadata'].to_pylist()
        
        return {ext_id: meta for ext_id, meta in zip(external_ids, metadata_list)}
