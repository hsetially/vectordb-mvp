import sqlite3
import struct
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np

from .wal import WALWriter, WALReader, OP_INSERT, OP_DELETE
from .binlog import BinlogWriter, BinlogReader
from core.segment import Segment, SegmentState
from core.config import IndexConfig

class SnapshotManager:
    def __init__(
        self,
        storage_dir: str,
        wal_writer: WALWriter,
        binlog_writer: BinlogWriter,
        binlog_reader: BinlogReader,
        config: IndexConfig,
        dim: int
    ):
        self.storage_dir = Path(storage_dir)
        self.db_path = self.storage_dir / "metadata.db"
        
        self.wal_writer = wal_writer
        self.binlog_writer = binlog_writer
        self.binlog_reader = binlog_reader
        self.config = config
        self.dim = dim
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        self._init_db()

    def _init_db(self):
        """Initializes the SQLite database and tables."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS segments (
                        id TEXT PRIMARY KEY,
                        state TEXT NOT NULL,
                        created_at REAL NOT NULL,
                        sealed_at REAL,
                        num_vectors INTEGER
                    )
                """)
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS checkpoints (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp REAL NOT NULL,
                        lsn INTEGER NOT NULL,
                        wal_segment_path TEXT NOT NULL
                    )
                """)
                conn.commit()
        except sqlite3.Error as e:
            print(f"Database error in _init_db: {e}")
            raise


    def checkpoint(self, segments: List[Segment]):
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                current_lsn = self.wal_writer.current_lsn

                for seg in segments:
                    if seg.state == SegmentState.SEALED:
                        cursor.execute("SELECT 1 FROM segments WHERE id=?", (seg.segment_id,))
                        if cursor.fetchone() is None:
                            print(f"Persisting new sealed segment {seg.segment_id} to binlog")
                            self.binlog_writer.write_segment(seg.segment_id, seg.vectors)
                            
                            cursor.execute("""
                                INSERT OR REPLACE INTO segments 
                                (id, state, created_at, sealed_at, num_vectors)
                                VALUES (?, ?, ?, ?, ?)
                            """, (
                                seg.segment_id,
                                seg.state.value,
                                seg.created_at,
                                seg.sealed_at,
                                len(seg.vectors)
                            ))
                
                checkpoint_lsn = self.wal_writer.checkpoint("GLOBAL_CHECKPOINT", current_lsn)
                
                cursor.execute("""
                    INSERT INTO checkpoints (timestamp, lsn, wal_segment_path)
                    VALUES (?, ?, ?)
                """, (
                    time.time(),
                    checkpoint_lsn,
                    str(self.wal_writer.current_segment_path)
                ))
                
                conn.commit()
        except sqlite3.Error as e:
            print(f"Database error during checkpoint: {e}")
            raise

    def recover(self) -> Tuple[List[Segment], int]:
        print("Starting database recovery...")
        recovered_segments = []
        checkpoint_lsn = 0
        
        if not self.db_path.exists():
            print("No metadata database found. Starting fresh.")
        else:
            try:
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.cursor()
                    cursor.execute("SELECT lsn FROM checkpoints ORDER BY id DESC LIMIT 1")
                    row = cursor.fetchone()
                    if row:
                        checkpoint_lsn = row[0]

                    cursor.execute("SELECT id, created_at, sealed_at FROM segments")
                    for seg_id, created_at, sealed_at in cursor.fetchall():
                        vectors = self.binlog_reader.read_segment(seg_id)
                        seg = Segment(dim=self.dim, config=self.config)
                        seg.segment_id = seg_id
                        seg.vectors = vectors
                        seg.created_at = created_at
                        seg.sealed_at = sealed_at
                        seg.seal()
                        recovered_segments.append(seg)
            except sqlite3.Error as e:
                print(f"Database error during recovery: {e}. Attempting WAL-only recovery.")

        growing_segment = Segment(dim=self.dim, config=self.config)
        
        wal_reader = WALReader(self.wal_writer.wal_dir)
        last_lsn = checkpoint_lsn
        
        for entry in wal_reader.replay(from_lsn=checkpoint_lsn):
            last_lsn = entry.lsn
            if entry.op_type == OP_INSERT:
                ext_id, = struct.unpack('<Q', entry.payload[:8])
                vec_data = np.frombuffer(entry.payload[8:], dtype=np.float32)
                growing_segment.insert(ext_id, vec_data)
            elif entry.op_type == OP_DELETE:
                ext_id, = struct.unpack('<Q', entry.payload)
                if ext_id in growing_segment.vectors:
                    del growing_segment.vectors[ext_id]
                else:
                    for seg in recovered_segments:
                        if ext_id in seg.vectors:
                            del seg.vectors[ext_id]
                            break
        
        if len(growing_segment) > 0:
            recovered_segments.append(growing_segment)
            
        print(f"Recovery complete. Loaded {len(recovered_segments)} segments. Last LSN is {last_lsn}.")
        return recovered_segments, last_lsn
