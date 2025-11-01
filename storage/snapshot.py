import sqlite3
import time
from pathlib import Path
from typing import List

from .wal import WALWriter
from .binlog import BinlogWriter, BinlogReader
from core.segment import Segment, SegmentState

class SnapshotManager:
    """
    Coordinates the creation of durable snapshots by persisting sealed segments
    to a columnar format (binlog) and recording metadata in a SQLite database.
    """
    def __init__(
        self,
        storage_dir: str,
        wal_writer: WALWriter,
        binlog_writer: BinlogWriter,
        binlog_reader: BinlogReader
    ):
        self.storage_dir = Path(storage_dir)
        self.db_path = self.storage_dir / "metadata.db"
        
        self.wal_writer = wal_writer
        self.binlog_writer = binlog_writer
        self.binlog_reader = binlog_reader
        
        self._init_db()

    def _init_db(self):
        """
        Initializes the SQLite database and creates the necessary tables
        for storing checkpoint and segment metadata if they don't exist.
        """
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
        """
        Creates a new checkpoint.
        
        This process involves:
        1. Persisting any newly sealed segments to the binlog (Parquet).
        2. Recording the metadata for these segments in the SQLite database.
        3. Writing a special checkpoint marker to the WAL.
        4. Recording the checkpoint event itself in the SQLite database.
        
        Args:
            segments: The current list of all segments (growing and sealed).
        """
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
                print(f"Checkpoint created successfully at LSN {checkpoint_lsn}.")

        except sqlite3.Error as e:
            print(f"Database error during checkpoint: {e}")
            raise
