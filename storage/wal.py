import os
import struct
import time
import threading
from pathlib import Path
from dataclasses import dataclass
from typing import Iterator, List, Optional

import numpy as np

# Define operation types
OP_INSERT = 0x01
OP_DELETE = 0x02
OP_CHECKPOINT = 0x03

HEADER_FORMAT = '<QBI'
HEADER_SIZE = struct.calcsize(HEADER_FORMAT)


@dataclass
class WALEntry:
    """Represents a single entry in the Write-Ahead Log."""
    lsn: int
    op_type: int
    payload: bytes
    timestamp: float = 0.0

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = time.time()

    def serialize(self) -> bytes:
        """Packs the WAL entry into a binary format: [Header][Payload]"""
        header = struct.pack(HEADER_FORMAT, self.lsn, self.op_type, len(self.payload))
        return header + self.payload

    @staticmethod
    def deserialize(data: bytes) -> 'WALEntry':
        """Unpacks a binary WAL entry into a WALEntry object."""
        lsn, op_type, payload_len = struct.unpack(HEADER_FORMAT, data[:HEADER_SIZE])
        payload = data[HEADER_SIZE : HEADER_SIZE + payload_len]
        return WALEntry(lsn=lsn, op_type=op_type, payload=payload)


class WALWriter:
    """
    Handles writing entries to the write ahead Log in a durable,
    thread-safe, and segmented manner.
    """
    def __init__(self, wal_dir: str, segment_size_mb: int = 100):
        self.wal_dir = Path(wal_dir)
        self.wal_dir.mkdir(parents=True, exist_ok=True)
        self.segment_size_bytes = segment_size_mb * 1024 * 1024

        self.lock = threading.Lock()
        self.current_lsn = self._load_last_lsn()
        self._open_latest_segment()

    def _open_latest_segment(self):
        """Opens the latest WAL segment file for appending or creates a new one."""
        wal_files = sorted(self.wal_dir.glob('wal_*.log'))
        if not wal_files:
            self.current_segment_path = self.wal_dir / 'wal_00000001.log'
        else:
            self.current_segment_path = wal_files[-1]
        
        self.current_file = open(self.current_segment_path, 'ab')

    def append(self, op_type: int, payload: bytes) -> int:
        """
        Appends a new entry to the WAL in a thread-safe manner.

        Returns:
            The Log Sequence Number (LSN) assigned to this entry.
        """
        with self.lock:
            self.current_lsn += 1
            entry = WALEntry(self.current_lsn, op_type, payload)
            data = entry.serialize()

            self.current_file.write(data)
            self.current_file.flush()
            os.fsync(self.current_file.fileno())

            if self.current_file.tell() > self.segment_size_bytes:
                self._rotate()
            
            return self.current_lsn

    def insert(self, external_id: int, vector: np.ndarray) -> int:
        """Logs an INSERT operation."""
        payload = struct.pack('<Q', external_id) + vector.astype(np.float32).tobytes()
        return self.append(OP_INSERT, payload)

    def delete(self, external_id: int) -> int:
        """Logs a DELETE operation."""
        payload = struct.pack('<Q', external_id)
        return self.append(OP_DELETE, payload)

    def checkpoint(self, segment_id: str, lsn: int) -> int:
        """Logs a CHECKPOINT operation."""
        payload = segment_id.encode('utf-8').ljust(36, b'\x00') + struct.pack('<Q', lsn)
        return self.append(OP_CHECKPOINT, payload)

    def _rotate(self):
        """Closes the current WAL file and opens a new, incremented one."""
        self.current_file.close()
        
        last_segment_num = int(self.current_segment_path.stem.split('_')[1])
        new_segment_num = last_segment_num + 1
        self.current_segment_path = self.wal_dir / f'wal_{new_segment_num:08d}.log'
        
        self.current_file = open(self.current_segment_path, 'ab')

    def _load_last_lsn(self) -> int:
        """Scans all WAL files to find the highest LSN to resume from."""
        max_lsn = 0
        reader = WALReader(self.wal_dir)
        for entry in reader.replay():
            max_lsn = max(max_lsn, entry.lsn)
        return max_lsn

    def close(self):
        """Closes the currently open WAL file."""
        if not self.current_file.closed:
            self.current_file.close()


class WALReader:
    """Handles reading entries from a sequence of WAL segment files."""
    def __init__(self, wal_dir: str):
        self.wal_dir = Path(wal_dir)

    def replay(self, from_lsn: int = 0) -> Iterator[WALEntry]:
        """
        Reads and yields all WAL entries from a given LSN, in order.
        Used for state recovery on startup.
        """
        wal_files = sorted(self.wal_dir.glob('wal_*.log'))
        
        for wal_file in wal_files:
            with open(wal_file, 'rb') as f:
                while True:
                    header_data = f.read(HEADER_SIZE)
                    if len(header_data) < HEADER_SIZE:
                        break

                    _, _, payload_len = struct.unpack(HEADER_FORMAT, header_data)
                    payload_data = f.read(payload_len)

                    if len(payload_data) < payload_len:
                        break

                    entry = WALEntry.deserialize(header_data + payload_data)
                    if entry.lsn > from_lsn:
                        yield entry
