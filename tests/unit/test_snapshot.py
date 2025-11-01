import sqlite3
import pytest
from unittest.mock import Mock, patch

from storage.snapshot import SnapshotManager
from core.segment import Segment, SegmentState
from core.config import IndexConfig

@pytest.fixture
def mock_dependencies(tmp_path):
    """Fixture to create mock writers/readers and a temporary storage directory."""
    storage_dir = tmp_path / "snapshot_test"
    storage_dir.mkdir()
    
    mock_wal_writer = Mock()
    mock_wal_writer.current_lsn = 100
    mock_wal_writer.current_segment_path = "wal_001.log"
    mock_wal_writer.checkpoint.return_value = 101

    mock_binlog_writer = Mock()
    mock_binlog_reader = Mock()
    
    return {
        "storage_dir": str(storage_dir),
        "wal_writer": mock_wal_writer,
        "binlog_writer": mock_binlog_writer,
        "binlog_reader": mock_binlog_reader
    }

@pytest.fixture
def snapshot_manager(mock_dependencies):
    """Provides a SnapshotManager instance with mock dependencies."""
    return SnapshotManager(**mock_dependencies)

def test_db_initialization(snapshot_manager):
    """Test that the metadata.db file and tables are created on init."""
    db_path = snapshot_manager.db_path
    assert db_path.exists()
    
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='segments';")
        assert cursor.fetchone() is not None
        
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='checkpoints';")
        assert cursor.fetchone() is not None

def test_checkpoint_writes_sealed_segment(snapshot_manager, mock_dependencies):
    """Test that a sealed segment is written to the binlog and its metadata is saved."""
    binlog_writer = mock_dependencies["binlog_writer"]
    
    sealed_segment = Segment(dim=4)
    sealed_segment.state = SegmentState.SEALED
    sealed_segment.vectors = {i: f"vec{i}" for i in range(10)}
    
    snapshot_manager.checkpoint([sealed_segment])
    
    binlog_writer.write_segment.assert_called_once_with(
        sealed_segment.segment_id, sealed_segment.vectors
    )
    
    with sqlite3.connect(snapshot_manager.db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT id, state, num_vectors FROM segments WHERE id=?", (sealed_segment.segment_id,))
        row = cursor.fetchone()
        assert row is not None
        assert row[0] == sealed_segment.segment_id
        assert row[1] == "sealed"
        assert row[2] == 10

def test_checkpoint_ignores_growing_segment(snapshot_manager, mock_dependencies):
    """Test that a segment in the GROWING state is ignored during checkpoint."""
    binlog_writer = mock_dependencies["binlog_writer"]
    
    growing_segment = Segment(dim=4)
    growing_segment.state = SegmentState.GROWING
    snapshot_manager.checkpoint([growing_segment])
    binlog_writer.write_segment.assert_not_called()
    
    with sqlite3.connect(snapshot_manager.db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM segments")
        assert cursor.fetchone()[0] == 0

def test_checkpoint_writes_wal_marker_and_db_record(snapshot_manager, mock_dependencies):
    """Test that a checkpoint record is written to the WAL and the DB."""
    wal_writer = mock_dependencies["wal_writer"]
    
    snapshot_manager.checkpoint([])
    wal_writer.checkpoint.assert_called_once()
    
    with sqlite3.connect(snapshot_manager.db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT lsn, wal_segment_path FROM checkpoints")
        row = cursor.fetchone()
        assert row is not None
        assert row[0] == 101 
        assert row[1] == "wal_001.log"

