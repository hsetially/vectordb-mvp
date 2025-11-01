import shutil
import pytest
import numpy as np

from storage.snapshot import SnapshotManager
from storage.wal import WALWriter
from storage.binlog import BinlogWriter, BinlogReader
from core.segment import Segment, SegmentState
from core.config import IndexConfig

@pytest.fixture
def recovery_env(tmp_path):
    """
    Sets up a clean, isolated environment for each recovery test.
    """
    storage_dir = tmp_path / "recovery_test"
    if storage_dir.exists():
        shutil.rmtree(storage_dir)
    storage_dir.mkdir()
    
    wal_dir = storage_dir / "wal"
    binlog_dir = storage_dir / "binlog"
    
    config = IndexConfig(M=8, ef_construction=100, metric="l2")
    dim = 4

    wal_writer = WALWriter(wal_dir=str(wal_dir))
    binlog_writer = BinlogWriter(binlog_dir=str(binlog_dir))
    binlog_reader = BinlogReader(binlog_dir=str(binlog_dir))
    
    snapshot_manager = SnapshotManager(
        storage_dir=str(storage_dir),
        wal_writer=wal_writer,
        binlog_writer=binlog_writer,
        binlog_reader=binlog_reader,
        config=config,
        dim=dim
    )
    
    yield snapshot_manager, wal_writer, binlog_writer, binlog_reader, config, dim
    wal_writer.close()


def test_recovery_with_no_checkpoint(recovery_env):
    snapshot_manager, _, _, _, _, _ = recovery_env
    segments, last_lsn = snapshot_manager.recover()
    assert len(segments) == 0
    assert last_lsn == 0

def test_recovery_from_one_checkpoint(recovery_env):
    snapshot_manager, wal_writer, _, _, config, dim = recovery_env
    
    sealed_seg = Segment(dim=dim, config=config)
    sealed_seg.insert(0, np.array([1,1,1,1], dtype=np.float32))
    sealed_seg.insert(1, np.array([2,2,2,2], dtype=np.float32))
    sealed_seg.seal()
    snapshot_manager.checkpoint([sealed_seg])
    
    wal_writer.close() 

    new_wal_writer = WALWriter(wal_dir=wal_writer.wal_dir)
    new_manager = SnapshotManager(
        storage_dir=snapshot_manager.storage_dir,
        wal_writer=new_wal_writer,
        binlog_writer=snapshot_manager.binlog_writer,
        binlog_reader=snapshot_manager.binlog_reader,
        config=config,
        dim=dim
    )
    
    recovered_segments, last_lsn = new_manager.recover()
    
    assert len(recovered_segments) == 1
    assert last_lsn > 0
    recovered_seg = recovered_segments[0]
    assert recovered_seg.state == SegmentState.SEALED
    assert len(recovered_seg) == 2
    
    new_wal_writer.close()

def test_recovery_with_wal_replay(recovery_env):
    snapshot_manager, wal_writer, _, _, config, dim = recovery_env
    
    sealed_seg = Segment(dim=dim, config=config)
    for i in range(10):
        vec = np.random.rand(dim).astype(np.float32)
        wal_writer.insert(i, vec)
        sealed_seg.insert(i, vec)
    sealed_seg.seal()
    snapshot_manager.checkpoint([sealed_seg])
    
    for i in range(10, 20):
        wal_writer.insert(i, np.random.rand(dim).astype(np.float32))
    
    wal_writer.close()

    new_wal_writer = WALWriter(wal_dir=wal_writer.wal_dir)
    new_manager = SnapshotManager(
        storage_dir=snapshot_manager.storage_dir,
        wal_writer=new_wal_writer,
        binlog_writer=snapshot_manager.binlog_writer,
        binlog_reader=snapshot_manager.binlog_reader,
        config=config,
        dim=dim
    )
    
    recovered_segments, last_lsn = new_manager.recover()
    
    assert last_lsn == 21
    
    sealed_count = sum(1 for s in recovered_segments if s.state == SegmentState.SEALED)
    growing_count = sum(1 for s in recovered_segments if s.state == SegmentState.GROWING)
    
    assert sealed_count == 1
    assert growing_count == 1
    
    total_vectors = sum(len(s) for s in recovered_segments)
    assert total_vectors == 20
    
    new_wal_writer.close()

