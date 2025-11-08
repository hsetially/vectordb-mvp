import pytest
import numpy as np
from pathlib import Path

import common_pb2
import vectordb_pb2
from api.grpc_service import VectorDBServicer
from api.query_router import SegmentManager, QueryRouter
from core.config import IndexConfig, SegmentConfig
from storage.wal import WALWriter
from storage.binlog import BinlogWriter, BinlogReader
from storage.snapshot import SnapshotManager


@pytest.fixture
def full_system_setup(tmp_path):
    """Sets up a complete system without gRPC for direct testing."""
    wal_dir = tmp_path / "wal"
    binlog_dir = tmp_path / "binlog"
    data_dir = tmp_path / "data"
    
    wal_writer = WALWriter(wal_dir=str(wal_dir))
    binlog_writer = BinlogWriter(binlog_dir=str(binlog_dir))
    binlog_reader = BinlogReader(binlog_dir=str(binlog_dir))
    
    index_config = IndexConfig(M=8, ef_construction=100, metric="l2")
    segment_config = SegmentConfig(max_vectors_per_segment=100, dim=4)
    
    snapshot_manager = SnapshotManager(
        storage_dir=str(data_dir),
        wal_writer=wal_writer,
        binlog_writer=binlog_writer,
        binlog_reader=binlog_reader,
        config=index_config,
        dim=4
    )
    
    segment_manager = SegmentManager(
        dim=4,
        wal_writer=wal_writer,
        binlog_writer=binlog_writer,
        snapshot_manager=snapshot_manager,
        index_config=index_config,
        segment_config=segment_config
    )
    
    query_router = QueryRouter(segment_manager=segment_manager)
    
    servicer = VectorDBServicer(segment_manager, wal_writer, query_router)
    
    yield {
        "servicer": servicer,
        "segment_manager": segment_manager,
        "query_router": query_router,
        "snapshot_manager": snapshot_manager,
        "wal_writer": wal_writer
    }
    
    wal_writer.close()


class MockContext:
    """Mock gRPC context for testing without a real server."""
    def set_code(self, code):
        pass
    
    def set_details(self, details):
        pass


def test_insert_through_servicer(full_system_setup):
    """Test inserting vectors through the gRPC servicer."""
    servicer = full_system_setup["servicer"]
    
    vectors = [
        common_pb2.VectorRecord(id=i, vector=np.random.rand(4).astype(np.float32))
        for i in range(50)
    ]
    
    request = vectordb_pb2.InsertRequest(vectors=vectors)
    response = servicer.Insert(request, context=MockContext())
    
    assert response.inserted_count == 50
    assert response.lsn > 0
    print(f"Inserted {response.inserted_count} vectors at LSN {response.lsn}")

def test_stats_through_servicer(full_system_setup):
    """Test getting stats through the gRPC servicer."""
    servicer = full_system_setup["servicer"]
    
    response = servicer.GetStats(vectordb_pb2.StatsRequest(), context=MockContext())
    
    assert response.total_vectors >= 0
    assert response.num_segments > 0
    
    print(f"Stats: {response.total_vectors} vectors in {response.num_segments} segments")


def test_delete_through_servicer(full_system_setup):
    """Test deleting vectors through the gRPC servicer."""
    servicer = full_system_setup["servicer"]
    
    vectors = [
        common_pb2.VectorRecord(id=i, vector=np.random.rand(4).astype(np.float32))
        for i in range(20)
    ]
    servicer.Insert(vectordb_pb2.InsertRequest(vectors=vectors), context=MockContext())
    
    request = vectordb_pb2.DeleteRequest(ids=[0, 1, 2])
    response = servicer.Delete(request, context=MockContext())
    
    assert response.deleted_count == 3
    print(f"Deleted {response.deleted_count} vectors")


def test_data_persistence(full_system_setup):
    """Test that data persists across recovery."""
    servicer = full_system_setup["servicer"]
    snapshot_manager = full_system_setup["snapshot_manager"]
    segment_manager = full_system_setup["segment_manager"]
    
    vectors = [
        common_pb2.VectorRecord(id=i, vector=np.random.rand(4).astype(np.float32))
        for i in range(50)
    ]
    insert_response = servicer.Insert(vectordb_pb2.InsertRequest(vectors=vectors), context=MockContext())
    initial_lsn = insert_response.lsn
    
    all_segments = segment_manager.get_all_segments()
    snapshot_manager.checkpoint(all_segments)
    
    assert snapshot_manager.db_path.exists()
    print(f"Checkpoint created at {snapshot_manager.db_path}")
    
    recovered_segments, recovered_lsn = snapshot_manager.recover()
    assert recovered_lsn >= initial_lsn
    print(f"Recovery successful: recovered {len(recovered_segments)} segments at LSN {recovered_lsn}")
