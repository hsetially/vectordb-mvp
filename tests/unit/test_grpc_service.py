import time
import threading
import pytest
import grpc
import numpy as np

import common_pb2
import vectordb_pb2
import vectordb_pb2_grpc

from api.grpc_service import serve
from api.query_router import SegmentManager, QueryRouter
from storage.wal import WALWriter
from core.config import IndexConfig

SERVER_PORT = 50051
SERVER_ADDRESS = f"localhost:{SERVER_PORT}"

@pytest.fixture(scope="module")
def grpc_server(tmp_path_factory):
    tmp_dir = tmp_path_factory.mktemp("grpc_test_data")
    wal_writer = WALWriter(wal_dir=str(tmp_dir / "wal"))
    config = IndexConfig(M=8, ef_construction=100, metric="l2")
    segment_manager = SegmentManager(dim=4, config=config, wal_writer=wal_writer)
    query_router = QueryRouter(segment_manager)
    
    server_thread = threading.Thread(
        target=serve, args=(SERVER_PORT, segment_manager, wal_writer, query_router), daemon=True
    )
    server_thread.start()
    time.sleep(1)
    yield
    wal_writer.close()

@pytest.fixture(scope="module")
def grpc_stub(grpc_server):
    with grpc.insecure_channel(SERVER_ADDRESS) as channel:
        yield vectordb_pb2_grpc.VectorDBStub(channel)

def test_grpc_channel_creation(grpc_stub):
    assert grpc_stub is not None

def test_insert_rpc(grpc_stub):
    vectors_to_insert = [common_pb2.VectorRecord(id=i, vector=np.random.rand(4)) for i in range(100)]
    request = vectordb_pb2.InsertRequest(vectors=vectors_to_insert)
    response = grpc_stub.Insert(request)
    assert response.inserted_count == 100
    assert len(response.inserted_ids) == 100
    assert response.lsn >= 100

def test_insert_increments_lsn(grpc_stub):
    vec1 = common_pb2.VectorRecord(id=1000, vector=np.random.rand(4))
    resp1 = grpc_stub.Insert(vectordb_pb2.InsertRequest(vectors=[vec1]))
    
    vec2 = common_pb2.VectorRecord(id=1001, vector=np.random.rand(4))
    resp2 = grpc_stub.Insert(vectordb_pb2.InsertRequest(vectors=[vec2]))
    
    assert resp2.lsn > resp1.lsn

def test_insert_invalid_dimension(grpc_stub):
    invalid_vec = common_pb2.VectorRecord(id=2000, vector=[1.0, 2.0])
    request = vectordb_pb2.InsertRequest(vectors=[invalid_vec])
    with pytest.raises(grpc.RpcError) as e:
        grpc_stub.Insert(request)
    assert e.value.code() == grpc.StatusCode.INVALID_ARGUMENT
    assert "Invalid vector dimension" in e.value.details()

