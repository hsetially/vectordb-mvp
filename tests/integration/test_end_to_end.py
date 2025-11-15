import subprocess
import time
import shutil
from pathlib import Path

import pytest
import grpc
import numpy as np

import common_pb2
import vectordb_pb2
import vectordb_pb2_grpc

SERVER_PORT = 50051
SERVER_ADDRESS = f"localhost:{SERVER_PORT}"
TEST_DATA_DIR = Path("./test_end_to_end_data")

@pytest.fixture(scope="module")
def server_process():
    """
    A robust fixture that starts the server, waits for it to be ready,
    and handles cleanup.
    """
    if TEST_DATA_DIR.exists():
        shutil.rmtree(TEST_DATA_DIR, ignore_errors=True)
    
    proc = subprocess.Popen([
        "python", "main.py",
        "--data-dir", str(TEST_DATA_DIR),
        "--port", str(SERVER_PORT),
        "--log-level", "INFO",
        "--override", "segment.max_vectors_per_segment=1000"
    ])
    
    retries = 5
    while retries > 0:
        try:
            with grpc.insecure_channel(SERVER_ADDRESS) as channel:
                if grpc.channel_ready_future(channel).result(timeout=1) is None:
                    print("Server is ready.")
                    break
        except grpc.FutureTimeoutError:
            print(f"Waiting for server to start... ({retries} retries left)")
            retries -= 1
            time.sleep(1)
    if retries == 0:
        proc.kill()
        pytest.fail("Server failed to start in time.")
    
    yield proc
    
    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
    
    if TEST_DATA_DIR.exists():
        shutil.rmtree(TEST_DATA_DIR, ignore_errors=True)

@pytest.fixture(scope="module")
def grpc_stub(server_process):
    """Provides a gRPC client stub."""
    with grpc.insecure_channel(SERVER_ADDRESS) as channel:
        yield vectordb_pb2_grpc.VectorDBStub(channel)

def test_stats_reporting(grpc_stub):
    vectors = [
        common_pb2.VectorRecord(id=i+5000, vector=np.random.rand(128).astype(np.float32))
        for i in range(500)
    ]
    grpc_stub.Insert(vectordb_pb2.InsertRequest(vectors=vectors))
    
    stats_resp = grpc_stub.GetStats(vectordb_pb2.StatsRequest())
    assert stats_resp.total_vectors > 0
    assert stats_resp.num_segments > 0
    assert stats_resp.memory_mb >= 0.0
