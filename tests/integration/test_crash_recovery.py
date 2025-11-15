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

SERVER_PORT = 50052
SERVER_ADDRESS = f"localhost:{SERVER_PORT}"
CRASH_TEST_DATA_DIR = Path("./test_crash_data")

def start_server(data_dir, port):
    return subprocess.Popen([
        "python", "main.py", "--data-dir", str(data_dir), "--port", str(port),
        "--log-level", "INFO", "--override", "segment.max_vectors_per_segment=1000"
    ])

def stop_server(proc, graceful=True):
    if graceful:
        proc.terminate()
    else:
        proc.kill()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()

@pytest.fixture
def crash_test_env():
    if CRASH_TEST_DATA_DIR.exists():
        shutil.rmtree(CRASH_TEST_DATA_DIR, ignore_errors=True)
    yield
    if CRASH_TEST_DATA_DIR.exists():
        shutil.rmtree(CRASH_TEST_DATA_DIR, ignore_errors=True)

def test_crash_during_insert_and_wal_replay(crash_test_env):
    dim = 128
    proc = start_server(CRASH_TEST_DATA_DIR, SERVER_PORT)
    time.sleep(3)
    
    with grpc.insecure_channel(SERVER_ADDRESS) as channel:
        grpc.channel_ready_future(channel).result(timeout=5)
        stub = vectordb_pb2_grpc.VectorDBStub(channel)
        vectors = [
            common_pb2.VectorRecord(id=i, vector=np.random.rand(dim))
            for i in range(1000)
        ]
        stub.Insert(vectordb_pb2.InsertRequest(vectors=vectors))
    
    stop_server(proc, graceful=False)
    
    proc = start_server(CRASH_TEST_DATA_DIR, SERVER_PORT)
    time.sleep(4)
    
    with grpc.insecure_channel(SERVER_ADDRESS) as channel:
        grpc.channel_ready_future(channel).result(timeout=5)
        stub = vectordb_pb2_grpc.VectorDBStub(channel)
        stats = stub.GetStats(vectordb_pb2.StatsRequest())
        assert stats.total_vectors >= 1000
    
    stop_server(proc, graceful=True)

def test_recovery_from_checkpoint(crash_test_env):
    dim = 128
    proc = start_server(CRASH_TEST_DATA_DIR, SERVER_PORT)
    time.sleep(3)
    
    with grpc.insecure_channel(SERVER_ADDRESS) as channel:
        grpc.channel_ready_future(channel).result(timeout=5)
        stub = vectordb_pb2_grpc.VectorDBStub(channel)
        vectors = [
            common_pb2.VectorRecord(id=i, vector=np.random.rand(dim)) for i in range(1500)
        ]
        stub.Insert(vectordb_pb2.InsertRequest(vectors=vectors))
        
    stop_server(proc, graceful=True)
    
    proc = start_server(CRASH_TEST_DATA_DIR, SERVER_PORT)
    time.sleep(4)
    
    with grpc.insecure_channel(SERVER_ADDRESS) as channel:
        grpc.channel_ready_future(channel).result(timeout=5)
        stub = vectordb_pb2_grpc.VectorDBStub(channel)
        stats = stub.GetStats(vectordb_pb2.StatsRequest())
        assert stats.total_vectors >= 1500
        assert stats.num_sealed_segments >= 1
    
    stop_server(proc, graceful=True)
