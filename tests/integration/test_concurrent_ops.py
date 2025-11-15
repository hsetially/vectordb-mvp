import threading
import time
import pytest
import grpc
import numpy as np
import subprocess
import shutil
from pathlib import Path

import common_pb2
import vectordb_pb2
import vectordb_pb2_grpc

SERVER_PORT = 50053
SERVER_ADDRESS = f"localhost:{SERVER_PORT}"
TEST_DATA_DIR = Path("./test_concurrent_data")

@pytest.fixture(scope="module")
def server_process():
    """Starts a dedicated server for concurrency tests."""
    if TEST_DATA_DIR.exists():
        shutil.rmtree(TEST_DATA_DIR, ignore_errors=True)
    
    proc = subprocess.Popen([
        "python", "main.py", "--data-dir", str(TEST_DATA_DIR), "--port", str(SERVER_PORT)
    ])
    
    retries = 5
    while retries > 0:
        try:
            with grpc.insecure_channel(SERVER_ADDRESS) as channel:
                if grpc.channel_ready_future(channel).result(timeout=2) is None: break
        except grpc.FutureTimeoutError:
            retries -= 1; time.sleep(2)
    if retries == 0:
        proc.kill(); pytest.fail("Concurrency test server failed to start.")
    
    yield proc
    
    proc.terminate()
    try: proc.wait(timeout=5)
    except subprocess.TimeoutExpired: proc.kill()
    if TEST_DATA_DIR.exists(): shutil.rmtree(TEST_DATA_DIR, ignore_errors=True)

@pytest.fixture(scope="module")
def concurrent_stub(server_process):
    """Provides a stub for the concurrency test."""
    with grpc.insecure_channel(SERVER_ADDRESS) as channel:
        yield vectordb_pb2_grpc.VectorDBStub(channel)

def insert_worker(stub, start_id, num_vectors, dim):
    vectors = [
        common_pb2.VectorRecord(id=i, vector=np.random.rand(dim))
        for i in range(start_id, start_id + num_vectors)
    ]
    try:
        stub.Insert(vectordb_pb2.InsertRequest(vectors=vectors))
    except grpc.RpcError as e:
        print(f"Insert worker caught RpcError: {e.details()}")

def search_worker(stub, num_searches, dim, results_list):
    for _ in range(num_searches):
        try:
            resp = stub.Search(vectordb_pb2.SearchRequest(query_vector=np.random.rand(dim), k=10))
            results_list.append(len(resp.results))
        except grpc.RpcError as e:
            print(f"Search worker caught RpcError: {e.details()}")
            results_list.append(0)
        time.sleep(0.05)

def test_concurrent_inserts_and_searches(concurrent_stub):
    dim = 128
    search_results = []
    
    search_thread = threading.Thread(target=search_worker, args=(concurrent_stub, 20, dim, search_results))
    search_thread.start()
    
    insert_threads = []
    for i in range(4):
        thread = threading.Thread(target=insert_worker, args=(concurrent_stub, 20000 + i * 100, 100, dim))
        thread.start()
        insert_threads.append(thread)

    for t in insert_threads: t.join(timeout=25)
    search_thread.join(timeout=25)
    
    assert all(not t.is_alive() for t in insert_threads)
    assert not search_thread.is_alive()
    assert len(search_results) == 20
    
    stats = concurrent_stub.GetStats(vectordb_pb2.StatsRequest())
    assert stats.total_vectors >= 400

