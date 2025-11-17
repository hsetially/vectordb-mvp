# VectorBase

Project authors -
Davanapally Itesh (M25AI1046) ;
Harpreet Singh (M25AI1084) ;
Deepak Kumar Sahoo (M25AI1095) ;
Sanidhya Dash - (M25AI1036) ;
Anish Iyer - (M25AI1102)

A fully-functional, crash-resistant vector database implementing the HNSW (Hierarchical Navigable Small World) algorithm with durability guarantees, parallel search, and automatic recovery.

## Features

**HNSW Indexing** - fast approximate nearest neighbor search with configurable recall/speed tradeoffs
**Durability** - write ahead log (WAL) ensures zero data loss on crashes
**Crash Recovery** - automatic state recovery from checkpoints and WAL replay
**Segment Architecture** - immutable sealed segments + growing segment for efficient memory management
**Parallel Search** - multi threaded queries across segments with result merging
**gRPC API** - high performance language agnostic network service
**Columnar Storage** - parquet based binlog with efficient compression


## Installation

### Step 1: Clone the Repository

```bash
git clone <your-repo-url>
cd vectordb-mvp
```

### Step 2: Create Conda Environment

```bash
# Create a new conda environment named 'vectorbase'
conda create -n vectorbase python=3.10

# Activate the environment
conda activate vectorbase
```

On Windows, if the above doesn't work:
```bash
conda create -n vectorbase python=3.10
conda activate vectorbase
```

### Step 3: Install Dependencies

```bash
# Install all required packages
pip install -r requirements.txt
```

### Step 4: Generate Protobuf Files

The gRPC service definitions need to be compiled. Run:

```bash
python -m grpc_tools.protoc -I./proto --python_out=. --grpc_python_out=. proto/common.proto proto/vectordb.proto
```

You should see these files generated in the project root:
- common_pb2.py
- vectordb_pb2.py
- vectordb_pb2_grpc.py

## Quick Start

### 1. Start the Server

```bash
# Basic startup (uses default config)
python main.py

# With custom settings
python main.py --port 9000 --data-dir ./my_data --log-level INFO

### 3. Use the Python Client

In a separate terminal/Python session:

```python
import grpc
import numpy as np
import common_pb2
import vectordb_pb2
import vectordb_pb2_grpc

# Connect to the server
channel = grpc.insecure_channel('localhost:9000')
stub = vectordb_pb2_grpc.VectorDBStub(channel)

# Insert 100 random 128-dimensional vectors
vectors = [
    common_pb2.VectorRecord(
        id=i,
        vector=np.random.rand(128).astype(np.float32),
        metadata={"source": "example"}
    )
    for i in range(100)
]

response = stub.Insert(vectordb_pb2.InsertRequest(vectors=vectors))
print(f"Inserted {response.inserted_count} vectors at LSN {response.lsn}")

# Search for the 10 nearest neighbors to a random query
query = np.random.rand(128).astype(np.float32)
response = stub.Search(vectordb_pb2.SearchRequest(
    query_vector=query,
    k=10,
    ef=100
))

print(f"Found {len(response.results)} results in {response.query_time_ms:.2f}ms:")
for i, result in enumerate(response.results, 1):
    print(f"  {i}. Vector ID: {result.id}, Distance: {result.distance:.6f}")

# Get statistics
stats = stub.GetStats(vectordb_pb2.StatsRequest())
print(f"Database Stats:")
print(f"  Total vectors: {stats.total_vectors}")
print(f"  Num segments: {stats.num_segments}")
print(f"  Memory usage: {stats.memory_mb:.2f} MB")
```

### 4. Configuration

By default, the server uses hardcoded configuration. To use a custom config file:

```bash
# Copy the example config
cp config.yaml.example config.yaml

# Edit as needed and then start server with custom config
python main.py --config config.yaml
```

Example config.yaml:

```yaml
storage:
  data_dir: ./data
  wal_dir: ./data/wal
  binlog_dir: ./data/binlog

index:
  M: 16
  ef_construction: 200
  metric: l2
  dim: 128

segment:
  max_vectors_per_segment: 100000

server:
  grpc_port: 9000

logging:
  level: INFO
```