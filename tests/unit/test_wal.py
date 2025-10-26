import os
import shutil
import struct
import time
import pytest
import numpy as np

from storage.wal import WALWriter, WALReader, WALEntry, OP_INSERT, OP_DELETE, HEADER_SIZE

@pytest.fixture
def wal_directory(tmp_path):
    """Creates a temporary directory for WAL tests and cleans it up afterward."""
    wal_dir = tmp_path / "wal_test"
    wal_dir.mkdir()
    yield wal_dir
    shutil.rmtree(wal_dir)

def test_wal_entry_serialization_deserialization():
    """Test that a WALEntry can be serialized and deserialized without data loss."""
    entry = WALEntry(lsn=1, op_type=OP_INSERT, payload=b'test_payload')
    serialized = entry.serialize()
    
    deserialized = WALEntry.deserialize(serialized)
    
    assert deserialized.lsn == entry.lsn
    assert deserialized.op_type == entry.op_type
    assert deserialized.payload == entry.payload

def test_wal_writer_append_and_lsn_monotonicity(wal_directory):
    """Test that appending entries increments the LSN correctly."""
    writer = WALWriter(wal_directory)
    
    lsn1 = writer.append(OP_INSERT, b'payload1')
    lsn2 = writer.append(OP_DELETE, b'payload2')
    
    assert lsn1 == 1
    assert lsn2 == 2
    assert writer.current_lsn == 2
    writer.close()

def test_wal_reader_replay(wal_directory):
    """Test that the WALReader can replay all entries in the correct order."""
    writer = WALWriter(wal_directory)
    writer.append(OP_INSERT, b'payload1')
    writer.append(OP_DELETE, b'payload2')
    writer.append(OP_INSERT, b'payload3')
    writer.close()

    reader = WALReader(wal_directory)
    entries = list(reader.replay())
    
    assert len(entries) == 3
    assert entries[0].lsn == 1
    assert entries[0].payload == b'payload1'
    assert entries[1].lsn == 2
    assert entries[1].payload == b'payload2'
    assert entries[2].lsn == 3
    assert entries[2].payload == b'payload3'

def test_insert_delete_payload_format(wal_directory):
    """Test the specific payload formats for insert and delete operations."""
    writer = WALWriter(wal_directory)
    vector = np.array([1.5, 2.5], dtype=np.float32)
    
    # test insert
    writer.insert(external_id=101, vector=vector)
    
    # test delete
    writer.delete(external_id=202)
    writer.close()
    
    reader = WALReader(wal_directory)
    entries = list(reader.replay())
    
    # verify an insert entry
    insert_entry = entries[0]
    assert insert_entry.op_type == OP_INSERT
    ext_id, = struct.unpack('<Q', insert_entry.payload[:8])
    vec_data = np.frombuffer(insert_entry.payload[8:], dtype=np.float32)
    assert ext_id == 101
    np.testing.assert_array_equal(vec_data, vector)
    
    # verify delete entry
    delete_entry = entries[1]
    assert delete_entry.op_type == OP_DELETE
    ext_id, = struct.unpack('<Q', delete_entry.payload)
    assert ext_id == 202

def test_wal_rotation(wal_directory):
    """Test that the WAL file rotates when it exceeds the segment size."""
    writer = WALWriter(wal_directory, segment_size_mb=0.001) # 1KB
    
    for i in range(100):
        writer.append(OP_INSERT, os.urandom(20))
        
    writer.close()
    
    wal_files = sorted(wal_directory.glob('wal_*.log'))
    assert len(wal_files) > 1
    assert wal_files[0].name == 'wal_00000001.log'
    assert wal_files[-1].name != 'wal_00000001.log'

def test_wal_recovers_lsn_on_startup(wal_directory):
    """Test that a new WALWriter correctly resumes LSN from existing logs."""
    writer1 = WALWriter(wal_directory)
    writer1.append(OP_INSERT, b'data')
    assert writer1.current_lsn == 1
    writer1.close()
    
    writer2 = WALWriter(wal_directory)
    assert writer2.current_lsn == 1
    lsn = writer2.append(OP_INSERT, b'more_data')
    assert lsn == 2
    writer2.close()

def test_stress_wal_write_read(wal_directory):
    """Write a large number of entries and verify all are replayed correctly."""
    writer = WALWriter(wal_directory)
    num_entries = 10000
    
    for i in range(num_entries):
        writer.insert(i, np.random.rand(4).astype(np.float32))
        
    writer.close()
    
    reader = WALReader(wal_directory)
    entries = list(reader.replay())
    
    assert len(entries) == num_entries
    for i, entry in enumerate(entries):
        assert entry.lsn == i + 1
