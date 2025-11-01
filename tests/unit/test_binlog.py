import os
import pytest
import numpy as np

from storage.binlog import BinlogWriter, BinlogReader


@pytest.fixture
def binlog_dir(tmp_path):
    """fixture providing a temporary directory for binlog tests."""
    binlog_dir = tmp_path / "binlog_test"
    binlog_dir.mkdir()
    return binlog_dir


@pytest.fixture
def writer_reader(binlog_dir):
    """fixture providing both writer and reader instances."""
    writer = BinlogWriter(str(binlog_dir))
    reader = BinlogReader(str(binlog_dir))
    return writer, reader


def test_write_segment_basic(writer_reader, binlog_dir):
    """test basic segment writing."""
    writer, _ = writer_reader
    
    vectors = {i: np.random.rand(16).astype(np.float32) for i in range(100)}
    segment_id = "test_segment_001"
    output_path = writer.write_segment(segment_id, vectors)
    
    assert output_path.exists()
    assert output_path.name == "segment_test_segment_001.parquet"
    assert output_path.parent == binlog_dir


def test_read_segment_round_trip(writer_reader):
    """Test that data written can be read back exactly."""
    writer, reader = writer_reader
    
    vectors = {
        0: np.array([1.0, 2.0, 3.0], dtype=np.float32),
        1: np.array([4.0, 5.0, 6.0], dtype=np.float32),
        2: np.array([7.0, 8.0, 9.0], dtype=np.float32)
    }
    segment_id = "test_round_trip"
    
    writer.write_segment(segment_id, vectors)
    recovered = reader.read_segment(segment_id)

    assert set(recovered.keys()) == set(vectors.keys())
    for vec_id, expected_vec in vectors.items():
        np.testing.assert_array_almost_equal(
            recovered[vec_id], expected_vec, 
            err_msg=f"Vector {vec_id} mismatch"
        )


def test_read_segment_with_large_dataset(writer_reader):
    """Test reading and writing a larger dataset."""
    writer, reader = writer_reader
    
    num_vectors = 10000
    dim = 128
    vectors = {i: np.random.rand(dim).astype(np.float32) for i in range(num_vectors)}
    segment_id = "large_segment"
    writer.write_segment(segment_id, vectors)
    recovered = reader.read_segment(segment_id)
    
    assert len(recovered) == num_vectors
    
    for vid in [0, 100, 5000, 9999]:
        np.testing.assert_array_almost_equal(
            recovered[vid], vectors[vid], decimal=5
        )


def test_compression_ratio(writer_reader, binlog_dir):
    """Test that Parquet compression reduces file size significantly."""
    writer, _ = writer_reader
    
    num_vectors = 10000
    dim = 128
    vectors = {i: np.random.rand(dim).astype(np.float32) for i in range(num_vectors)}
    segment_id = "compression_test"
    
    output_path = writer.write_segment(segment_id, vectors)
    compressed_size = output_path.stat().st_size
    
    uncompressed_size = num_vectors * dim * 4
    compression_ratio = compressed_size / uncompressed_size
    
    print(f"\nCompression ratio: {compression_ratio:.2%}")
    print(f"Uncompressed Vector Payload: {uncompressed_size / (1024*1024):.2f} MB")
    print(f"Compressed Parquet File Size (including metadata): {compressed_size / (1024*1024):.2f} MB")
    
    assert compression_ratio < 1.20, \
        f"Compression ratio {compression_ratio:.2%} is unexpectedly high. Check for file bloat."


def test_read_vectors_only(writer_reader):
    """Test reading only vectors as a dense array."""
    writer, reader = writer_reader
    
    num_vectors = 100
    dim = 32
    vectors = {i: np.random.rand(dim).astype(np.float32) for i in range(num_vectors)}
    segment_id = "vectors_only_test"
    
    writer.write_segment(segment_id, vectors)
    vectors_array = reader.read_vectors_only(segment_id)
    
    assert vectors_array.shape == (num_vectors, dim), \
        f"Expected shape ({num_vectors}, {dim}), got {vectors_array.shape}"
    
    for i in range(num_vectors):
        np.testing.assert_array_almost_equal(
            vectors_array[i], vectors[i], decimal=5
        )


def test_metadata_round_trip(writer_reader):
    """Test that metadata is preserved through write/read cycle."""
    writer, reader = writer_reader
    
    vectors = {i: np.random.rand(8).astype(np.float32) for i in range(10)}
    metadata = {
        0: "metadata_for_0",
        1: "metadata_for_1",
        5: "metadata_for_5"
    }
    segment_id = "metadata_test"
    
    writer.write_segment(segment_id, vectors, metadata=metadata)
    recovered_metadata = reader.read_metadata(segment_id)
    
    assert recovered_metadata[0] == "metadata_for_0"
    assert recovered_metadata[1] == "metadata_for_1"
    assert recovered_metadata[5] == "metadata_for_5"
    assert recovered_metadata[2] == ""


def test_empty_segment_raises_error(writer_reader):
    """Test that writing an empty segment raises an error."""
    writer, _ = writer_reader
    
    with pytest.raises(ValueError, match="Cannot write empty segment"):
        writer.write_segment("empty_segment", {})


def test_read_nonexistent_segment_raises_error(writer_reader):
    """Test that reading a non-existent segment raises FileNotFoundError."""
    _, reader = writer_reader
    
    with pytest.raises(FileNotFoundError):
        reader.read_segment("nonexistent_segment")


def test_multiple_segments(writer_reader):
    """Test writing and reading multiple segments."""
    writer, reader = writer_reader
    num_segments = 5
    segment_data = {}
    
    for seg_idx in range(num_segments):
        segment_id = f"segment_{seg_idx}"
        vectors = {i: np.random.rand(16).astype(np.float32) for i in range(100)}
        writer.write_segment(segment_id, vectors)
        segment_data[segment_id] = vectors
    
    for segment_id, expected_vectors in segment_data.items():
        recovered = reader.read_segment(segment_id)
        assert len(recovered) == len(expected_vectors)
        
        for vid, expected_vec in expected_vectors.items():
            np.testing.assert_array_almost_equal(recovered[vid], expected_vec)
