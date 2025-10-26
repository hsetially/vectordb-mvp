import pytest
import h5py
import numpy as np
from pathlib import Path
from unittest.mock import patch

from benchmarks.datasets import ANNDataset, download_dataset

@pytest.fixture
def fake_hdf5_dataset(tmp_path):
    """Creates a fake HDF5 dataset file for testing in a temporary directory."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    file_path = data_dir / "test_dataset.hdf5"
    
    with h5py.File(file_path, 'w') as f:
        f.create_dataset('train', data=np.random.rand(100, 16))
        f.create_dataset('test', data=np.random.rand(10, 16))
        f.create_dataset('neighbors', data=np.random.randint(0, 100, size=(10, 5)))
        f.create_dataset('distances', data=np.random.rand(10, 5))
        
    return "test_dataset", data_dir

def test_anndataset_loads_correctly(fake_hdf5_dataset):
    """Test that ANN Dataset loads data and sets properties correctly."""
    name, data_dir = fake_hdf5_dataset
    
    dataset = ANNDataset(name=name, data_dir=data_dir)

    assert dataset.name == name
    assert dataset.N == 100
    assert dataset.D == 16
    assert dataset.Q == 10

    assert dataset.train.shape == (100, 16)
    assert dataset.test.shape == (10, 16)
    assert dataset.neighbors.shape == (10, 5)
    assert dataset.distances.shape == (10, 5)

def test_anndataset_raises_file_not_found(tmp_path):
    """Test that an error is raised if the dataset file does not exist."""
    non_existent_dir = tmp_path / "non_existent"
    with pytest.raises(FileNotFoundError):
        ANNDataset(name="non_existent_dataset", data_dir=non_existent_dir)

@patch('benchmarks.datasets.download_url')
def test_download_dataset_calls_downloader(mock_download_url, tmp_path):
    """Test that the download function is called with the correct URL."""
    data_dir = tmp_path / "downloads"
    dataset_name = "sift"
    
    download_dataset(name=dataset_name, data_dir=data_dir)
    
    expected_url = "http://ann-benchmarks.com/sift-128-euclidean.hdf5"
    expected_path = data_dir / f"{dataset_name}.hdf5"
    
    mock_download_url.assert_called_once_with(expected_url, expected_path)
