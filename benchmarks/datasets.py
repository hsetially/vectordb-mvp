import h5py
import numpy as np
from pathlib import Path
from typing import Tuple
import urllib.request
from tqdm import tqdm

class TqdmUpTo(tqdm):
    """Provides `update_to(block_num, block_size, total_size)` hook for urlretrieve."""
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)

def download_url(url, output_path):
    """Download a file with a progress bar."""
    with TqdmUpTo(unit='B', unit_scale=True, unit_divisor=1024, miniters=1, desc=Path(output_path).name) as t:
        urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)

class ANNDataset:
    """
    Handles loading of standard ANN benchmark datasets stored in HDF5 format.
    """
    def __init__(self, name: str, data_dir: str = "./data"):
        self.name = name
        self.data_dir = Path(data_dir)
        self.data_path = self.data_dir / f"{name}.hdf5"

        if not self.data_path.exists():
            raise FileNotFoundError(
                f"dataset not found at {self.data_path}"
                f"run 'python scripts/download_datasets.py --dataset {name}'"
            )
        
        self._load()

    def _load(self):
        """Loads the dataset from the HDF5 file into memory."""
        with h5py.File(self.data_path, 'r') as f:
            self.train: np.ndarray = np.array(f['train'])
            self.test: np.ndarray = np.array(f['test'])
            self.neighbors: np.ndarray = np.array(f['neighbors'])
            self.distances: np.ndarray = np.array(f['distances'])
        
        self.N, self.D = self.train.shape
        self.Q = self.test.shape[0]
        
        print(f"Successfully loaded dataset: {self.name}")
        print(f"  Training vectors: {self.N} x {self.D}")
        print(f"  Query vectors:    {self.Q} x {self.D}")
        print(f"  Ground truth:     {self.neighbors.shape[0]} x {self.neighbors.shape[1]}")

    def get_train(self) -> np.ndarray:
        """Returns the training (base) vectors."""
        return self.train

    def get_test(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Returns the test queries, ground truth neighbor IDs, and distances."""
        return self.test, self.neighbors, self.distances

def download_dataset(name: str, data_dir: str):
    """
    Downloads a specified ANN benchmark dataset.
    
    Args:
        name: name of the dataset
        data_dir: dir to save the dataset file.
    """
    base_url = "http://ann-benchmarks.com/"
    dataset_files = {
        "sift": "sift-128-euclidean.hdf5",
        "glove": "glove-100-angular.hdf5",
        "mnist": "mnist-784-euclidean.hdf5"
    }

    if name not in dataset_files:
        raise ValueError(f"Unknown dataset '{name}'. Available: {list(dataset_files.keys())}")
    
    filename = dataset_files[name]
    url = base_url + filename
    
    output_dir = Path(data_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{name}.hdf5"
    
    if output_path.exists():
        print(f"Dataset '{name}' already exists at {output_path}. Skipping download.")
        return

    print(f"Downloading '{name}' dataset from {url} to {output_path}...")
    try:
        download_url(url, output_path)
        print(f"\nDownload complete for '{name}'.")
    except Exception as e:
        print(f"\nError downloading dataset: {e}")
        if output_path.exists():
            output_path.unlink()
