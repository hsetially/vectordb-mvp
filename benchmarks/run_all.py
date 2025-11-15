# benchmarks/run_all.py

import json
import os
import sys
from typing import List, Callable, Dict, Any # <-- FIX 1: Add typing imports

import numpy as np

# Ensure the project root is in the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from benchmarks.datasets import ANNDataset
from benchmarks.evaluator import BenchmarkEvaluator
from benchmarks.baselines import FAISSBaseline
from core.hnsw_index import HNSWIndex
from core.config import IndexConfig


# --- Helper functions for our VectorBase HNSWIndex ---

def build_vectorbase(vectors: np.ndarray, M: int, ef_construction: int) -> HNSWIndex:
    """Builds our custom HNSWIndex."""
    dim = vectors.shape[1]
    config = IndexConfig(M=M, ef_construction=ef_construction, metric="l2")
    index = HNSWIndex(dim=dim, config=config)
    
    # Insert vectors one by one
    print(f"Building VectorBase index with {len(vectors)} vectors...")
    for i, vec in enumerate(vectors):
        index.add(i, vec)
        if (i + 1) % 100000 == 0:
            print(f"  ...inserted {i+1} vectors")
        
    return index

def search_vectorbase(index: HNSWIndex, query: np.ndarray, k: int, ef: int) -> List[int]:
    """Search function for our custom HNSWIndex."""
    results = index.search(query, k, ef)
    return [res[0] for res in results]


# --- Benchmark Runners ---

def run_vectorbase_benchmark(dataset: ANNDataset, evaluator: BenchmarkEvaluator) -> dict:
    """Runs the benchmark suite for our VectorBase HNSW implementation."""
    
    configs = [
        ("VectorBase-HNSW-M16", 
         lambda vecs: build_vectorbase(vecs, M=16, ef_construction=200),
         search_vectorbase),
        ("VectorBase-HNSW-M32",
         lambda vecs: build_vectorbase(vecs, M=32, ef_construction=200),
         search_vectorbase),
    ]
    
    results = {}
    for name, builder, searcher in configs:
        print(f"\n--- Benchmarking {name} ---")
        results[name] = evaluator.benchmark_index(
            index_builder=lambda: builder(dataset.train),
            search_func=searcher,
            k_values=[1, 10, 100],
            ef_values=[32, 64, 128, 256]
        )
    
    return results


def run_faiss_benchmark(dataset: ANNDataset, evaluator: BenchmarkEvaluator) -> dict:
    """Runs the benchmark suite for FAISS baselines."""
    
    # <-- FIX 2: Correctly define the builder and searcher lambdas -->
    configs = [
        ("FAISS-FlatL2",
         lambda vecs: FAISSBaseline().build_flat(vecs),
         lambda index, q, k, ef: [res[0] for res in index.search(q, k, ef)]),
        ("FAISS-HNSW-M16",
         lambda vecs: FAISSBaseline().build_hnsw(vecs, M=16, efConstruction=200),
         lambda index, q, k, ef: [res[0] for res in index.search(q, k, ef)]),
        ("FAISS-IVFPQ",
         lambda vecs: FAISSBaseline().build_ivfpq(vecs, nlist=100, m=16),
         lambda index, q, k, ef: [res[0] for res in index.search(q, k, ef)]),
    ]
    
    results = {}
    for name, builder, searcher in configs:
        print(f"\n--- Benchmarking {name} ---")
        results[name] = evaluator.benchmark_index(
            index_builder=lambda: builder(dataset.train),
            search_func=searcher,
            k_values=[1, 10, 100],
            ef_values=[32, 64, 128, 256] # Pass all required arguments
        )
    
    return results


if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)
    
    print("Loading SIFT1M dataset (will download if not present)...")
    sift_dataset = ANNDataset("sift")
    evaluator = BenchmarkEvaluator(sift_dataset)
    
    print("\n=== Running VectorBase Benchmarks ===")
    vb_results = run_vectorbase_benchmark(sift_dataset, evaluator)
    
    print("\n=== Running FAISS Benchmarks ===")
    faiss_results = run_faiss_benchmark(sift_dataset, evaluator)
    
    # Combine and save results
    all_results = {**vb_results, **faiss_results}
    output_file = "results/benchmark_sift1m_results.json"
    
    evaluator.save_results(all_results, output_file)
    
    print(f"\nAll benchmarks complete. Results saved to {output_file}")

