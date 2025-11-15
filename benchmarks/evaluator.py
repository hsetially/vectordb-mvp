import json
import time
from typing import Callable, Dict, List, Any

import numpy as np

from .datasets import ANNDataset


class BenchmarkEvaluator:
    """
    Used to calculate standard ANN benchmark metrics
    """
    def __init__(self, dataset: ANNDataset):
        self.dataset = dataset
        self.ground_truth_ids = self.dataset.neighbors
        self.queries = self.dataset.test

    def compute_recall_at_k(self, predicted_neighbors: List[List[int]], k: int) -> float:
        """
        Computes Recall@k for a set of queries.

        Args:
            predicted_neighbors: list of lists, where each inner list contains the
            predicted neighbor IDs for a query.
            k: The "k" in Recall@k.

        Returns:
            The mean recall value across all queries.
        """
        if len(predicted_neighbors) != len(self.ground_truth_ids):
            raise ValueError("Number of predicted results must match number of queries.")

        total_recall = 0.0
        num_queries = len(predicted_neighbors)

        for i in range(num_queries):
            predicted_set = set(predicted_neighbors[i][:k])
            ground_truth_set = set(self.ground_truth_ids[i][:k])
            
            intersection_size = len(predicted_set.intersection(ground_truth_set))
            
            if len(ground_truth_set) > 0:
                recall = intersection_size / len(ground_truth_set)
                total_recall += recall
        
        return total_recall / num_queries if num_queries > 0 else 0.0

    def benchmark_index(
        self,
        index_builder: Callable[[], Any],
        search_func: Callable[[Any, np.ndarray, int, int], List[int]],
        k_values: List[int],
        ef_values: List[int]
    ) -> Dict[str, Any]:
        """
        Runs a full benchmark suite on a given index.

        Args:
            index_builder: A function that takes no arguments and returns a built index.
            search_func: A function that takes (index, query, k, ef) and returns neighbor IDs.
            k_values: A list of k values for which to compute recall (e.g., [1, 10, 100]).
            ef_values: A list of ef (search parameter) values to sweep.

        Returns:
            A dictionary containing the structured benchmark results.
        """
        print("Building index...")
        start_time = time.perf_counter()
        index = index_builder()
        build_time = time.perf_counter() - start_time
        print(f"Index built in {build_time:.2f} seconds.")
        
        results = {
            'build_time': build_time,
            'dataset': self.dataset.name,
            'n_train': self.dataset.N,
            'n_test': self.dataset.Q,
            'dim': self.dataset.D,
            'ef_sweep': []
        }

        for ef in ef_values:
            print(f"\nBenchmarking with ef = {ef}...")
            latencies = []
            all_predicted_neighbors = []
            
            for query in self.queries:
                start_query_time = time.perf_counter()
                
                predicted_ids = search_func(index, query, max(k_values), ef)
                
                end_query_time = time.perf_counter()
                latencies.append((end_query_time - start_query_time) * 1000)
                all_predicted_neighbors.append(predicted_ids)

            mean_latency = np.mean(latencies)
            qps = 1000 / mean_latency if mean_latency > 0 else float('inf')
            
            ef_results = {
                'ef': ef,
                'latency_mean': mean_latency,
                'latency_p50': np.percentile(latencies, 50),
                'latency_p95': np.percentile(latencies, 95),
                'latency_p99': np.percentile(latencies, 99),
                'qps': qps,
                'recalls': {}
            }
            
            for k in k_values:
                recall_at_k = self.compute_recall_at_k(all_predicted_neighbors, k)
                ef_results['recalls'][f'recall@{k}'] = recall_at_k
                print(f"  Recall@{k}: {recall_at_k:.4f}")

            results['ef_sweep'].append(ef_results)
            
        return results

    def save_results(self, results: dict, output_path: str):
        """Saves the benchmark results to a JSON file."""
        print(f"\nSaving results to {output_path}...")
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        print("Results saved.")

