from typing import List, Tuple

import faiss
import numpy as np


class FAISSBaseline:
    """
    Wrapper for various FAISS index types to provide a consistent
    build and search interface for benchmarking.
    """
    def __init__(self):
        self.index: faiss.Index = None
        self.dim: int = 0

    def build_flat(self, vectors: np.ndarray):
        """
        Builds a brute-force IndexFlatL2 index
        """
        self.dim = vectors.shape[1]
        self.index = faiss.IndexFlatL2(self.dim)
        self.index.add(vectors)
        print(f"Built IndexFlatL2 with {self.index.ntotal} vectors.")
        return self

    def build_hnsw(self, vectors: np.ndarray, M: int = 32, efConstruction: int = 200):
        """Builds an IndexHNSWFlat index."""
        self.dim = vectors.shape[1]
        index = faiss.IndexHNSWFlat(self.dim, M)
        index.hnsw.efConstruction = efConstruction
        
        print(f"Building HNSW index with M={M}, efConstruction={efConstruction}...")
        index.add(vectors)
        
        self.index = index
        print(f"Built IndexHNSWFlat with {self.index.ntotal} vectors.")
        return self

    def build_ivfpq(
        self,
        vectors: np.ndarray,
        nlist: int = 100,
        m: int = 8,
        nbits: int = 8
    ):
        """
        Builds an IndexIVFPQ (Inverted File with Product Quantization) index.
        """
        self.dim = vectors.shape[1]
        
        quantizer = faiss.IndexFlatL2(self.dim)
        
        index = faiss.IndexIVFPQ(quantizer, self.dim, nlist, m, nbits)
        
        print(f"Training IVFPQ index with nlist={nlist}, m={m}...")
        index.train(vectors)
        
        index.add(vectors)
        
        self.index = index
        print(f"Built IndexIVFPQ with {self.index.ntotal} vectors.")
        return self

    def search(self, query: np.ndarray, k: int, ef: int = 100) -> List[Tuple[int, float]]:
        """
        Performs a search on the built FAISS index.

        Args:
            query: The query vector.
            k: The number of nearest neighbors to retrieve.
            ef: The search-time parameter. For HNSW, this is efSearch. For IVFPQ,
                this is used to derive nprobe.

        Returns:
            A list of (id, distance) tuples.
        """
        if self.index is None:
            raise RuntimeError("Index has not been built yet. Call a build_* method first.")

        if isinstance(self.index, faiss.IndexHNSW):
            self.index.hnsw.efSearch = ef
        elif isinstance(self.index, faiss.IndexIVF):
            self.index.nprobe = max(1, ef // 10)

        query_vector = query.reshape(1, -1)
        
        distances, indices = self.index.search(query_vector, k)
        
        results = []
        for i in range(k):
            if indices[0][i] != -1:
                results.append((indices[0][i], distances[0][i]))
        
        return results

